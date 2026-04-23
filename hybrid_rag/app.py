from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llamaindex_shared import (  # noqa: E402
    ChatUiConfig,
    add_corpus_documents,
    build_chat_ui_tabs,
    build_cluster_server_urls,
    build_query_engine,
    collect_sources,
    configure_models,
    create_job,
    delete_corpus_documents,
    ensure_vector_index,
    get_job,
    list_corpus_documents,
    load_shared_config,
    post_json,
    render_chat_ui,
    should_apply_similarity_threshold,
    wait_for_job,
)
from llamaindex_shared.benchmark_runtime import parse_benchmark_profile_payload, runtime_overrides_signature  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".qdrant_state.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
CURRENT_RAG_ID = "hybrid"
RAG_LOCK = Lock()
ADMIN_LOCK = Lock()


@lru_cache(maxsize=1)
# Tạo HTML giao diện dùng chung cho tab Hybrid RAG.
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id=CURRENT_RAG_ID,
            page_title="NTU Hybrid RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description=(
                "Hybrid RAG kết hợp dense và sparse retrieval. Đây là giao diện gốc "
                "được dùng chung cho Baseline, Hybrid và Graph RAG."
            ),
            header_badge="Hybrid RAG",
            header_subtitle="Hybrid retrieval | Qdrant | LlamaIndex",
            assistant_label="Hybrid NTU Bot",
            empty_title="Hybrid RAG sẵn sàng",
            empty_description=(
                "Bạn có thể giữ nguyên bố cục này và chuyển tab để đổi sang Baseline "
                "hoặc Graph RAG nhằm so sánh câu trả lời."
            ),
            placeholder="Nhập câu hỏi tuyển sinh của bạn...",
            composer_hint="Ví dụ: Điểm chuẩn ngành CNTT năm 2025?",
            loading_message="Đang truy xuất tài liệu bằng hybrid retrieval...",
            ready_message="Đã trả lời xong.",
            storage_key="ntu_fusion_hybrid_sessions",
            new_chat_label="Cuộc trò chuyện mới",
            manage_data_label="Thêm dữ liệu",
            send_button_label="Gửi",
            initial_title="Cuộc trò chuyện mới",
            history_label="Lịch sử",
            empty_history_text="Chưa có tin nhắn",
            continue_history_text="Tiếp tục hội thoại",
            sending_error_prefix="Không thể xử lý câu hỏi.",
            data_modal_title="Quản lý dữ liệu",
            suggestions=[
                "Điểm chuẩn năm 2025 là bao nhiêu?",
                "Ngành nào có nhiều học bổng nhất?",
                "Học phí một năm là bao nhiêu?",
                "Hồ sơ đăng ký cần những gì?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


@lru_cache(maxsize=16)
# Tải config, model và query engine rồi cache lại theo cấu hình runtime.
def get_resources(profile_name: str = "default", overrides_key: str = "{}"):
    config = load_shared_config(collection_name="ntu_hybrid_llamaindex", overrides=json.loads(overrides_key))
    configure_models(config)
    index = ensure_vector_index(
        config,
        state_path=STATE_PATH,
        enable_hybrid=True,
    )
    query_engine = build_query_engine(index, config, enable_hybrid=True)
    return config, query_engine


# Trả lời một câu hỏi bằng Hybrid RAG và gom danh sách nguồn liên quan.
def answer_query(query: str, *, profile_name: str = "default", runtime_overrides: dict | None = None) -> dict:
    config, query_engine = get_resources(profile_name, runtime_overrides_signature(runtime_overrides))
    response = query_engine.query(query)
    sources = collect_sources(response, limit=config.retrieval_top_n)

    if not sources:
        answer = config.query_refusal_response
    else:
        scores = [float(item["score"]) for item in sources if item.get("score") is not None]
        is_low_confidence = (
            should_apply_similarity_threshold(config)
            and bool(scores)
            and config.retrieval_similarity_threshold > 0
            and max(scores) < config.retrieval_similarity_threshold
        )
        answer = config.query_refusal_response if is_low_confidence else str(response).strip()

    return {
        "answer": answer,
        "rewritten_query": query,
        "sources": sources,
        "benchmark_profile": profile_name,
    }


# Xóa cache tài nguyên cũ rồi nạp lại collection cục bộ của Hybrid.
def reload_local_resources() -> dict[str, str]:
    get_resources.cache_clear()
    config, _ = get_resources()
    return {
        "rag_id": CURRENT_RAG_ID,
        "collection": config.qdrant_collection,
        "message": f"Đã reload collection {config.qdrant_collection}.",
    }


# Chuẩn hóa danh sách chunk mới/xóa để gửi delta sync sang Graph RAG.
def _build_graph_sync_payload(summary: dict | None) -> dict[str, list[str] | bool]:
    summary = summary or {}
    chunk_relative_paths: list[str] = []
    deleted_relative_paths: list[str] = []

    for item in list((summary.get("added") or {}).get("web") or []):
        chunk_relative_path = str(item.get("chunk_relative_path") or "").strip()
        if chunk_relative_path:
            chunk_relative_paths.append(chunk_relative_path)
    for item in list((summary.get("added") or {}).get("pdf") or []):
        chunk_relative_path = str(item.get("chunk_relative_path") or "").strip()
        if chunk_relative_path:
            chunk_relative_paths.append(chunk_relative_path)
    for item in list(summary.get("removed") or []):
        txt_relative_path = str(item.get("txt_relative_path") or "").strip()
        if txt_relative_path:
            deleted_relative_paths.append(txt_relative_path.replace("data_txt/", "", 1))

    return {
        "full_reload": False,
        "chunk_relative_paths": chunk_relative_paths,
        "deleted_relative_paths": deleted_relative_paths,
    }


# Reload Hybrid cục bộ rồi yêu cầu các backend còn lại đồng bộ theo job.
def reload_cluster_resources(graph_sync_payload: dict | None = None) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    local_summary = run_reload_job()
    results[CURRENT_RAG_ID] = {"status": "ok", "message": local_summary["message"]}
    for rag_id, base_url in build_cluster_server_urls().items():
        if rag_id == CURRENT_RAG_ID:
            continue
        try:
            request_payload = {"async_mode": True}
            if rag_id == "graph" and graph_sync_payload:
                request_payload.update(graph_sync_payload)
            job = post_json(f"{base_url}/api/admin/reload", request_payload)
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("Backend khong tra ve job_id reload.")
            payload = wait_for_job(f"{base_url}/api/admin/jobs/{job_id}")
            result = payload.get("result") or {}
            results[rag_id] = {"status": "ok", "message": str(result.get("message") or "Da reload.")}
        except Exception as exc:
            results[rag_id] = {"status": "error", "message": str(exc)}
    return results


# Kiểm tra kết quả reload cụm và báo lỗi nếu còn backend nào thất bại.
def ensure_reload_success(reloads: dict[str, dict[str, str]], *, action_label: str) -> None:
    failed = {rag_id: result for rag_id, result in reloads.items() if result.get("status") != "ok"}
    if not failed:
        return
    details = " | ".join(f"{rag_id}: {result.get('message') or 'Lỗi chưa xác định'}" for rag_id, result in failed.items())
    raise RuntimeError(f"Đã {action_label} dữ liệu thô nhưng chưa reload xong toàn bộ hệ thống. {details}")


# Tạo job nền để đồng bộ lại cụm sau khi có thay đổi dữ liệu.
def start_cluster_reload_job(*, graph_sync_payload: dict | None, action_label: str) -> dict:
    return create_job(
        action="cluster_reload",
        runner=lambda: _run_cluster_reload_job(graph_sync_payload=graph_sync_payload, action_label=action_label),
    )


# Thực thi job reload cụm và trả về thông tin tổng hợp cho UI.
def _run_cluster_reload_job(*, graph_sync_payload: dict | None, action_label: str) -> dict:
    reloads = reload_cluster_resources(graph_sync_payload)
    ensure_reload_success(reloads, action_label=action_label)
    return {
        "reloads": reloads,
        "message": f"Đã reload xong 3 hệ sau khi {action_label} dữ liệu.",
    }


# Warm-up index trong nền để server có thể mở cổng sớm hơn.
def warm_resources_in_background() -> None:
    try:
        config, _ = get_resources()
        print(f"[hybrid] Warm-up hoàn tất cho collection {config.qdrant_collection}.")
    except Exception as exc:
        print(f"[hybrid] Warm-up lỗi: {exc}")


# Reload riêng tài nguyên Hybrid dưới lock truy vấn.
def run_reload_job() -> dict[str, str]:
    with RAG_LOCK:
        return reload_local_resources()


# Chuyển job thêm dữ liệu sang trạng thái lỗi nếu mọi PDF mới đều bị hoàn tác.
def _raise_if_add_failed(summary: dict[str, object]) -> None:
    if summary.get("changed"):
        return
    if summary.get("has_failures"):
        raise RuntimeError(str(summary.get("message") or "Không thể nạp dữ liệu PDF."))


# Thêm dữ liệu thô mới, build chunk và xếp job reload nền cho toàn cụm.
def run_add_data_job(*, links_text: str, pdf_files: list[dict] | None = None, display_name: str = "") -> dict:
    with ADMIN_LOCK:
        summary = add_corpus_documents(
            links_text=links_text,
            pdf_files=list(pdf_files or []),
            display_name=display_name,
        )
    _raise_if_add_failed(summary)
    reload_job = start_cluster_reload_job(
        graph_sync_payload=_build_graph_sync_payload(summary),
        action_label="nạp",
    ) if summary.get("changed") else None
    return {
        **summary,
        "reload_job_id": str((reload_job or {}).get("job_id") or ""),
        "reload_status": "queued" if reload_job else "skipped",
    }


# Xóa tài liệu khỏi corpus dùng chung rồi xếp job reload nền cho toàn cụm.
def run_delete_data_job(document_ids: list[str]) -> dict:
    with ADMIN_LOCK:
        summary = delete_corpus_documents(document_ids)
    reload_job = start_cluster_reload_job(
        graph_sync_payload=_build_graph_sync_payload(summary),
        action_label="xóa",
    ) if summary.get("changed") else None
    return {
        **summary,
        "reload_job_id": str((reload_job or {}).get("job_id") or ""),
        "reload_status": "queued" if reload_job else "skipped",
    }


# Xử lý toàn bộ API HTTP cho giao diện chat và quản trị dữ liệu của Hybrid.
class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUHybridLlamaIndexHTTP/3.0"

    # Điều phối các route GET như UI, health, danh sách tài liệu và trạng thái job.
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(load_ui_html())
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/api/admin/documents":
            self._send_json(list_corpus_documents())
            return
        if parsed.path.startswith("/api/admin/jobs/"):
            self._handle_admin_job_request(parsed.path)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Điều phối các route POST như chat, add/delete dữ liệu và reload.
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat_request()
            return
        if parsed.path == "/api/admin/add":
            self._handle_add_data_request()
            return
        if parsed.path == "/api/admin/delete":
            self._handle_delete_data_request()
            return
        if parsed.path == "/api/admin/reload":
            self._handle_reload_request()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Xử lý request chat và gọi Hybrid RAG để sinh câu trả lời.
    def _handle_chat_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return

        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return

        profile_name, runtime_overrides = parse_benchmark_profile_payload(payload)
        try:
            with RAG_LOCK:
                response = answer_query(query, profile_name=profile_name, runtime_overrides=runtime_overrides)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(response, status=HTTPStatus.OK)

    # Xử lý request thêm dữ liệu và hỗ trợ chế độ async bằng job.
    def _handle_add_data_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        links_text = str(payload.get("links_text") or "")
        display_name = str(payload.get("display_name") or "")
        pdf_files = list(payload.get("pdf_files") or [])
        if payload.get("async_mode"):
            job = create_job(
                action="add_data",
                runner=lambda: run_add_data_job(links_text=links_text, pdf_files=pdf_files, display_name=display_name),
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_add_data_job(links_text=links_text, pdf_files=pdf_files, display_name=display_name)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể nạp dữ liệu. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(summary, status=HTTPStatus.OK)

    # Xử lý request xóa tài liệu và hỗ trợ chế độ async bằng job.
    def _handle_delete_data_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        raw_documents = payload.get("documents") or payload.get("document_ids") or []
        if not isinstance(raw_documents, list) or not raw_documents:
            self._send_json({"error": "Vui lòng chọn tài liệu cần xóa."}, status=HTTPStatus.BAD_REQUEST)
            return

        document_ids = [str(item.get("id") if isinstance(item, dict) else item).strip() for item in raw_documents]
        if payload.get("async_mode"):
            job = create_job(
                action="delete_data",
                runner=lambda: run_delete_data_job(document_ids),
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_delete_data_job(document_ids)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể xóa dữ liệu. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(summary, status=HTTPStatus.OK)

    # Xử lý request reload riêng Hybrid hoặc tạo job reload async.
    def _handle_reload_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        if payload.get("async_mode"):
            job = create_job(action="reload", runner=run_reload_job)
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_reload_job()
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể reload Hybrid RAG. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(summary, status=HTTPStatus.OK)

    # Trả về trạng thái hiện tại của một job quản trị theo job_id.
    def _handle_admin_job_request(self, path: str) -> None:
        job_id = path.rsplit("/", 1)[-1].strip()
        if not job_id:
            self._send_json({"error": "Thiếu job_id."}, status=HTTPStatus.BAD_REQUEST)
            return
        job = get_job(job_id)
        if job is None:
            self._send_json({"error": "Không tìm thấy job."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(job, status=HTTPStatus.OK)

    # Đọc và parse JSON body từ request hiện tại.
    def _read_json_payload(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return None

    # Tắt log mặc định của BaseHTTPRequestHandler để terminal gọn hơn.
    def log_message(self, format: str, *args) -> None:
        return

    # Gửi phản hồi HTML UTF-8 cho giao diện chat.
    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self._send_response_body(body, "text/html; charset=utf-8", status=status)

    # Gửi phản hồi JSON UTF-8 cho API.
    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_response_body(body, "application/json; charset=utf-8", status=status)

    def _send_response_body(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True


# Chạy Hybrid RAG ở chế độ CLI đơn giản để hỏi đáp trong terminal.
def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Hybrid RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input("\nNhập câu hỏi: ").strip()
        if query.lower() == "exit":
            break
        with RAG_LOCK:
            result = answer_query(query)
        print("\n=== TRẢ LỜI ===")
        print(result["answer"])


# Lấy cổng server từ biến môi trường hoặc dùng mặc định của Hybrid.
def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Khởi động HTTP server, warm-up tài nguyên nền và giữ vòng lặp phục vụ request.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    Thread(target=warm_resources_in_background, daemon=True).start()
    print("Đang khởi tạo Hybrid RAG...")
    print(f"Hybrid RAG đang chạy tại http://{host}:{resolved_port}")
    print("Warm-up index đang chạy nền; giao diện và /health sẽ sẵn sàng ngay.")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


# Chọn chế độ chạy phù hợp cho app Hybrid.
def main() -> None:
    if "--cli" in sys.argv:
        # Bước 1: nếu người dùng yêu cầu CLI thì chuyển sang chế độ hỏi đáp trong terminal.
        run_cli()
        return
    # Bước 2: nếu không có cờ CLI thì khởi động server HTTP cho giao diện web.
    run_server()


if __name__ == "__main__":
    main()
