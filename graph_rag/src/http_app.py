from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_service import answer_question, warm_up_graph
from config import load_config
from ingest import _load_records, _run_ingest_pipeline
from llamaindex_shared import (
    ChatUiConfig,
    add_corpus_documents,
    build_chat_ui_tabs,
    build_cluster_server_urls,
    create_job,
    delete_corpus_documents,
    get_job,
    list_corpus_documents,
    post_json,
    render_chat_ui,
    wait_for_job,
)
from llamaindex_shared.benchmark_runtime import parse_benchmark_profile_payload
from utils import configure_console_utf8


configure_console_utf8()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8502
CURRENT_RAG_ID = "graph"
GRAPH_LOCK = Lock()


def _answer_to_payload(question: str, result) -> dict:
    return {
        "answer": result.answer,
        "rewritten_query": question,
        "sources": [
            {
                "source": fact.relative_path or fact.heading_path or "graph_fact",
                "relative_path": fact.relative_path,
                "heading_path": fact.heading_path,
                "content": fact.content,
                "score": fact.score,
            }
            for fact in result.facts
        ],
    }


@lru_cache(maxsize=1)
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id=CURRENT_RAG_ID,
            page_title="NTU Graph RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description=(
                "Graph RAG tổng hợp thông tin từ các fact trong đồ thị tri thức. "
                "Bạn có thể so sánh trực tiếp với Baseline và Hybrid trên cùng giao diện."
            ),
            header_badge="Graph RAG",
            header_subtitle="Neo4j graph retrieval | Query fusion | Fact graph",
            assistant_label="Graph NTU Bot",
            empty_title="Graph RAG sẵn sàng",
            empty_description=(
                "Bạn có thể đặt cùng một câu hỏi và chuyển tab để so sánh kết quả "
                "giữa Graph, Hybrid và Baseline RAG."
            ),
            placeholder="Ví dụ: Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
            composer_hint="Đặt cùng câu hỏi ở 3 tab để so sánh kết quả",
            loading_message="Đang truy xuất graph facts...",
            ready_message="Đã trả lời xong.",
            storage_key="ntu_fusion_graph_sessions",
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
                "Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
                "Hồ sơ nhập học gồm những gì?",
                "Học phí dự kiến một năm là bao nhiêu?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


def reload_local_resources() -> dict[str, str]:
    config = load_config()
    args = SimpleNamespace(chunk_jsonl_root=None, output_dir=None, limit=None, dry_run=False, reset_graph=True)
    all_records = _load_records(args, config)
    summary = _run_ingest_pipeline(
        all_records=all_records,
        config=config,
        reset_graph_first=True,
        dry_run=False,
    )
    warm_up_graph.cache_clear()
    warm_up_graph()
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": str(summary["fact_count"]),
        "message": (
            f"Đã reload graph với {summary['chunk_count']} chunk, "
            f"{summary['entity_count']} entity và {summary['fact_count']} fact."
        ),
    }


def reload_cluster_resources() -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    local_summary = reload_local_resources()
    results[CURRENT_RAG_ID] = {"status": "ok", "message": local_summary["message"]}
    for rag_id, base_url in build_cluster_server_urls().items():
        if rag_id == CURRENT_RAG_ID:
            continue
        try:
            job = post_json(f"{base_url}/api/admin/reload", {"async_mode": True})
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("Backend khong tra ve job_id reload.")
            payload = wait_for_job(f"{base_url}/api/admin/jobs/{job_id}")
            result = payload.get("result") or {}
            results[rag_id] = {"status": "ok", "message": str(result.get("message") or "Da reload.")}
        except Exception as exc:
            results[rag_id] = {"status": "error", "message": str(exc)}
    return results


def ensure_reload_success(reloads: dict[str, dict[str, str]], *, action_label: str) -> None:
    failed = {rag_id: result for rag_id, result in reloads.items() if result.get("status") != "ok"}
    if not failed:
        return
    details = " | ".join(f"{rag_id}: {result.get('message') or 'Lỗi chưa xác định'}" for rag_id, result in failed.items())
    raise RuntimeError(f"Đã {action_label} dữ liệu thô nhưng chưa reload xong toàn bộ hệ thống. {details}")


def warm_resources_in_background() -> None:
    try:
        warm_up_graph.cache_clear()
        warm_up_graph()
        print("[graph] Warm-up hoàn tất.")
    except Exception as exc:
        print(f"[graph] Warm-up lỗi: {exc}")


def run_reload_job() -> dict[str, str]:
    with GRAPH_LOCK:
        return reload_local_resources()


def run_add_data_job(*, links_text: str, pdf_files: list[dict] | None = None) -> dict:
    with GRAPH_LOCK:
        summary = add_corpus_documents(
            links_text=links_text,
            pdf_files=list(pdf_files or []),
        )
        reloads = reload_cluster_resources() if summary.get("changed") else {}
        ensure_reload_success(reloads, action_label="nạp")
        return {**summary, "reloads": reloads}


def run_delete_data_job(document_ids: list[str]) -> dict:
    with GRAPH_LOCK:
        summary = delete_corpus_documents(document_ids)
        reloads = reload_cluster_resources() if summary.get("changed") else {}
        ensure_reload_success(reloads, action_label="xóa")
        return {**summary, "reloads": reloads}


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUGraphRagHTTP/5.0"

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

    def _handle_chat_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return

        question = str(payload.get("query") or payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return

        profile_name, runtime_overrides = parse_benchmark_profile_payload(payload)
        try:
            with GRAPH_LOCK:
                result = answer_question(question, runtime_overrides=runtime_overrides)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        response_payload = _answer_to_payload(question, result)
        response_payload["benchmark_profile"] = profile_name
        self._send_json(response_payload, status=HTTPStatus.OK)

    def _handle_add_data_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        links_text = str(payload.get("links_text") or "")
        pdf_files = list(payload.get("pdf_files") or [])
        if payload.get("async_mode"):
            job = create_job(
                action="add_data",
                runner=lambda: run_add_data_job(links_text=links_text, pdf_files=pdf_files),
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_add_data_job(links_text=links_text, pdf_files=pdf_files)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể nạp dữ liệu. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(summary, status=HTTPStatus.OK)

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
                {"error": f"Không thể reload Graph RAG. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._send_json(summary, status=HTTPStatus.OK)

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

    def _read_json_payload(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return None

    def log_message(self, format: str, *args) -> None:
        return

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Graph RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        question = input("\nNhập câu hỏi: ").strip()
        if question.lower() == "exit":
            break
        with GRAPH_LOCK:
            result = answer_question(question)
        print("\n=== TRẢ LỜI ===")
        print(result.answer)


def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    config = load_config()
    Thread(target=warm_resources_in_background, daemon=True).start()
    print("Đang khởi tạo Graph RAG...")
    print(f"Graph RAG đang chạy tại http://{host}:{resolved_port}")
    print(f"Neo4j: {config.neo4j_uri} / database={config.neo4j_database}")
    print("Warm-up graph đang chạy nền; giao diện và /health sẽ sẵn sàng ngay.")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


def main() -> None:
    if "--cli" in sys.argv:
        run_cli()
        return
    run_server()


if __name__ == "__main__":
    main()
