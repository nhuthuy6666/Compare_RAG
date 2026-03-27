from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llamaindex_shared import (  # noqa: E402
    ChatUiConfig,
    build_chat_ui_tabs,
    build_query_engine,
    collect_sources,
    configure_models,
    ensure_vector_index,
    load_shared_config,
    render_chat_ui,
)


BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".qdrant_state.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
RAG_LOCK = Lock()


# Tạo HTML UI dùng chung cho baseline RAG với tab điều hướng sang 2 hệ còn lại.
@lru_cache(maxsize=1)
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id="baseline",
            page_title="NTU Baseline RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description=(
                "Baseline RAG dùng dense retrieval thuần để làm mốc so sánh. "
                "Giao diện này được dùng chung với Hybrid và Graph RAG."
            ),
            header_badge="Baseline RAG",
            header_subtitle="Dense retrieval | Qdrant | LlamaIndex",
            assistant_label="Baseline NTU Bot",
            empty_title="Baseline RAG sẵn sàng",
            empty_description=(
                "Bạn có thể đặt cùng một câu hỏi và chuyển tab để so sánh kết quả "
                "giữa Baseline, Hybrid và Graph RAG."
            ),
            placeholder="Hỏi về điểm chuẩn, học phí, mã trường, phương thức tuyển sinh...",
            composer_hint="Enter để gửi | Shift + Enter để xuống dòng",
            loading_message="Đang truy xuất dense index trong Qdrant...",
            ready_message="Đã hoàn tất.",
            storage_key="ntu_project_baseline_sessions",
            suggestions=[
                "Mã trường Đại học Nha Trang là gì?",
                "Điện thoại tuyển sinh là số nào?",
                "Ngành CNTT năm 2025 lấy bao nhiêu chỉ tiêu?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


# Khởi tạo config, model và vector index baseline chỉ một lần trong vòng đời server.
@lru_cache(maxsize=1)
def get_resources():
    config = load_shared_config(collection_name="ntu_rag")
    configure_models(config)
    index = ensure_vector_index(
        config,
        state_path=STATE_PATH,
        enable_hybrid=False,
    )
    query_engine = build_query_engine(index, config, enable_hybrid=False)
    return config, query_engine


# Xử lý một câu hỏi theo baseline RAG và trả payload thống nhất cho UI/API.
def answer_query(query: str) -> dict:
    config, query_engine = get_resources()
    response = query_engine.query(query)
    sources = collect_sources(response, limit=config.retrieval_top_n)

    if not sources:
        answer = config.query_refusal_response
    else:
        scores = [float(item["score"]) for item in sources if item.get("score") is not None]
        is_low_confidence = (
            bool(scores)
            and config.retrieval_similarity_threshold > 0
            and max(scores) < config.retrieval_similarity_threshold
        )
        answer = config.query_refusal_response if is_low_confidence else str(response).strip()

    return {
        "answer": answer,
        "rewritten_query": query,
        "sources": sources,
    }


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTURagHTTP/2.0"

    # Phục vụ UI gốc, health check và favicon cho baseline RAG.
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(load_ui_html())
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Nhận payload chat và trả kết quả cho frontend.
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json_payload()
        if payload is None:
            return

        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            with RAG_LOCK:
                response = answer_query(query)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(response, status=HTTPStatus.OK)

    # Đọc body JSON từ request và tự xử lý lỗi payload sai định dạng.
    def _read_json_payload(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return None

    # Tắt log mặc định của BaseHTTPRequestHandler để console gọn hơn.
    def log_message(self, format: str, *args) -> None:
        return

    # Gửi HTML cho trình duyệt với content-type phù hợp.
    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Gửi JSON cho frontend và giữ nguyên Unicode để dễ debug.
    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# Chạy vòng lặp hỏi đáp trong terminal để test nhanh baseline RAG.
def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Baseline RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input("\nNhập câu hỏi: ").strip()
        if query.lower() == "exit":
            break
        with RAG_LOCK:
            result = answer_query(query)
        print("\n=== TRẢ LỜI ===")
        print(result["answer"])


# Đọc port UI từ biến môi trường để launcher có thể thống nhất 3 server.
def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Khởi động HTTP server baseline sau khi đảm bảo vector index đã sẵn sàng.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Đang khởi tạo Baseline RAG...")
    config, _ = get_resources()
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    print(f"Baseline RAG đang chạy tại http://{host}:{resolved_port}")
    print(f"Qdrant collection: {config.qdrant_collection} @ {config.qdrant_url}")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


# Entry point của baseline RAG.
# 1) Kiểm tra xem user có muốn chạy chế độ CLI hay không.
# 2) Nếu có `--cli` thì mở vòng lặp hỏi đáp trong terminal.
# 3) Nếu không thì đọc `UI_PORT` và khởi động web UI/API dùng giao diện chung.
def main() -> None:
    if "--cli" in sys.argv:
        run_cli()
        return
    run_server()


if __name__ == "__main__":
    main()
