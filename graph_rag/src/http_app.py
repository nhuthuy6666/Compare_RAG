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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_service import answer_question
from llamaindex_shared import ChatUiConfig, build_chat_ui_tabs, render_chat_ui
from utils import configure_console_utf8


configure_console_utf8()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8502
GRAPH_LOCK = Lock()


# Chuyển kết quả GraphRAG thành schema thống nhất với 2 hệ còn lại.
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


# Tạo HTML UI dùng chung cho GraphRAG và đánh dấu tab Graph đang được chọn.
@lru_cache(maxsize=1)
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id="graph",
            page_title="NTU Graph RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description=(
                "Graph RAG tổng hợp thông tin từ các fact trong đồ thị tri thức. "
                "Giao diện này được dùng chung với Baseline và Hybrid RAG."
            ),
            header_badge="Graph RAG",
            header_subtitle="Graph retrieval | Query fusion | Fact graph",
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
            storage_key="ntu_project_graph_sessions",
            suggestions=[
                "Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
                "Hồ sơ nhập học gồm những gì?",
                "Học phí dự kiến một năm là bao nhiêu?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUGraphRagHTTP/2.0"

    # Phục vụ UI gốc, health check và favicon cho GraphRAG.
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

        question = str(payload.get("query") or payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            with GRAPH_LOCK:
                result = answer_question(question)
        except Exception as exc:
            self._send_json(
                {"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(_answer_to_payload(question, result), status=HTTPStatus.OK)

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


# Chạy vòng lặp hỏi đáp trong terminal để test nhanh GraphRAG.
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


# Đọc port UI từ biến môi trường để launcher có thể thống nhất 3 server.
def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Khởi động HTTP server GraphRAG theo cùng cách với baseline và hybrid.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Đang khởi tạo Graph RAG...")
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    print(f"Graph RAG đang chạy tại http://{host}:{resolved_port}")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


# Entry point của GraphRAG.
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
