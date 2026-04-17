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

from chat_service import answer_question, warm_up_graph
from config import load_config
from llamaindex_shared import ChatUiConfig, build_chat_ui_tabs, render_chat_ui
from llamaindex_shared.benchmark_runtime import parse_benchmark_profile_payload
from utils import configure_console_utf8


configure_console_utf8()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8502
GRAPH_LOCK = Lock()


# Chuyển kết quả GraphRAG thành schema thống nhất với 2 hệ còn lại.
def _answer_to_payload(question: str, result) -> dict:
    """Chuẩn hóa kết quả GraphRAG về schema trả về chung cho frontend và benchmark."""

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


# Tạo HTML UI dùng chung cho GraphRAG và dùng cùng modal ingest với hai hệ còn lại.
@lru_cache(maxsize=1)
def load_ui_html() -> str:
    """Render HTML giao diện chat của GraphRAG."""

    return render_chat_ui(
        ChatUiConfig(
            current_rag_id="graph",
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
            suggestions=[
                "Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
                "Hồ sơ nhập học gồm những gì?",
                "Học phí dự kiến một năm là bao nhiêu?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler cho UI, chat API và ingest API của GraphRAG."""

    server_version = "NTUGraphRagHTTP/4.0"

    # Phục vụ UI gốc, health check và favicon cho GraphRAG.
    def do_GET(self) -> None:
        """Phục vụ UI gốc, health check và favicon cho GraphRAG."""

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

    # Điều phối request POST sang endpoint chat.
    def do_POST(self) -> None:
        """Điều phối request POST sang endpoint chat."""

        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            self._handle_chat_request()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Xử lý một câu hỏi chat và trả lại kết quả GraphRAG cho frontend.
    def _handle_chat_request(self) -> None:
        """Nhận request chat, gọi GraphRAG và trả kết quả JSON cho frontend."""

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

    # Đọc body JSON từ request và tự xử lý lỗi payload sai định dạng.
    def _read_json_payload(self) -> dict | None:
        """Đọc body JSON từ request và tự xử lý lỗi payload sai định dạng."""

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return None

    # Tắt log mặc định của BaseHTTPRequestHandler để console gọn hơn.
    def log_message(self, format: str, *args) -> None:
        """Tắt log mặc định của BaseHTTPRequestHandler để console gọn hơn."""

        return

    # Gửi HTML cho trình duyệt với content-type phù hợp.
    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Gửi HTML cho trình duyệt với content-type phù hợp."""

        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Gửi JSON cho frontend và giữ nguyên Unicode để dễ debug.
    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Gửi JSON cho frontend và giữ nguyên Unicode để dễ debug."""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# Chạy vòng lặp hỏi đáp trong terminal để test nhanh GraphRAG.
def run_cli() -> None:
    """Chạy vòng lặp hỏi đáp trong terminal để test nhanh GraphRAG."""

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
    """Đọc cổng server từ biến môi trường để launcher có thể thống nhất cấu hình."""

    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Khởi động HTTP server GraphRAG theo cùng cách với baseline và hybrid.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    """Khởi động HTTP server GraphRAG theo cùng cách với baseline và hybrid."""

    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Đang khởi tạo Graph RAG...")
    config = load_config()
    warm_up_graph()
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    print(f"Graph RAG đang chạy tại http://{host}:{resolved_port}")
    print(f"Neo4j: {config.neo4j_uri} / database={config.neo4j_database}")
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
    """Điểm vào chính của GraphRAG."""

    # Bước 1: kiểm tra cờ `--cli` để quyết định chạy chế độ terminal hay web server.
    if "--cli" in sys.argv:
        # Bước 2: nếu đang test tay trong terminal, chạy vòng lặp hỏi đáp rồi thoát.
        run_cli()
        return
    # Bước 3: nếu không có `--cli`, khởi động HTTP server phục vụ UI và API.
    run_server()


if __name__ == "__main__":
    main()
