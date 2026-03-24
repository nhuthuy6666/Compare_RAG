from __future__ import annotations

import json
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
    build_query_engine,
    collect_sources,
    configure_models,
    ensure_vector_index,
    load_shared_config,
)


BASE_DIR = Path(__file__).resolve().parent
UI_TEMPLATE_PATH = BASE_DIR / "chat_ui.html"
STATE_PATH = BASE_DIR / ".qdrant_state.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
RAG_LOCK = Lock()


# Tải file HTML giao diện cho Hybrid RAG.
def load_ui_html() -> str:
    return UI_TEMPLATE_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
# Khởi tạo config, model, hybrid vector index và query engine một lần duy nhất.
def get_resources():
    config = load_shared_config(collection_name="ntu_hybrid_llamaindex")
    configure_models(config)
    index = ensure_vector_index(
        config,
        state_path=STATE_PATH,
        enable_hybrid=True,
    )
    query_engine = build_query_engine(index, config, enable_hybrid=True)
    return config, query_engine


# Chạy pipeline hybrid retrieval + generation và trả payload cho UI/API.
def answer_query(query: str) -> dict:
    config, query_engine = get_resources()
    response = query_engine.query(query)
    sources = collect_sources(response, limit=config.retrieval_top_n)
    if not sources:
        answer = config.query_refusal_response
    else:
        scores = [float(item["score"]) for item in sources if item.get("score") is not None]
        if (
            scores
            and config.retrieval_similarity_threshold > 0
            and max(scores) < config.retrieval_similarity_threshold
        ):
            answer = config.query_refusal_response
        else:
            answer = str(response).strip()

    return {
        "answer": answer,
        "rewritten_query": query,
        "sources": sources,
    }


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUHybridLlamaIndexHTTP/1.0"

    # Phục vụ UI, health check và favicon.
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

    # Nhận payload chat JSON, chạy Hybrid RAG và trả kết quả cho frontend/benchmark.
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
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

    # Tắt log mặc định của server để dễ theo dõi log business.
    def log_message(self, format: str, *args) -> None:
        return

    # Gửi HTML cho route gốc.
    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # Gửi JSON cho API và frontend.
    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# Chạy terminal chat để test nhanh retrieval của Hybrid RAG.
def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Hybrid LlamaIndex RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input("\nNhập câu hỏi: ").strip()
        if query.lower() == "exit":
            break
        with RAG_LOCK:
            result = answer_query(query)
        print("\n=== TRẢ LỜI ===")
        print(result["answer"])


# Khởi động web UI và API sau khi đảm bảo collection Qdrant đã sẵn sàng.
def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Đang khởi tạo Hybrid LlamaIndex RAG...")
    config, _ = get_resources()
    server = ThreadingHTTPServer((host, port), ChatHTTPRequestHandler)
    print(f"Giao diện chatbot đang chạy tại http://{host}:{port}")
    print(f"Qdrant collection: {config.qdrant_collection} @ {config.qdrant_url}")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


# Entry point cho Hybrid RAG.
# 1) Kiểm tra xem user có muốn vào CLI hay không.
# 2) Nếu có `--cli` thì chạy vòng lặp hỏi đáp trong terminal.
# 3) Nếu không thì khởi động web UI và endpoint `/api/chat`.
def main() -> None:
    if "--cli" in sys.argv:
        run_cli()
        return
    run_server()


if __name__ == "__main__":
    main()
