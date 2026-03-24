from __future__ import annotations

import os
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

from chat_service import answer_question
from chat_store import ChatStore
from utils import configure_console_utf8


configure_console_utf8()


## Chuẩn hóa chat session object về JSON-serializable dict cho UI/API.
def _session_to_dict(session):
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "messages": [asdict(message) for message in session.messages],
    }


## Chuẩn hóa kết quả hỏi đáp stateless về schema chung `answer/sources`.
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


## Tạo Flask app cho GraphRAG.
## App này đồng thời phục vụ web UI có session và endpoint stateless `/api/chat` cho benchmark.
def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    store = ChatStore()

    # Render trang chủ cùng danh sách session và session đang active.
    @app.get("/")
    def index():
        sessions = store.list_sessions()
        active_session = sessions[0] if sessions else store.create_session()
        sessions = store.list_sessions()
        return render_template(
            "index.html",
            sessions=[_session_to_dict(session) for session in sessions],
            active_session=_session_to_dict(active_session),
        )

    # Endpoint health check cho local smoke test và benchmark runner.
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    # Endpoint stateless dùng schema chung với các hệ RAG còn lại.
    @app.post("/api/chat")
    def answer_chat():
        payload = request.get_json(silent=True) or {}
        question = (payload.get("query") or payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Vui long nhap cau hoi."}), 400

        result = answer_question(question)
        return jsonify(_answer_to_payload(question, result))

    # Tạo một cuộc trò chuyện mới cho UI có session.
    @app.post("/api/chats")
    def create_chat():
        session = store.create_session()
        return jsonify({"session": _session_to_dict(session)})

    # Lấy chi tiết một session theo id.
    @app.get("/api/chats/<session_id>")
    def get_chat(session_id: str):
        session = store.get_session(session_id)
        if session is None:
            return jsonify({"error": "Khong tim thay cuoc tro chuyen."}), 404
        return jsonify({"session": _session_to_dict(session)})

    # Xóa một session rồi trả lại danh sách session còn lại.
    @app.delete("/api/chats/<session_id>")
    def delete_chat(session_id: str):
        store.delete_session(session_id)
        sessions = store.list_sessions()
        return jsonify({"sessions": [_session_to_dict(session) for session in sessions]})

    # Ghi thêm một lượt hỏi đáp vào session hiện có.
    @app.post("/api/chats/<session_id>/messages")
    def send_message(session_id: str):
        payload = request.get_json(silent=True) or {}
        question = (payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Vui long nhap cau hoi."}), 400

        result = answer_question(question)
        session = store.append_turn(
            session_id=session_id,
            question=question,
            answer=result.answer,
            facts=[asdict(fact) for fact in result.facts],
        )
        return jsonify({"session": _session_to_dict(session)})

    return app


app = create_app()


## Entry point chạy Flask server cục bộ cho GraphRAG.
## 1) Đọc `UI_PORT` từ môi trường.
## 2) Khởi động Flask app trên `127.0.0.1`.
def main() -> None:
    port = int(os.getenv("UI_PORT", "8502"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
