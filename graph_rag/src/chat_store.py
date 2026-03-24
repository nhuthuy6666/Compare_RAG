from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config import load_config


# Tạo timestamp UTC chuẩn ISO để dùng nhất quán cho session và message.
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Sinh tiêu đề ngắn từ câu hỏi đầu tiên của người dùng.
def _title_from_question(question: str) -> str:
    normalized = " ".join(question.split())
    return normalized[:48] + ("..." if len(normalized) > 48 else "")


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: str = field(default_factory=_utc_now)
    facts: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[ChatMessage]


class ChatStore:
    # Khởi tạo storage path và bảo đảm thư mục lưu session đã tồn tại.
    def __init__(self, storage_path: Path | None = None) -> None:
        config = load_config()
        self.storage_path = storage_path or config.processed_dir / "chat_sessions.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc toàn bộ session từ file JSON và phục hồi về dataclass.
    def _load_all(self) -> list[ChatSession]:
        if not self.storage_path.exists():
            return []

        raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        sessions: list[ChatSession] = []
        for item in raw:
            messages = [ChatMessage(**message) for message in item.get("messages", [])]
            sessions.append(
                ChatSession(
                    id=item["id"],
                    title=item["title"],
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    messages=messages,
                ),
            )
        return sessions

    # Ghi toàn bộ session hiện tại về file JSON.
    def _save_all(self, sessions: list[ChatSession]) -> None:
        payload = []
        for session in sessions:
            payload.append(
                {
                    "id": session.id,
                    "title": session.title,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "messages": [asdict(message) for message in session.messages],
                },
            )
        self.storage_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Trả về danh sách session theo thứ tự mới cập nhật trước.
    def list_sessions(self) -> list[ChatSession]:
        sessions = self._load_all()
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    # Tìm một session theo id.
    def get_session(self, session_id: str) -> ChatSession | None:
        for session in self._load_all():
            if session.id == session_id:
                return session
        return None

    # Tạo session mới rỗng và lưu ngay xuống disk.
    def create_session(self) -> ChatSession:
        sessions = self._load_all()
        now = _utc_now()
        session = ChatSession(
            id=uuid.uuid4().hex,
            title="Cuộc trò chuyện mới",
            created_at=now,
            updated_at=now,
            messages=[],
        )
        sessions.append(session)
        self._save_all(sessions)
        return session

    # Xóa một session theo id rồi ghi lại danh sách còn lại.
    def delete_session(self, session_id: str) -> None:
        sessions = [session for session in self._load_all() if session.id != session_id]
        self._save_all(sessions)

    # Gắn thêm một lượt hỏi đáp vào session hiện có và cập nhật thời gian chỉnh sửa cuối.
    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        facts: list[dict[str, str]],
    ) -> ChatSession:
        sessions = self._load_all()
        for session in sessions:
            if session.id != session_id:
                continue

            now = _utc_now()
            if not session.messages:
                session.title = _title_from_question(question)
            session.messages.append(ChatMessage(role="user", content=question, created_at=now))
            session.messages.append(
                ChatMessage(role="assistant", content=answer, created_at=now, facts=facts),
            )
            session.updated_at = now
            self._save_all(sessions)
            return session

        raise KeyError(f"Chat session not found: {session_id}")
