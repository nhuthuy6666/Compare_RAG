from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAT_UI_TEMPLATE_PATH = PROJECT_ROOT / "llamaindex_shared" / "chat_ui_template.html"


@dataclass(frozen=True)
class ChatUiTab:
    id: str
    label: str
    href: str


@dataclass(frozen=True)
class ChatUiConfig:
    current_rag_id: str
    page_title: str
    brand_badge: str
    brand_title: str
    brand_description: str
    header_badge: str
    header_subtitle: str
    assistant_label: str
    empty_title: str
    empty_description: str
    placeholder: str
    composer_hint: str
    loading_message: str
    ready_message: str
    storage_key: str
    suggestions: list[str]
    tabs: list[ChatUiTab]
    api_chat_url: str = "/api/chat"
    new_chat_label: str = "Cuộc trò chuyện mới"
    send_button_label: str = "Gửi"
    initial_title: str = "Cuộc trò chuyện mới"
    history_label: str = "Lịch sử"
    empty_history_text: str = "Chưa có tin nhắn"
    continue_history_text: str = "Tiếp tục hội thoại"
    sending_error_prefix: str = "Không thể xử lý câu hỏi."


# Đọc template HTML dùng chung cho cả 3 giao diện và cache lại để tải nhanh.
@lru_cache(maxsize=1)
def _load_chat_ui_template() -> str:
    return CHAT_UI_TEMPLATE_PATH.read_text(encoding="utf-8")


# Chuyển dataclass config sang JSON để frontend có thể đọc trực tiếp bằng JavaScript.
def _serialize_chat_ui_config(config: ChatUiConfig) -> str:
    payload = asdict(config)
    payload["tabs"] = [asdict(tab) for tab in config.tabs]
    return json.dumps(payload, ensure_ascii=False)


# Lấy host cho các tab UI từ biến môi trường để dễ đổi port khi cần.
def _get_ui_host() -> str:
    return os.getenv("RAG_UI_HOST", "127.0.0.1")


# Lấy port theo từng RAG từ biến môi trường, nếu không có thì dùng mặc định trong code.
def _get_ui_port(env_name: str, default_port: int) -> int:
    return int(os.getenv(env_name, str(default_port)))


# Tạo danh sách tab điều hướng giữa 3 RAG trên cùng một giao diện.
def build_chat_ui_tabs() -> list[ChatUiTab]:
    host = _get_ui_host()
    return [
        ChatUiTab(id="hybrid", label="Hybrid", href=f"http://{host}:{_get_ui_port('HYBRID_UI_PORT', 8000)}/"),
        ChatUiTab(id="baseline", label="Baseline", href=f"http://{host}:{_get_ui_port('BASELINE_UI_PORT', 8001)}/"),
        ChatUiTab(id="graph", label="Graph", href=f"http://{host}:{_get_ui_port('GRAPH_UI_PORT', 8502)}/"),
    ]


# Render HTML cuối cùng bằng cách chèn JSON config vào template dùng chung.
def render_chat_ui(config: ChatUiConfig) -> str:
    template = _load_chat_ui_template()
    config_json = _serialize_chat_ui_config(config)
    return template.replace("__CHAT_UI_CONFIG__", config_json)
