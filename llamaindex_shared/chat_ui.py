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
    api_documents_url: str = "/api/admin/documents"
    api_add_data_url: str = "/api/admin/add"
    api_delete_data_url: str = "/api/admin/delete"
    new_chat_label: str = "Cuộc trò chuyện mới"
    manage_data_label: str = "Thêm dữ liệu"
    send_button_label: str = "Gửi"
    initial_title: str = "Cuộc trò chuyện mới"
    history_label: str = "Lịch sử"
    empty_history_text: str = "Chưa có tin nhắn"
    continue_history_text: str = "Tiếp tục hội thoại"
    sending_error_prefix: str = "Không thể xử lý câu hỏi."
    data_modal_title: str = "Quản lý dữ liệu"


@lru_cache(maxsize=1)
def _load_chat_ui_template() -> str:
    return CHAT_UI_TEMPLATE_PATH.read_text(encoding="utf-8")


def _serialize_chat_ui_config(config: ChatUiConfig) -> str:
    payload = asdict(config)
    payload["tabs"] = [asdict(tab) for tab in config.tabs]
    return json.dumps(payload, ensure_ascii=False)


def _get_ui_host() -> str:
    return os.getenv("RAG_UI_HOST", "127.0.0.1")


def _get_ui_port(env_name: str, default_port: int) -> int:
    return int(os.getenv(env_name, str(default_port)))


def build_chat_ui_tabs() -> list[ChatUiTab]:
    host = _get_ui_host()
    return [
        ChatUiTab(id="hybrid", label="Hybrid", href=f"http://{host}:{_get_ui_port('HYBRID_UI_PORT', 8000)}/"),
        ChatUiTab(id="baseline", label="Baseline", href=f"http://{host}:{_get_ui_port('BASELINE_UI_PORT', 8001)}/"),
        ChatUiTab(id="graph", label="Graph", href=f"http://{host}:{_get_ui_port('GRAPH_UI_PORT', 8502)}/"),
    ]


def render_chat_ui(config: ChatUiConfig) -> str:
    template = _load_chat_ui_template()
    config_json = _serialize_chat_ui_config(config)
    return template.replace("__CHAT_UI_CONFIG__", config_json)