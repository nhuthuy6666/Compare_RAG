from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLIENT_UI_TEMPLATE_PATH = PROJECT_ROOT / "llamaindex_shared" / "client_ui_template.html"
ADMIN_UI_TEMPLATE_PATH = PROJECT_ROOT / "llamaindex_shared" / "admin_ui_template.html"


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
    admin_href: str = ""
    api_chat_url: str = "/api/chat"
    api_session_url: str = "/api/auth/session"
    api_login_url: str = "/api/auth/login"
    api_register_url: str = "/api/auth/register"
    api_change_password_url: str = "/api/auth/change-password"
    api_delete_account_url: str = "/api/auth/delete-account"
    api_logout_url: str = "/api/auth/logout"
    new_chat_label: str = "Cuộc trò chuyện mới"
    send_button_label: str = "Gửi"
    initial_title: str = "Cuộc trò chuyện mới"
    history_label: str = "Lịch sử"
    empty_history_text: str = "Chưa có tin nhắn"
    continue_history_text: str = "Tiếp tục hội thoại"
    sending_error_prefix: str = "Không thể xử lý câu hỏi."


@dataclass(frozen=True)
class AdminUiConfig:
    current_rag_id: str
    page_title: str
    brand_badge: str
    brand_title: str
    brand_description: str
    tabs: list[ChatUiTab]
    canonical_admin_href: str = ""
    client_href: str = "/"
    api_session_url: str = "/api/auth/session"
    api_login_url: str = "/api/auth/login"
    api_change_password_url: str = "/api/auth/change-password"
    api_delete_account_url: str = "/api/auth/delete-account"
    api_logout_url: str = "/api/auth/logout"
    api_documents_url: str = "/api/admin/documents"
    api_add_data_url: str = "/api/admin/add"
    api_delete_data_url: str = "/api/admin/delete"
    api_reload_url: str = "/api/admin/reload"
    api_job_url: str = "/api/admin/jobs"
    api_system_url: str = "/api/admin/system"
    api_status_url: str = "/api/admin/status"
    api_compare_url: str = "/api/admin/compare"
    api_runtime_config_url: str = "/api/admin/runtime-config"


# Nap `client template` cho luong xu ly hien tai.
@lru_cache(maxsize=1)
## Nạp template client UI từ đĩa và cache lại cho các request sau.
def _load_client_template() -> str:
    return CLIENT_UI_TEMPLATE_PATH.read_text(encoding="utf-8")


# Nap `admin template` cho luong xu ly hien tai.
@lru_cache(maxsize=1)
## Nạp template admin UI từ đĩa và cache lại để giảm đọc file lặp lại.
def _load_admin_template() -> str:
    return ADMIN_UI_TEMPLATE_PATH.read_text(encoding="utf-8")


## Serialize config dataclass thành JSON để inject trực tiếp vào HTML template.
def _serialize_config(config: ChatUiConfig | AdminUiConfig) -> str:
    payload = asdict(config)
    payload["tabs"] = [asdict(tab) for tab in config.tabs]
    return json.dumps(payload, ensure_ascii=False)


## Lấy host chung cho 3 tab UI từ environment hoặc dùng localhost mặc định.
def _get_ui_host() -> str:
    return os.getenv("RAG_UI_HOST", "127.0.0.1")


## Lấy port UI theo biến môi trường cụ thể của từng hệ RAG.
def _get_ui_port(env_name: str, default_port: int) -> int:
    return int(os.getenv(env_name, str(default_port)))


## Dựng danh sách tab điều hướng giữa Hybrid, Baseline và GraphRAG.
def build_chat_ui_tabs() -> list[ChatUiTab]:
    host = _get_ui_host()
    return [
        ChatUiTab(id="hybrid", label="Hybrid", href=f"http://{host}:{_get_ui_port('HYBRID_UI_PORT', 8000)}/"),
        ChatUiTab(id="baseline", label="Baseline", href=f"http://{host}:{_get_ui_port('BASELINE_UI_PORT', 8001)}/"),
        ChatUiTab(id="graph", label="GraphRAG", href=f"http://{host}:{_get_ui_port('GRAPH_UI_PORT', 8502)}/"),
    ]


## Trả về URL canonical của admin UI dùng chung trên backend Hybrid.
def build_admin_ui_url() -> str:
    host = _get_ui_host()
    return f"http://{host}:{_get_ui_port('HYBRID_UI_PORT', 8000)}/admin"


## Render client chat UI bằng cách thay placeholder config trong template HTML.
def render_chat_ui(config: ChatUiConfig) -> str:
    return _load_client_template().replace("__CHAT_UI_CONFIG__", _serialize_config(config))


## Render admin UI bằng cách thay placeholder config trong template HTML.
def render_admin_ui(config: AdminUiConfig) -> str:
    return _load_admin_template().replace("__ADMIN_UI_CONFIG__", _serialize_config(config))
