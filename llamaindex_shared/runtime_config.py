from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = PROJECT_ROOT / ".runtime_config.json"
PRIMARY_SCOPE = "shared"
RUNTIME_SCOPES = ("shared", "baseline", "hybrid", "graph")
LEGACY_SCOPES = tuple(scope for scope in RUNTIME_SCOPES if scope != PRIMARY_SCOPE)


FIELD_DEFINITIONS: dict[str, list[dict[str, Any]]] = {
    "shared": [
        {"key": "llm_base_url", "label": "LLM Base URL", "type": "string"},
        {"key": "llm_model", "label": "Chat Model", "type": "string"},
        {"key": "embed_model", "label": "Embedding Model", "type": "string"},
        {"key": "llm_timeout", "label": "LLM Timeout (s)", "type": "int"},
        {"key": "embed_timeout", "label": "Embedding Timeout (s)", "type": "int"},
        {"key": "retrieval_top_n", "label": "Retrieval Top N", "type": "int"},
        {"key": "retrieval_similarity_threshold", "label": "Similarity Threshold", "type": "float"},
        {"key": "query_fusion_enabled", "label": "Enable Query Fusion", "type": "bool"},
        {"key": "query_fusion_num_queries", "label": "Fusion Query Count", "type": "int"},
        {"key": "query_fusion_mode", "label": "Fusion Mode", "type": "string"},
        {"key": "generation_temperature", "label": "Temperature", "type": "float"},
        {"key": "generation_top_p", "label": "Top P", "type": "float"},
        {"key": "max_output_tokens", "label": "Max Output Tokens", "type": "int"},
        {"key": "llm_seed", "label": "LLM Seed", "type": "int"},
        {"key": "prompt", "label": "Shared Prompt", "type": "multiline"},
        {"key": "query_refusal_response", "label": "Refusal Response", "type": "multiline"},
        {"key": "graph_vector_candidates", "label": "Graph Vector Candidates", "type": "int"},
        {"key": "graph_neighbor_hops", "label": "Graph Neighbor Hops", "type": "int"},
        {"key": "graph_neighbor_facts_limit", "label": "Graph Neighbor Fact Limit", "type": "int"},
    ],
    "baseline": [],
    "hybrid": [],
    "graph": [],
}


# Tạo index `field_key -> field_type` để parse payload cập nhật nhanh hơn.
def _field_type_index() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for fields in FIELD_DEFINITIONS.values():
        for field in fields:
            mapping[field["key"]] = field["type"]
    return mapping


FIELD_TYPES = _field_type_index()


# Tạo state runtime config rỗng với đầy đủ mọi scope chuẩn của hệ thống.
def _empty_payload() -> dict[str, dict[str, Any]]:
    return {scope: {} for scope in RUNTIME_SCOPES}


# Gom cac scope cu ve scope `shared` de runtime config luon duoc ap dung chung cho 3 RAG.
def _collapse_legacy_scopes(payload: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    collapsed = _empty_payload()
    shared = dict(payload.get(PRIMARY_SCOPE) or {})
    for scope in LEGACY_SCOPES:
        for key, value in (payload.get(scope) or {}).items():
            if key not in shared:
                shared[key] = value
    collapsed[PRIMARY_SCOPE] = shared
    return collapsed


# Đọc file runtime config từ đĩa và chuẩn hóa về đúng cấu trúc scope.
def load_runtime_config_state() -> dict[str, dict[str, Any]]:
    if not RUNTIME_CONFIG_PATH.exists():
        return _empty_payload()
    try:
        payload = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_payload()
    if not isinstance(payload, dict):
        return _empty_payload()
    normalized = _empty_payload()
    for scope in RUNTIME_SCOPES:
        values = payload.get(scope)
        if isinstance(values, dict):
            normalized[scope] = dict(values)
    return _collapse_legacy_scopes(normalized)


# Ghi runtime config xuống đĩa sau khi lọc các field rỗng hoặc không hợp lệ.
def save_runtime_config_state(payload: dict[str, dict[str, Any]]) -> None:
    cleaned = _empty_payload()
    collapsed = _collapse_legacy_scopes(payload)
    for scope in RUNTIME_SCOPES:
        values = collapsed.get(scope) or {}
        if not isinstance(values, dict):
            continue
        cleaned[scope] = {
            str(key): value
            for key, value in values.items()
            if str(key) in FIELD_TYPES and value is not None and value != ""
        }
    for scope in LEGACY_SCOPES:
        cleaned[scope] = {}
    RUNTIME_CONFIG_PATH.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# Parse giá trị boolean từ payload form/JSON theo nhiều cách nhập phổ biến.
def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Giá trị boolean không hợp lệ: {value!r}")


# Parse giá trị runtime theo type khai báo của field tương ứng.
def _parse_value(field_key: str, value: Any) -> Any:
    field_type = FIELD_TYPES[field_key]
    if value in (None, ""):
        return None
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return float(value)
    if field_type == "bool":
        return _parse_bool(value)
    return str(value).strip()


# Chuan hoa ten scope tu payload cu ve scope `shared` duy nhat.
def _resolve_scope(scope: str) -> str:
    normalized_scope = str(scope or "").strip()
    if normalized_scope not in RUNTIME_SCOPES:
        raise ValueError(f"Scope cấu hình không hợp lệ: {normalized_scope}")
    return PRIMARY_SCOPE


# Loc payload gia tri runtime de UI co the hien thi "gia tri dang chay" hien tai.
def serialize_runtime_values(values: dict[str, Any] | None) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in (values or {}).items():
        normalized_key = str(key)
        if normalized_key not in FIELD_TYPES or value is None or value == "":
            continue
        serialized[normalized_key] = value
    return serialized


# Lọc và chuẩn hóa payload cập nhật của một scope trước khi ghi xuống state.
def normalize_scope_updates(scope: str, values: dict[str, Any]) -> dict[str, Any]:
    resolved_scope = _resolve_scope(scope)
    allowed_keys = {field["key"] for field in FIELD_DEFINITIONS.get(resolved_scope, [])}
    normalized: dict[str, Any] = {}
    for key, value in (values or {}).items():
        key = str(key)
        if key not in allowed_keys or key not in FIELD_TYPES:
            continue
        normalized[key] = _parse_value(key, value)
    return normalized


# Áp cập nhật runtime config cho một scope và persist ngay xuống file JSON.
def update_runtime_scope(scope: str, values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state = load_runtime_config_state()
    resolved_scope = _resolve_scope(scope)
    normalized_updates = normalize_scope_updates(scope, values)
    current = dict(state.get(resolved_scope) or {})
    for key, value in normalized_updates.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    state[resolved_scope] = current
    save_runtime_config_state(state)
    return state


# Hợp nhất override của scope `shared` với override riêng của từng RAG.
def get_runtime_overrides_for_rag(rag_id: str) -> dict[str, Any]:
    state = load_runtime_config_state()
    return dict(state.get(PRIMARY_SCOPE) or {})


# Trả về payload đầy đủ để admin UI render form runtime config.
def build_runtime_config_payload(*, current_values: dict[str, Any] | None = None) -> dict[str, Any]:
    state = load_runtime_config_state()
    return {
        "scopes": state,
        "definitions": FIELD_DEFINITIONS,
        "config_path": str(RUNTIME_CONFIG_PATH),
        "effective_scope": PRIMARY_SCOPE,
        "current_values": serialize_runtime_values(current_values),
    }
