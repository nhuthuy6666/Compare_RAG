from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


# Benchmark example with question, answer, and expected evidence.
@dataclass(frozen=True)
class EvalExample:

    id: str
    question: str
    reference_answer: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_source_hints: list[str] = field(default_factory=list)
    context_keywords: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)
    refusal_expected: bool = False
    topic: str = ""
    strength_bucket: str = ""
    split: str = "all"


# Normalized source record returned by a RAG system.
@dataclass(frozen=True)
class SourceRecord:

    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Prediction package for one benchmark example.
@dataclass(frozen=True)
class EvalPrediction:

    system: str
    example_id: str
    question: str
    answer: str
    sources: list[SourceRecord]
    latency_ms: float
    error: str | None = None


# Resolve đường dẫn tuyệt đối từ path cấu hình tương đối hoặc tuyệt đối.
def resolve_path(path_like: str | Path) -> Path:

    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


# Đọc file text UTF-8 sau khi đã resolve đường dẫn.
def load_text(path_like: str | Path) -> str:

    path = resolve_path(path_like)
    return path.read_text(encoding="utf-8")


# Nạp config/manifest JSON hoặc YAML về dict Python dùng chung.
def load_structured_config(path_like: str | Path) -> dict[str, Any]:

    raw = load_text(path_like).strip()
    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("File config không phải JSON hợp lệ và môi trường chưa có PyYAML.") from exc
        data = yaml.safe_load(raw) or {}

    if not isinstance(data, dict):
        raise ValueError("Config phải là object/dict.")
    return data


# Đọc file `.env` đơn giản theo format `KEY=VALUE`.
def load_env_file(path_like: str | Path) -> dict[str, str]:

    env: dict[str, str] = {}
    if not str(path_like or "").strip():
        return env
    path = resolve_path(path_like)
    if not path.exists() or path.is_dir():
        return env

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip().lstrip("\ufeff")] = value.strip()
    return env


# Tái sử dụng một HTTP session theo từng hệ để giữ cookie đăng nhập khi benchmark nhiều câu hỏi.
def get_http_session(system_config: dict[str, Any]) -> requests.Session:

    existing = system_config.get("__http_session")
    if isinstance(existing, requests.Session):
        return existing
    session = requests.Session()
    system_config["__http_session"] = session
    return session


# Đọc credential đánh giá từ config hoặc env, mặc định dùng tài khoản user của cụm local.
def get_eval_credentials(system_config: dict[str, Any]) -> tuple[str, str]:

    env = load_env_file(system_config.get("env_file", ""))
    username = str(
        system_config.get("auth_username")
        or env.get("NTU_EVAL_USERNAME")
        or env.get("NTU_USER_USERNAME")
        or env.get("NTU_ADMIN_USERNAME")
        or "user"
    ).strip()
    password = str(
        system_config.get("auth_password")
        or env.get("NTU_EVAL_PASSWORD")
        or env.get("NTU_USER_PASSWORD")
        or env.get("NTU_ADMIN_PASSWORD")
        or "user123"
    )
    return username, password


# Trích thông điệp lỗi dễ đọc từ response JSON/text để log benchmark không bị mơ hồ.
def response_error_message(response: requests.Response) -> str:

    try:
        payload = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        error = str(payload.get("error") or "").strip()
        if error:
            return error
    text = response.text.strip()
    if text:
        return text
    return f"HTTP {response.status_code}"


# Đăng nhập một lần cho session benchmark nếu app hiện tại yêu cầu auth cookie.
def ensure_authenticated_session(system_config: dict[str, Any], timeout: tuple[int, int]) -> requests.Session:

    session = get_http_session(system_config)
    if system_config.get("workspace_slug"):
        return session

    auth_mode = str(system_config.get("auth_mode") or "session").strip().lower()
    if auth_mode in {"none", "off", "disabled"}:
        system_config["__auth_ready"] = True
        return session
    if system_config.get("__auth_ready"):
        return session

    base_url = str(system_config["base_url"]).rstrip("/")
    login_endpoint = str(system_config.get("login_endpoint") or "/api/auth/login").strip() or "/api/auth/login"
    username, password = get_eval_credentials(system_config)
    response = session.post(
        f"{base_url}{login_endpoint}",
        json={"username": username, "password": password},
        timeout=timeout,
    )

    # Một số endpoint cũ không có auth; khi đó cứ dùng session trần để giữ tương thích ngược.
    if response.status_code == 404:
        system_config["__auth_ready"] = True
        return session

    if response.status_code >= 400:
        detail = response_error_message(response)
        raise RuntimeError(f"Không đăng nhập được cho evaluator tại {base_url}: HTTP {response.status_code} - {detail}")

    system_config["__auth_ready"] = True
    return session


# Bỏ dấu tiếng Việt để việc so khớp text bền hơn trước biến thể nhập liệu.
def strip_accents(text: str) -> str:

    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


# Chuẩn hóa text về dạng lower-case, bỏ dấu câu và rút gọn khoảng trắng.
def normalize_text(text: str) -> str:

    lowered = strip_accents(text).lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


# Tách token đơn giản từ text đã normalize.
def tokenize(text: str) -> list[str]:

    return [token for token in normalize_text(text).split() if token]


# Rút các chuỗi số để kiểm tra claim định lượng trong câu trả lời.
def extract_numbers(text: str) -> list[str]:

    return re.findall(r"\d+(?:[.,]\d+)?", text)


# Nối toàn bộ source thành một chuỗi lớn để dò keyword hoặc source hint.
def flatten_sources_text(sources: list[SourceRecord]) -> str:

    parts: list[str] = []
    for source in sources:
        parts.append(source.label)
        parts.append(source.content)
        if source.metadata:
            parts.append(json.dumps(source.metadata, ensure_ascii=False))
    return "\n".join(part for part in parts if part).strip()


# Phát hiện các mẫu câu từ chối hoặc thiếu thông tin trong câu trả lời.
def refusal_detected(answer: str) -> bool:

    answer_norm = normalize_text(answer)
    signals = (
        "khong tim thay chac chan",
        "chua tim thay chac chan",
        "khong co thong tin",
        "chua co thong tin",
        "chua thay du thong tin",
        "khong tim thay trong tai lieu",
        "chua tim thay trong tai lieu",
    )
    return any(signal in answer_norm for signal in signals)


# Tạo thư mục đầu ra nếu chưa tồn tại và trả lại đường dẫn đã resolve.
def ensure_dir(path_like: str | Path) -> Path:

    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Ghi payload ra file JSON đẹp để dễ đọc và diff.
def write_json(path_like: str | Path, payload: Any) -> Path:

    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# Ghi danh sách row ra CSV với header là hợp của toàn bộ khóa xuất hiện.
def write_csv(path_like: str | Path, rows: list[dict[str, Any]]) -> Path:

    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


# Đệ quy chuyển dataclass/list/dict về kiểu thuần Python để dump JSON.
def dataclass_to_dict(obj: Any) -> Any:

    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {key: dataclass_to_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj
