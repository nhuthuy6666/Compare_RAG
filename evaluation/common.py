from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = PROJECT_ROOT / "evaluation"


# Mô tả một câu hỏi benchmark cùng các tín hiệu ground-truth phục vụ chấm điểm.
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


# Chuẩn hóa một nguồn được hệ RAG trả về để các metric xử lý thống nhất.
@dataclass(frozen=True)
class SourceRecord:
    label: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Gói kết quả dự đoán của một hệ cho một câu hỏi benchmark.
@dataclass(frozen=True)
class EvalPrediction:
    system: str
    example_id: str
    question: str
    answer: str
    sources: list[SourceRecord]
    latency_ms: float
    error: str | None = None


def resolve_path(path_like: str | Path) -> Path:
    # Cho phép cấu hình dùng cả đường dẫn tuyệt đối lẫn tương đối từ project root.
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_text(path_like: str | Path) -> str:
    # Đọc text UTF-8 từ đường dẫn đã được resolve.
    path = resolve_path(path_like)
    return path.read_text(encoding="utf-8")


def load_structured_config(path_like: str | Path) -> dict[str, Any]:
    # Hỗ trợ config ở dạng JSON hoặc YAML.
    raw = load_text(path_like).strip()
    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "config.yaml khong phai JSON hop le va moi truong chua co PyYAML."
            ) from exc
        data = yaml.safe_load(raw) or {}

    if not isinstance(data, dict):
        raise ValueError("Config phai la object/dict.")
    return data


def load_env_file(path_like: str | Path) -> dict[str, str]:
    # Đọc file .env đơn giản theo format KEY=VALUE.
    env: dict[str, str] = {}
    path = resolve_path(path_like)
    if not path.exists():
        return env

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip().lstrip("\ufeff")] = value.strip()
    return env


def strip_accents(text: str) -> str:
    # Bỏ dấu để việc so khớp text ít phụ thuộc vào biến thể gõ tiếng Việt.
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    # Chuẩn hóa về lower-case, bỏ dấu câu và rút gọn khoảng trắng trước khi so khớp.
    lowered = strip_accents(text).lower().strip()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def tokenize(text: str) -> list[str]:
    # Tách token dựa trên chuỗi đã normalize.
    return [token for token in normalize_text(text).split() if token]


def extract_numbers(text: str) -> list[str]:
    # Rút các chuỗi số để kiểm tra claim chứa số liệu có thật trong nguồn hay không.
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def flatten_sources_text(sources: list[SourceRecord]) -> str:
    # Nối toàn bộ label/content/metadata của nguồn thành một text lớn để dò hint hoặc keyword.
    parts: list[str] = []
    for source in sources:
        parts.append(source.label)
        parts.append(source.content)
        if source.metadata:
            parts.append(json.dumps(source.metadata, ensure_ascii=False))
    return "\n".join(part for part in parts if part).strip()


def refusal_detected(answer: str) -> bool:
    # Xác định model có đang “từ chối trả lời” dựa trên một số mẫu câu cố định.
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


def ensure_dir(path_like: str | Path) -> Path:
    # Tạo thư mục đầu ra nếu chưa tồn tại.
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path_like: str | Path, payload: Any) -> Path:
    # Ghi JSON đẹp để dễ đọc và lưu vết từng lần benchmark.
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_csv(path_like: str | Path, rows: list[dict[str, Any]]) -> Path:
    # Ghi CSV với header là hợp của toàn bộ khóa xuất hiện trong danh sách rows.
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


def dataclass_to_dict(obj: Any) -> Any:
    # Đệ quy chuyển dataclass/list/dict sang kiểu nguyên thủy để dump JSON.
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {key: dataclass_to_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    return obj
