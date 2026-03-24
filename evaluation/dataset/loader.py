from __future__ import annotations

import json
from pathlib import Path

from evaluation.common import EvalExample, resolve_path


SUPPLEMENT_DATASET_PATH = Path(__file__).with_name("testset_additions_v1.json")
STRENGTH_BUCKET_PATH = Path(__file__).with_name("strength_buckets_v1.json")


DEFAULT_TOPIC_BUCKETS = {
    "methods": "dense_shared",
    "quota": "dense_shared",
    "scores": "dense_shared",
    "ielts": "dense_shared",
    "multi_fact": "dense_shared",
    "consulting": "graph",
    "facilities": "graph",
    "scope": "graph",
    "identity": "graph",
    "fees": "neutral",
    "refusal": "neutral",
}


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError(f"Dataset tại {path} phải là list JSON.")
    return payload


def _load_strength_buckets() -> dict[str, str]:
    if not STRENGTH_BUCKET_PATH.exists():
        return {}
    raw = STRENGTH_BUCKET_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"File bucket tại {STRENGTH_BUCKET_PATH} phải là object JSON.")
    return {str(key): str(value) for key, value in payload.items()}


def load_examples(path_like: str | Path) -> list[EvalExample]:
    # Đọc file JSON benchmark và chuẩn hóa từng phần tử thành EvalExample.
    path = resolve_path(path_like)
    payload = _load_json_list(path)
    if path.name == "testset.json":
        payload.extend(_load_json_list(SUPPLEMENT_DATASET_PATH))

    strength_buckets = _load_strength_buckets()
    examples: list[EvalExample] = []
    for item in payload:
        row = dict(item)
        # Nếu dataset không khai báo riêng answer/context keywords thì dùng expected_keywords làm mặc định.
        row.setdefault("answer_keywords", list(row.get("expected_keywords") or []))
        row.setdefault("context_keywords", list(row.get("expected_keywords") or []))
        row.setdefault(
            "strength_bucket",
            strength_buckets.get(
                str(row.get("id") or ""),
                DEFAULT_TOPIC_BUCKETS.get(str(row.get("topic") or ""), "neutral"),
            ),
        )
        examples.append(EvalExample(**row))
    return examples
