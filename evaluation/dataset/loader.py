from __future__ import annotations

import json
from pathlib import Path

from evaluation.common import EvalExample, resolve_path


def load_examples(path_like: str | Path) -> list[EvalExample]:
    # Đọc file JSON benchmark và chuẩn hóa từng phần tử thành EvalExample.
    path = resolve_path(path_like)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("Dataset phai la list JSON.")

    examples: list[EvalExample] = []
    for item in payload:
        row = dict(item)
        # Nếu dataset không khai báo riêng answer/context keywords thì dùng expected_keywords làm mặc định.
        row.setdefault("answer_keywords", list(row.get("expected_keywords") or []))
        row.setdefault("context_keywords", list(row.get("expected_keywords") or []))
        examples.append(EvalExample(**row))
    return examples
