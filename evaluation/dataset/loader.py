from __future__ import annotations

import json
from pathlib import Path

from evaluation.common import EvalExample, resolve_path


SUPPLEMENT_DATASET_PATH = Path(__file__).with_name("testset_additions_v1.json")
STRENGTH_BUCKET_PATH = Path(__file__).with_name("strength_buckets_v1.json")
SPLIT_ASSIGNMENT_PATH = Path(__file__).with_name("splits_v1.json")


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


def _load_split_assignments() -> dict[str, set[str]]:
    if not SPLIT_ASSIGNMENT_PATH.exists():
        return {}
    raw = SPLIT_ASSIGNMENT_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"File split tại {SPLIT_ASSIGNMENT_PATH} phải là object JSON.")

    assignments: dict[str, set[str]] = {}
    for split_name, example_ids in payload.items():
        if not isinstance(example_ids, list):
            raise ValueError(f"Split `{split_name}` phải là list example_id.")
        assignments[str(split_name)] = {str(example_id) for example_id in example_ids}
    return assignments


def _resolve_example_split(example_id: str, split_assignments: dict[str, set[str]]) -> str:
    for split_name, example_ids in split_assignments.items():
        if example_id in example_ids:
            return split_name
    return "all"


def load_examples(path_like: str | Path, *, split: str = "all") -> list[EvalExample]:
    # Đọc file JSON benchmark, gán bucket/split và cho phép lọc theo dev hoặc held_out_test.
    path = resolve_path(path_like)
    payload = _load_json_list(path)
    if path.name == "testset.json":
        payload.extend(_load_json_list(SUPPLEMENT_DATASET_PATH))

    strength_buckets = _load_strength_buckets()
    split_assignments = _load_split_assignments()
    known_splits = {"all", *split_assignments.keys()}
    if split not in known_splits:
        raise ValueError(f"Unknown split={split!r}. Supported: {', '.join(sorted(known_splits))}")

    examples: list[EvalExample] = []
    for item in payload:
        row = dict(item)
        row.setdefault("answer_keywords", list(row.get("expected_keywords") or []))
        row.setdefault("context_keywords", list(row.get("expected_keywords") or []))
        row.setdefault(
            "strength_bucket",
            strength_buckets.get(
                str(row.get("id") or ""),
                DEFAULT_TOPIC_BUCKETS.get(str(row.get("topic") or ""), "neutral"),
            ),
        )
        row["split"] = _resolve_example_split(str(row.get("id") or ""), split_assignments)
        if split != "all" and row["split"] != split:
            continue
        examples.append(EvalExample(**row))
    return examples
