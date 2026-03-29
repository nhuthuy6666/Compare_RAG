from __future__ import annotations

import json
from typing import Any, Mapping


ALLOWED_OVERRIDE_KEYS = {
    "retrieval_top_n",
    "retrieval_similarity_threshold",
    "query_fusion_enabled",
    "query_fusion_num_queries",
    "query_fusion_mode",
    "generation_temperature",
    "generation_top_p",
    "max_output_tokens",
    "llm_seed",
}


def normalize_runtime_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}

    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in ALLOWED_OVERRIDE_KEYS or value is None:
            continue
        if key in {"retrieval_top_n", "query_fusion_num_queries", "max_output_tokens", "llm_seed"}:
            normalized[key] = int(value)
        elif key in {"retrieval_similarity_threshold", "generation_temperature", "generation_top_p"}:
            normalized[key] = float(value)
        elif key == "query_fusion_enabled":
            if isinstance(value, bool):
                normalized[key] = value
            else:
                lowered = str(value).strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    normalized[key] = True
                elif lowered in {"0", "false", "no", "off"}:
                    normalized[key] = False
                else:
                    raise ValueError(f"Invalid boolean override for {key}: {value!r}")
        else:
            normalized[key] = str(value).strip()
    return normalized


def runtime_overrides_signature(overrides: Mapping[str, Any] | None) -> str:
    normalized = normalize_runtime_overrides(overrides)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def parse_benchmark_profile_payload(payload: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not payload:
        return "default", {}
    profile = payload.get("benchmark_profile")
    if isinstance(profile, Mapping):
        profile_name = str(profile.get("profile_name") or "default").strip() or "default"
        overrides = normalize_runtime_overrides(profile.get("runtime_overrides"))
        return profile_name, overrides
    overrides = normalize_runtime_overrides(payload.get("runtime_overrides"))
    return "default", overrides
