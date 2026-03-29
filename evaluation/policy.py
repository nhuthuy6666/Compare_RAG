from __future__ import annotations

from evaluation.common import load_structured_config


DEFAULT_POLICY_PATH = "evaluation/benchmark_policy_v1.json"
DEFAULT_LOCKED_PROFILES_PATH = "evaluation/locked_profiles_v1.json"
DEFAULT_PROFILE_CANDIDATES_PATH = "evaluation/profile_candidates_v1.json"


def load_benchmark_policy(config: dict) -> dict:
    policy_path = config.get("benchmark_policy_path") or DEFAULT_POLICY_PATH
    return load_structured_config(policy_path)


def load_locked_profiles(policy: dict) -> dict:
    locked_profiles_path = policy.get("locked_profiles_path") or DEFAULT_LOCKED_PROFILES_PATH
    return load_structured_config(locked_profiles_path)


def load_profile_candidates(policy: dict) -> dict:
    profile_candidates_path = policy.get("profile_candidates_path") or DEFAULT_PROFILE_CANDIDATES_PATH
    return load_structured_config(profile_candidates_path)


def resolve_mode(policy: dict, mode: str | None) -> str:
    resolved_mode = mode or str(policy.get("default_mode") or "controlled")
    supported = set((policy.get("modes") or {}).keys())
    if resolved_mode not in supported:
        raise ValueError(f"Unsupported mode={resolved_mode!r}. Supported: {', '.join(sorted(supported))}")
    return resolved_mode


def resolve_split(policy: dict, split: str | None) -> str:
    resolved_split = split or str(policy.get("default_split") or "held_out_test")
    supported = {"all", "dev", "held_out_test"}
    if resolved_split not in supported:
        raise ValueError(f"Unsupported split={resolved_split!r}. Supported: {', '.join(sorted(supported))}")
    return resolved_split


def mode_budget(policy: dict, mode: str) -> dict:
    return dict(((policy.get("modes") or {}).get(mode) or {}).get("budget") or {})


def _shared_candidate_payload(profile_candidates: dict, profile_name: str) -> dict:
    shared_profiles = (((profile_candidates.get("controlled") or {}).get("shared_profiles")) or {})
    payload = shared_profiles.get(profile_name)
    if payload is None:
        raise KeyError(f"Unknown shared profile candidate: {profile_name}")
    return dict(payload)


def _system_candidate_payload(profile_candidates: dict, system_name: str, profile_name: str) -> dict:
    systems = ((((profile_candidates.get("best_tuned") or {}).get("systems")) or {}))
    candidates = (((systems.get(system_name) or {}).get("candidates")) or {})
    payload = candidates.get(profile_name)
    if payload is None:
        raise KeyError(f"Unknown candidate profile for {system_name}: {profile_name}")
    return dict(payload)


def resolve_profile_payload(
    *,
    locked_profiles: dict,
    profile_candidates: dict,
    mode: str,
    system_name: str,
    source: str = "locked",
    profile_name: str | None = None,
) -> dict:
    if source == "locked":
        systems = (((locked_profiles.get(mode) or {}).get("systems")) or {})
        payload = dict(systems.get(system_name) or {})
        if not payload:
            raise KeyError(f"Missing locked profile for mode={mode}, system={system_name}")
        return payload

    if source != "candidate":
        raise ValueError(f"Unsupported profile source: {source}")
    if not profile_name:
        raise ValueError("profile_name is required when source='candidate'")

    if mode == "controlled":
        candidate_payload = _shared_candidate_payload(profile_candidates, profile_name)
    else:
        candidate_payload = _system_candidate_payload(profile_candidates, system_name, profile_name)
    return {
        "profile_name": profile_name,
        "locked": False,
        "runtime_overrides": dict(candidate_payload.get("runtime_overrides") or {}),
        "selection_note": str(candidate_payload.get("description") or "").strip(),
    }


def profile_name_for_system(locked_profiles: dict, mode: str, system_name: str) -> str:
    systems = (((locked_profiles.get(mode) or {}).get("systems")) or {})
    payload = systems.get(system_name) or {}
    return str(payload.get("profile_name") or "default")
