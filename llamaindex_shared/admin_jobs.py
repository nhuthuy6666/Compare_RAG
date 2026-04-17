from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def create_job(*, action: str, runner: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    payload = {
        "job_id": job_id,
        "action": action,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "result": None,
        "error": None,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = payload

    thread = threading.Thread(target=_run_job, args=(job_id, runner), daemon=True)
    thread.start()
    return _public_job_payload(payload)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        payload = _JOBS.get(job_id)
        if payload is None:
            return None
        return _public_job_payload(payload)


def _run_job(job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    _update_job(job_id, status="running", error=None)
    try:
        result = runner() or {}
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), result=None)
        return
    _update_job(job_id, status="completed", error=None, result=result)


def _update_job(job_id: str, **changes: Any) -> None:
    with _JOBS_LOCK:
        payload = _JOBS[job_id]
        payload.update(changes)
        payload["updated_at"] = _now()


def _public_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": payload["job_id"],
        "action": payload["action"],
        "status": payload["status"],
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "result": payload["result"],
        "error": payload["error"],
    }
