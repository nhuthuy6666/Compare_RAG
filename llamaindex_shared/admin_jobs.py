from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


_JOBS_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_JOB_CONTEXT = threading.local()


## Lấy timestamp hiện tại cho metadata của job.
def _now() -> float:
    return time.time()


## Tạo một background job mới và trả về payload public ngay cho UI.
def create_job(
    *,
    action: str,
    runner: Callable[[], dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    parent_job_id: str | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    payload = {
        "job_id": job_id,
        "action": action,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "completed_at": None,
        "progress": 0,
        "stage": "queued",
        "detail": "",
        "result": None,
        "error": None,
        "metadata": metadata or {},
        "parent_job_id": parent_job_id,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = payload

    thread = threading.Thread(target=_run_job, args=(job_id, runner), daemon=True)
    thread.start()
    return _public_job_payload(payload)


## Đọc trạng thái của một job theo `job_id`.
def get_job(job_id: str) -> dict[str, Any] | None:
    with _JOBS_LOCK:
        payload = _JOBS.get(job_id)
        if payload is None:
            return None
        return _public_job_payload(payload)


## Liệt kê các job gần nhất để trang admin có thể theo dõi tiến trình.
def list_jobs(*, limit: int = 50) -> list[dict[str, Any]]:
    with _JOBS_LOCK:
        rows = sorted(_JOBS.values(), key=lambda item: (item["created_at"], item["job_id"]), reverse=True)
    return [_public_job_payload(payload) for payload in rows[: max(1, limit)]]


## Cập nhật tiến độ cho job hiện hành từ bên trong runner đang chạy.
def set_job_progress(
    *,
    stage: str,
    progress: int,
    detail: str = "",
    result: dict[str, Any] | None = None,
) -> None:
    job_id = getattr(_JOB_CONTEXT, "job_id", None)
    if not job_id:
        return
    changes: dict[str, Any] = {
        "stage": str(stage or "").strip() or "running",
        "progress": max(0, min(100, int(progress))),
        "detail": str(detail or "").strip(),
    }
    if result is not None:
        changes["result"] = result
    _update_job(job_id, **changes)


## Chạy phần việc thật của job trong thread nền và ghi nhận success/failure.
def _run_job(job_id: str, runner: Callable[[], dict[str, Any]]) -> None:
    _JOB_CONTEXT.job_id = job_id
    _update_job(
        job_id,
        status="running",
        error=None,
        started_at=_now(),
        progress=5,
        stage="running",
        detail="Đang xử lý...",
    )
    try:
        result = runner() or {}
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            result=None,
            completed_at=_now(),
            stage="failed",
            detail=str(exc),
        )
        _JOB_CONTEXT.job_id = None
        return
    _update_job(
        job_id,
        status="completed",
        error=None,
        result=result,
        completed_at=_now(),
        progress=100,
        stage="completed",
        detail="Hoàn tất.",
    )
    _JOB_CONTEXT.job_id = None


## Ghi đè một phần trạng thái job và tự cập nhật `updated_at`.
def _update_job(job_id: str, **changes: Any) -> None:
    with _JOBS_LOCK:
        payload = _JOBS[job_id]
        payload.update(changes)
        payload["updated_at"] = _now()


## Rút gọn payload nội bộ thành dữ liệu public an toàn cho API.
def _public_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": payload["job_id"],
        "action": payload["action"],
        "status": payload["status"],
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "progress": payload.get("progress", 0),
        "stage": payload.get("stage", ""),
        "detail": payload.get("detail", ""),
        "result": payload["result"],
        "error": payload["error"],
        "metadata": payload.get("metadata") or {},
        "parent_job_id": payload.get("parent_job_id"),
    }
