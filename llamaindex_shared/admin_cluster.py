from __future__ import annotations

import json
import time
from urllib import error, request


def build_cluster_server_urls() -> dict[str, str]:
    import os

    host = os.getenv("RAG_UI_HOST", "127.0.0.1")
    return {
        "hybrid": f"http://{host}:{os.getenv('HYBRID_UI_PORT', '8000')}",
        "baseline": f"http://{host}:{os.getenv('BASELINE_UI_PORT', '8001')}",
        "graph": f"http://{host}:{os.getenv('GRAPH_UI_PORT', '8502')}",
    }


def post_json(url: str, payload: dict | None = None, *, timeout: int | None = None) -> dict:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        open_kwargs = {"timeout": timeout} if timeout is not None else {}
        with request.urlopen(req, **open_kwargs) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw or "{}")
    except error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        raise RuntimeError(payload.get("error") or f"HTTP {exc.code} calling {url}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Không kết nối được tới {url}: {exc.reason}") from exc


def get_json(url: str, *, timeout: int | None = None) -> dict:
    req = request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        open_kwargs = {"timeout": timeout} if timeout is not None else {}
        with request.urlopen(req, **open_kwargs) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw or "{}")
    except error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8")
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        raise RuntimeError(payload.get("error") or f"HTTP {exc.code} calling {url}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Không kết nối được tới {url}: {exc.reason}") from exc


def wait_for_job(job_url: str, *, poll_interval_seconds: float = 2.0) -> dict:
    while True:
        payload = get_json(job_url)
        status = str(payload.get("status") or "").strip().lower()
        if status == "completed":
            return payload
        if status == "failed":
            raise RuntimeError(str(payload.get("error") or f"Job thất bại: {job_url}"))
        if status not in {"queued", "running"}:
            raise RuntimeError(f"Trạng thái job không hợp lệ tại {job_url}: {status or 'unknown'}")
        time.sleep(poll_interval_seconds)
