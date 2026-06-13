from __future__ import annotations

import time
from typing import Any, Callable

from llamaindex_shared.admin_cluster import build_cluster_server_urls, get_json, post_json


# Thu thập trạng thái của cả 3 hệ RAG để admin UI hiển thị health tổng thể.
def collect_cluster_status(
    *,
    current_rag_id: str,
    local_status_builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    cluster: dict[str, Any] = {}
    for rag_id, base_url in build_cluster_server_urls().items():
        if rag_id == current_rag_id:
            try:
                cluster[rag_id] = local_status_builder()
            except Exception as exc:
                cluster[rag_id] = {"rag_id": rag_id, "health": "error", "error": str(exc)}
            continue
        try:
            cluster[rag_id] = get_json(f"{base_url}/api/admin/status")
        except Exception as exc:
            cluster[rag_id] = {
                "rag_id": rag_id,
                "port": _extract_port(base_url),
                "health": "offline",
                "error": str(exc),
            }
    return {"systems": cluster}


# Chạy cùng một câu hỏi trên 3 backend để so sánh answer, sources và latency.
def compare_cluster_answers(
    *,
    question: str,
    current_rag_id: str,
    local_answer_builder: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for rag_id, base_url in build_cluster_server_urls().items():
        started_at = time.perf_counter()
        try:
            if rag_id == current_rag_id:
                payload = local_answer_builder(question)
            else:
                payload = post_json(f"{base_url}/api/chat", {"query": question})
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            results[rag_id] = {
                "status": "ok",
                "answer": str(payload.get("answer") or "").strip(),
                "sources": list(payload.get("sources") or []),
                "latency_ms": duration_ms,
            }
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            results[rag_id] = {
                "status": "error",
                "error": str(exc),
                "answer": "",
                "sources": [],
                "latency_ms": duration_ms,
            }
    return {"question": question, "results": results}


# Tách port từ base URL để vẫn hiện được thông tin tối thiểu khi backend offline.
def _extract_port(base_url: str) -> int | None:
    try:
        return int(str(base_url).rsplit(":", 1)[-1])
    except Exception:
        return None
