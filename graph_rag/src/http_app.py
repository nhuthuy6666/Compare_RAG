from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from types import SimpleNamespace
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chat_service import answer_question, warm_up_graph
from config import build_neo4j_driver, load_config
from ingest import _load_records, _run_ingest_pipeline, sync_shared_chunk_changes
from llamaindex_shared import (
    AdminUiConfig,
    ChatUiConfig,
    authenticate_user,
    build_admin_ui_url,
    build_chat_ui_tabs,
    build_cluster_server_urls,
    build_logout_cookie,
    build_runtime_config_payload,
    build_session_cookie,
    collect_cluster_status,
    compare_cluster_answers,
    create_job,
    delete_corpus_documents,
    get_default_accounts_hint,
    get_job,
    has_role,
    is_internal_cluster_request,
    list_corpus_documents,
    list_jobs,
    load_document_ids_from_payload,
    post_json,
    read_session_from_cookie,
    render_admin_ui,
    render_chat_ui,
    set_job_progress,
    update_runtime_scope,
    wait_for_job,
    add_corpus_documents,
)
from llamaindex_shared.benchmark_runtime import parse_benchmark_profile_payload
from neo4j_store import graph_ready
from utils import configure_console_utf8


configure_console_utf8()

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8502
CURRENT_RAG_ID = "graph"
GRAPH_LOCK = Lock()
ADMIN_LOCK = Lock()
EMPTY_GRAPH_ERROR_ASCII_MARKERS = ("Graph Neo4j", "reset-graph")


# Helper cho `graph_has_data` trong module nay.
def _graph_has_data(config) -> bool:
    warm_up_graph.cache_clear()
    driver = build_neo4j_driver(config)
    try:
        return graph_ready(driver, config)
    finally:
        driver.close()


# Helper cho `graph_fact_count` trong module nay.
def _graph_fact_count(config) -> int:
    driver = build_neo4j_driver(config)
    try:
        with driver.session(database=config.neo4j_database) as session:
            row = session.run("MATCH (f:Fact) RETURN count(f) AS total").single()
            return int((row or {}).get("total") or 0)
    finally:
        driver.close()


# Kiem tra dieu kien `empty graph reload error` trong boi canh hien tai.
def _is_empty_graph_reload_error(exc: Exception) -> bool:
    message = str(exc)
    return all(marker in message for marker in EMPTY_GRAPH_ERROR_ASCII_MARKERS)


# Dung `empty graph reload result` cho luong xu ly hien tai.
def _build_empty_graph_reload_result(action: str) -> dict[str, str]:
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": "0",
        "message": f"Đã {action} đồng bộ graph. Neo4j hiện không còn fact nào sau khi cập nhật.",
    }


# Chay `reload job safe` cho luong xu ly hien tai.
def _run_reload_job_safe(sync_payload: dict | None = None) -> dict[str, str]:
    try:
        return run_reload_job(sync_payload)
    except Exception as exc:
        if _is_empty_graph_reload_error(exc):
            action = "xóa dữ liệu và" if sync_payload and sync_payload.get("deleted_relative_paths") else "reload và"
            return _build_empty_graph_reload_result(action)
        raise


# Xu ly `to payload` cho luong xu ly hien tai.
def _answer_to_payload(question: str, result) -> dict:
    return {
        "answer": result.answer,
        "rewritten_query": question,
        "sources": [
            {
                "source": fact.relative_path or fact.heading_path or "graph_fact",
                "relative_path": fact.relative_path,
                "heading_path": fact.heading_path,
                "content": fact.content,
                "score": fact.score,
            }
            for fact in result.facts
        ],
    }


# Nap `ui html` cho luong xu ly hien tai.
@lru_cache(maxsize=1)
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id=CURRENT_RAG_ID,
            page_title="NTU Graph RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description="Giao diện người dùng tập trung vào hỏi đáp, lịch sử hội thoại và chuyển nhanh giữa Baseline, Hybrid và GraphRAG.",
            header_badge="Graph RAG",
            header_subtitle="Neo4j graph retrieval | Query fusion | Fact graph",
            assistant_label="Graph NTU Bot",
            empty_title="Graph RAG sẵn sàng",
            empty_description="Đặt cùng một câu hỏi trên cả 3 tab để so sánh câu trả lời và nguồn tham chiếu.",
            placeholder="Ví dụ: Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
            composer_hint="Dùng cùng một câu hỏi ở cả 3 tab để so sánh",
            loading_message="Đang truy xuất graph facts...",
            ready_message="Đã hoàn tất.",
            storage_key="ntu_fusion_graph_sessions",
            admin_href=build_admin_ui_url(),
            suggestions=[
                "Ngành Marketing 2025 có bao nhiêu chỉ tiêu?",
                "Học phí dự kiến một năm là bao nhiêu?",
                "Hồ sơ nhập học gồm những gì?",
            ],
            tabs=build_chat_ui_tabs(),
        )
    )


# Nap `admin html` cho luong xu ly hien tai.
@lru_cache(maxsize=1)
def load_admin_html() -> str:
    return render_admin_ui(
        AdminUiConfig(
            current_rag_id=CURRENT_RAG_ID,
            page_title="NTU Shared Admin",
            brand_badge="NTU Admin",
            brand_title="Bảng điều khiển quản trị",
            brand_description="Giao diện quản trị dùng chung cho nhập liệu, theo dõi tác vụ, so sánh 3 RAG và cấu hình runtime.",
            tabs=build_chat_ui_tabs(),
            canonical_admin_href=build_admin_ui_url(),
        )
    )


def build_current_runtime_config_payload() -> dict:
    config = load_config()
    return build_runtime_config_payload(
        current_values={
            "llm_base_url": config.llm_base_url,
            "llm_model": config.llm_model,
            "embed_model": config.embed_model,
            "llm_timeout": config.llm_timeout,
            "embed_timeout": config.embed_timeout,
            "retrieval_top_n": config.retrieval_top_n,
            "retrieval_similarity_threshold": config.retrieval_similarity_threshold,
            "query_fusion_enabled": config.query_fusion_enabled,
            "query_fusion_num_queries": config.query_fusion_num_queries,
            "query_fusion_mode": config.query_fusion_mode,
            "generation_temperature": config.generation_temperature,
            "generation_top_p": config.generation_top_p,
            "max_output_tokens": config.max_output_tokens,
            "llm_seed": config.llm_seed,
            "prompt": config.shared_prompt,
            "query_refusal_response": config.query_refusal_response,
            "graph_vector_candidates": config.graph_vector_candidates,
            "graph_neighbor_hops": config.graph_neighbor_hops,
            "graph_neighbor_facts_limit": config.graph_neighbor_facts_limit,
        }
    )


# Reload `local resources` cho luong xu ly hien tai.
def reload_local_resources() -> dict[str, str]:
    config = load_config()
    args = SimpleNamespace(chunk_jsonl_root=None, output_dir=None, limit=None, dry_run=False, reset_graph=True)
    all_records = _load_records(args, config, verbose=False)
    summary = _run_ingest_pipeline(
        all_records=all_records,
        config=config,
        reset_graph_first=True,
        dry_run=False,
        verbose=False,
    )
    _graph_has_data(config)
    warm_up_graph.cache_clear()
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": str(summary["fact_count"]),
        "message": (
            f"Đã reload graph với {summary['chunk_count']} chunk, "
            f"{summary['entity_count']} entity và {summary['fact_count']} fact."
        ),
    }


# Làm mới trạng thái GraphRAG mà không rebuild lại Neo4j khi dữ liệu không thay đổi.
def refresh_local_resources() -> dict[str, str]:
    config = load_config()
    graph_is_ready = _graph_has_data(config)
    fact_count = _graph_fact_count(config)
    warm_up_graph.cache_clear()
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": str(fact_count),
        "message": (
            f"ÄÃ£ lÃ m má»›i tráº¡ng thÃ¡i GraphRAG. "
            f"Neo4j {'sáºµn sÃ ng' if graph_is_ready else 'chÆ°a sáºµn sÃ ng'} vá»›i {fact_count} fact."
        ),
    }


# Dung `graph sync payload` cho luong xu ly hien tai.
def _build_graph_sync_payload(summary: dict | None) -> dict[str, list[str] | bool]:
    summary = summary or {}
    chunk_relative_paths: list[str] = []
    deleted_relative_paths: list[str] = []
    for item in list((summary.get("added") or {}).get("web") or []):
        chunk_relative_path = str(item.get("chunk_relative_path") or "").strip()
        if chunk_relative_path:
            chunk_relative_paths.append(chunk_relative_path)
    for item in list((summary.get("added") or {}).get("pdf") or []):
        chunk_relative_path = str(item.get("chunk_relative_path") or "").strip()
        if chunk_relative_path:
            chunk_relative_paths.append(chunk_relative_path)
    for item in list(summary.get("removed") or []):
        txt_relative_path = str(item.get("txt_relative_path") or "").strip()
        if txt_relative_path:
            deleted_relative_paths.append(txt_relative_path.replace("data_txt/", "", 1))
    return {
        "full_reload": False,
        "chunk_relative_paths": chunk_relative_paths,
        "deleted_relative_paths": deleted_relative_paths,
    }


# Dong bo `local resources` cho luong xu ly hien tai.
def sync_local_resources(*, chunk_relative_paths: list[str] | None = None, deleted_relative_paths: list[str] | None = None) -> dict[str, str]:
    config = load_config()
    summary = sync_shared_chunk_changes(
        config=config,
        chunk_relative_paths=chunk_relative_paths,
        deleted_relative_paths=deleted_relative_paths,
        verbose=False,
    )
    graph_has_data = _graph_has_data(config)
    warm_up_graph.cache_clear()
    if not graph_has_data:
        summary["message"] = f"{summary.get('message') or 'Đã đồng bộ delta graph.'} Neo4j hiện không còn fact nào sau khi đồng bộ."
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": str(summary["fact_count"]),
        "message": str(summary.get("message") or "Đã đồng bộ delta graph."),
    }


# Reload `cluster resources` cho luong xu ly hien tai.
def reload_cluster_resources(graph_sync_payload: dict | None = None) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    remote_targets = [rag_id for rag_id in build_cluster_server_urls() if rag_id != CURRENT_RAG_ID]
    set_job_progress(stage="local_reload", progress=20, detail="Đang reload GraphRAG cục bộ.")
    local_summary = _run_reload_job_safe(graph_sync_payload)
    results[CURRENT_RAG_ID] = {"status": "ok", "message": local_summary["message"]}
    if not remote_targets:
        return results
    processed = 0
    for rag_id, base_url in build_cluster_server_urls().items():
        if rag_id == CURRENT_RAG_ID:
            continue
        processed += 1
        progress = 35 + int((processed / max(1, len(remote_targets))) * 45)
        set_job_progress(stage="cluster_reload", progress=progress, detail=f"Đang đồng bộ backend {rag_id}.")
        try:
            job = post_json(f"{base_url}/api/admin/reload", {"async_mode": True})
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                raise RuntimeError("Backend không trả về job_id reload.")
            payload = wait_for_job(f"{base_url}/api/admin/jobs/{job_id}")
            result = payload.get("result") or {}
            results[rag_id] = {"status": "ok", "message": str(result.get("message") or "Đã reload.")}
        except Exception as exc:
            results[rag_id] = {"status": "error", "message": str(exc)}
    set_job_progress(stage="cluster_reload", progress=90, detail="Đã gửi reload tới các backend còn lại.")
    return results


# Dam bao `reload success` cho luong xu ly hien tai.
def ensure_reload_success(reloads: dict[str, dict[str, str]], *, action_label: str) -> None:
    failed = {}
    for rag_id, result in reloads.items():
        status = result.get("status")
        message = str(result.get("message") or "")
        if status != "ok" and not (rag_id == "graph" and all(marker in message for marker in EMPTY_GRAPH_ERROR_ASCII_MARKERS)):
            failed[rag_id] = result
    if not failed:
        return
    details = " | ".join(f"{rag_id}: {result.get('message') or 'Lỗi chưa xác định'}" for rag_id, result in failed.items())
    raise RuntimeError(f"Đã {action_label} dữ liệu thô nhưng chưa reload xong toàn bộ hệ thống. {details}")


# Helper cho `start_cluster_reload_job` trong module nay.
def start_cluster_reload_job(*, graph_sync_payload: dict | None, action_label: str) -> dict:
    return create_job(
        action="cluster_reload",
        runner=lambda: _run_cluster_reload_job(graph_sync_payload=graph_sync_payload, action_label=action_label),
        metadata={"rag_id": CURRENT_RAG_ID, "action_label": action_label},
    )


# Chay `cluster reload job` cho luong xu ly hien tai.
def _run_cluster_reload_job(*, graph_sync_payload: dict | None, action_label: str) -> dict:
    reloads = reload_cluster_resources(graph_sync_payload)
    ensure_reload_success(reloads, action_label=action_label)
    return {
        "reloads": reloads,
        "message": f"Đã reload xong 3 hệ sau khi {action_label} dữ liệu.",
    }


# Warm-up `resources in background` cho luong xu ly hien tai.
def warm_resources_in_background() -> None:
    try:
        warm_up_graph.cache_clear()
        warm_up_graph()
        print("[graph] Warm-up hoàn tất.")
    except Exception as exc:
        print(f"[graph] Warm-up lỗi: {exc}")


# Chay `reload job` cho luong xu ly hien tai.
def run_reload_job(sync_payload: dict | None = None) -> dict[str, str]:
    with GRAPH_LOCK:
        payload = sync_payload or {}
        chunk_relative_paths = list(payload.get("chunk_relative_paths") or [])
        deleted_relative_paths = list(payload.get("deleted_relative_paths") or [])
        if payload.get("full_reload") or (not chunk_relative_paths and not deleted_relative_paths):
            set_job_progress(stage="graph_rebuild", progress=55, detail="Đang rebuild toàn bộ graph.")
            return reload_local_resources()
        set_job_progress(stage="graph_delta_sync", progress=55, detail="Đang đồng bộ delta graph.")
        return sync_local_resources(
            chunk_relative_paths=chunk_relative_paths,
            deleted_relative_paths=deleted_relative_paths,
        )


# Helper cho `raise_if_add_failed` trong module nay.
def _raise_if_add_failed(summary: dict[str, object]) -> None:
    if summary.get("changed"):
        return
    if summary.get("has_failures"):
        raise RuntimeError(str(summary.get("message") or "Không thể nạp dữ liệu PDF."))


# Chay `add data job` cho luong xu ly hien tai.
def run_add_data_job(*, links_text: str, pdf_files: list[dict] | None = None, display_name: str = "") -> dict:
    set_job_progress(stage="ingest_prepare", progress=15, detail="Đang chuẩn bị nạp tài liệu.")
    with ADMIN_LOCK:
        summary = add_corpus_documents(
            links_text=links_text,
            pdf_files=list(pdf_files or []),
            display_name=display_name,
        )
    _raise_if_add_failed(summary)
    set_job_progress(stage="ingest_done", progress=60, detail=str(summary.get("message") or "Đã nạp dữ liệu thô."))
    reload_job = start_cluster_reload_job(
        graph_sync_payload=_build_graph_sync_payload(summary),
        action_label="nạp",
    ) if summary.get("changed") else None
    return {
        **summary,
        "reload_job_id": str((reload_job or {}).get("job_id") or ""),
        "reload_status": "queued" if reload_job else "skipped",
    }


# Chay `delete data job` cho luong xu ly hien tai.
def run_delete_data_job(document_ids: list[str]) -> dict:
    set_job_progress(stage="delete_prepare", progress=15, detail="Đang chuẩn bị xóa tài liệu.")
    with ADMIN_LOCK:
        summary = delete_corpus_documents(document_ids)
    set_job_progress(stage="delete_done", progress=60, detail=str(summary.get("message") or "Đã xóa dữ liệu thô."))
    reload_job = start_cluster_reload_job(
        graph_sync_payload=_build_graph_sync_payload(summary),
        action_label="xóa",
    ) if summary.get("changed") else None
    return {
        **summary,
        "reload_job_id": str((reload_job or {}).get("job_id") or ""),
        "reload_status": "queued" if reload_job else "skipped",
    }


# Dung `local status` cho luong xu ly hien tai.
def build_local_status() -> dict[str, object]:
    config = load_config()
    graph_is_ready = _graph_has_data(config)
    fact_count = _graph_fact_count(config)
    return {
        "rag_id": CURRENT_RAG_ID,
        "port": resolve_server_port(),
        "backend_url": f"http://{DEFAULT_HOST}:{resolve_server_port()}",
        "health": "ok",
        "collection": config.neo4j_database,
        "point_count": None,
        "graph_ready": graph_is_ready,
        "fact_count": fact_count,
        "message": f"Neo4j {config.neo4j_uri}",
    }


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUGraphRagHTTP/6.0"

    # Dieu phoi cac route GET cua HTTP handler nay.
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(load_ui_html())
            return
        if parsed.path == "/admin":
            self._redirect(build_admin_ui_url())
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok", "rag_id": CURRENT_RAG_ID})
            return
        if parsed.path == "/api/auth/session":
            self._handle_auth_session_request()
            return
        if parsed.path == "/api/admin/documents":
            if not self._require_role("admin"):
                return
            self._send_json(list_corpus_documents())
            return
        if parsed.path == "/api/admin/jobs":
            if not self._require_role("admin"):
                return
            self._send_json({"jobs": list_jobs()})
            return
        if parsed.path.startswith("/api/admin/jobs/"):
            if not self._require_role("admin"):
                return
            self._handle_admin_job_request(parsed.path)
            return
        if parsed.path == "/api/admin/status":
            if not self._require_role("admin"):
                return
            self._handle_local_status_request()
            return
        if parsed.path == "/api/admin/system":
            if not self._require_role("admin"):
                return
            self._handle_system_status_request()
            return
        if parsed.path == "/api/admin/runtime-config":
            if not self._require_role("admin"):
                return
            self._send_json(build_current_runtime_config_payload())
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Dieu phoi cac route POST cua HTTP handler nay.
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/login":
            self._handle_login_request()
            return
        if parsed.path == "/api/auth/logout":
            self._handle_logout_request()
            return
        if parsed.path == "/api/chat":
            if not self._require_role("user"):
                return
            self._handle_chat_request()
            return
        if parsed.path == "/api/admin/add":
            if not self._require_role("admin"):
                return
            self._handle_add_data_request()
            return
        if parsed.path == "/api/admin/delete":
            if not self._require_role("admin"):
                return
            self._handle_delete_data_request()
            return
        if parsed.path == "/api/admin/reload":
            if not self._require_role("admin"):
                return
            self._handle_reload_request()
            return
        if parsed.path == "/api/admin/compare":
            if not self._require_role("admin"):
                return
            self._handle_compare_request()
            return
        if parsed.path == "/api/admin/runtime-config":
            if not self._require_role("admin"):
                return
            self._handle_runtime_config_update_request()
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    # Helper cho `handle_auth_session_request` trong module nay.
    def _handle_auth_session_request(self) -> None:
        session = self._get_session()
        self._send_json(
            {
                "authenticated": session is not None,
                "session": session,
                "accounts_hint": get_default_accounts_hint(),
            }
        )

    # Helper cho `handle_login_request` trong module nay.
    def _handle_login_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        account = authenticate_user(str(payload.get("username") or ""), str(payload.get("password") or ""))
        if account is None:
            self._send_json({"error": "Sai tên đăng nhập hoặc mật khẩu."}, status=HTTPStatus.UNAUTHORIZED)
            return
        self._send_json({"authenticated": True, "session": account}, headers={"Set-Cookie": build_session_cookie(account)})

    # Helper cho `handle_logout_request` trong module nay.
    def _handle_logout_request(self) -> None:
        self._send_json({"authenticated": False}, headers={"Set-Cookie": build_logout_cookie()})

    # Helper cho `handle_chat_request` trong module nay.
    def _handle_chat_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        question = str(payload.get("query") or payload.get("question") or "").strip()
        if not question:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return
        profile_name, runtime_overrides = parse_benchmark_profile_payload(payload)
        try:
            with GRAPH_LOCK:
                result = answer_question(question, runtime_overrides=runtime_overrides)
        except Exception as exc:
            self._send_json({"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        response_payload = _answer_to_payload(question, result)
        response_payload["benchmark_profile"] = profile_name
        self._send_json(response_payload)

    # Helper cho `handle_add_data_request` trong module nay.
    def _handle_add_data_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        links_text = str(payload.get("links_text") or "")
        display_name = str(payload.get("display_name") or "")
        pdf_files = list(payload.get("pdf_files") or [])
        if payload.get("async_mode"):
            job = create_job(
                action="add_data",
                runner=lambda: run_add_data_job(links_text=links_text, pdf_files=pdf_files, display_name=display_name),
                metadata={"rag_id": CURRENT_RAG_ID, "display_name": display_name},
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_add_data_job(links_text=links_text, pdf_files=pdf_files, display_name=display_name)
        except Exception as exc:
            self._send_json({"error": f"Không thể nạp dữ liệu. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(summary)

    # Helper cho `handle_delete_data_request` trong module nay.
    def _handle_delete_data_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        document_ids = load_document_ids_from_payload(payload)
        if not document_ids:
            self._send_json({"error": "Vui lòng chọn tài liệu cần xóa."}, status=HTTPStatus.BAD_REQUEST)
            return
        if payload.get("async_mode"):
            job = create_job(
                action="delete_data",
                runner=lambda: run_delete_data_job(document_ids),
                metadata={"rag_id": CURRENT_RAG_ID, "document_count": len(document_ids)},
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_delete_data_job(document_ids)
        except Exception as exc:
            self._send_json({"error": f"Không thể xóa dữ liệu. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(summary)

    # Helper cho `handle_reload_request` trong module nay.
    def _handle_reload_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        internal_request = is_internal_cluster_request(self.headers)
        if payload.get("async_mode"):
            if internal_request:
                job_payload = dict(payload)
                job_payload.pop("async_mode", None)
                job = create_job(
                    action="reload",
                    runner=lambda: _run_reload_job_safe(job_payload),
                    metadata={"rag_id": CURRENT_RAG_ID, "full_reload": bool(payload.get("full_reload"))},
                )
            else:
                graph_sync_payload = dict(payload)
                graph_sync_payload.pop("async_mode", None)
                job = start_cluster_reload_job(graph_sync_payload=graph_sync_payload or None, action_label="reload")
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = _run_reload_job_safe(payload) if internal_request else _run_cluster_reload_job(
                graph_sync_payload=(payload or None),
                action_label="reload",
            )
        except Exception as exc:
            self._send_json({"error": f"Không thể reload Graph RAG. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(summary)

    # Helper cho `handle_compare_request` trong module nay.
    def _handle_compare_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        question = str(payload.get("question") or payload.get("query") or "").strip()
        if not question:
            self._send_json({"error": "Vui lòng nhập câu hỏi để so sánh."}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            result = compare_cluster_answers(
                question=question,
                current_rag_id=CURRENT_RAG_ID,
                local_answer_builder=lambda value: _answer_to_payload(value, answer_question(value)),
            )
        except Exception as exc:
            self._send_json({"error": f"Không so sánh được 3 RAG. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(result)

    # Helper cho `handle_runtime_config_update_request` trong module nay.
    def _handle_runtime_config_update_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        scope = str(payload.get("scope") or "").strip()
        values = payload.get("values") or {}
        if not scope or not isinstance(values, dict):
            self._send_json({"error": "Payload cấu hình không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            update_runtime_scope(scope, values)
            warm_up_graph.cache_clear()
        except Exception as exc:
            self._send_json({"error": f"Không lưu được runtime config. Chi tiết: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(build_current_runtime_config_payload())

    # Helper cho `handle_admin_job_request` trong module nay.
    def _handle_admin_job_request(self, path: str) -> None:
        job_id = path.rsplit("/", 1)[-1].strip()
        if not job_id:
            self._send_json({"error": "Thiếu job_id."}, status=HTTPStatus.BAD_REQUEST)
            return
        job = get_job(job_id)
        if job is None:
            self._send_json({"error": "Không tìm thấy job."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_json(job)

    # Helper cho `handle_local_status_request` trong module nay.
    def _handle_local_status_request(self) -> None:
        try:
            self._send_json(build_local_status())
        except Exception as exc:
            self._send_json(
                {
                    "rag_id": CURRENT_RAG_ID,
                    "port": resolve_server_port(),
                    "backend_url": f"http://{DEFAULT_HOST}:{resolve_server_port()}",
                    "health": "error",
                    "error": str(exc),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    # Helper cho `handle_system_status_request` trong module nay.
    def _handle_system_status_request(self) -> None:
        try:
            payload = collect_cluster_status(current_rag_id=CURRENT_RAG_ID, local_status_builder=build_local_status)
        except Exception as exc:
            self._send_json({"error": f"Không tải được trạng thái hệ thống. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(payload)

    # Lay `session` cho luong xu ly hien tai.
    def _get_session(self) -> dict[str, str] | None:
        return read_session_from_cookie(self.headers.get("Cookie"))

    # Helper cho `require_role` trong module nay.
    def _require_role(self, role: str) -> bool:
        if is_internal_cluster_request(self.headers):
            return True
        session = self._get_session()
        if has_role(session, role):
            return True
        status = HTTPStatus.UNAUTHORIZED if session is None else HTTPStatus.FORBIDDEN
        self._send_json(
            {
                "error": "Bạn chưa đăng nhập." if session is None else "Bạn không có quyền truy cập tài nguyên này.",
                "authenticated": session is not None,
                "session": session,
            },
            status=status,
        )
        return False

    # Doc `json payload` cho luong xu ly hien tai.
    def _read_json_payload(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Payload JSON không hợp lệ."}, status=HTTPStatus.BAD_REQUEST)
            return None

    # Tat log mac dinh cua BaseHTTPRequestHandler de terminal gon hon.
    def log_message(self, format: str, *args) -> None:
        return

    # Helper cho `send_html` trong module nay.
    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
        self._send_response_body(html.encode("utf-8"), "text/html; charset=utf-8", status=status, headers=headers)

    # Helper cho `redirect` trong module nay.
    def _redirect(self, location: str, status: HTTPStatus = HTTPStatus.FOUND) -> None:
        self._send_response_body(b"", "text/plain; charset=utf-8", status=status, headers={"Location": location})

    # Helper cho `send_json` trong module nay.
    def _send_json(
        self,
        payload: dict,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._send_response_body(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
            headers=headers,
        )

    # Helper cho `send_response_body` trong module nay.
    def _send_response_body(
        self,
        body: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for header_name, header_value in (headers or {}).items():
                self.send_header(header_name, header_value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True


# Làm mới trạng thái GraphRAG mà không rebuild lại Neo4j khi dữ liệu không thay đổi.
def refresh_local_resources() -> dict[str, str]:
    config = load_config()
    graph_is_ready = _graph_has_data(config)
    fact_count = _graph_fact_count(config)
    warm_up_graph.cache_clear()
    return {
        "rag_id": CURRENT_RAG_ID,
        "fact_count": str(fact_count),
        "message": (
            f"Đã làm mới trạng thái GraphRAG. "
            f"Neo4j {'sẵn sàng' if graph_is_ready else 'chưa sẵn sàng'} với {fact_count} fact."
        ),
    }


# Ghi đè reload cũ: chỉ rebuild graph khi Neo4j đang trống; còn reload rỗng/full thì refresh nhanh.
def run_reload_job(sync_payload: dict | None = None) -> dict[str, str]:
    with GRAPH_LOCK:
        payload = sync_payload or {}
        chunk_relative_paths = list(payload.get("chunk_relative_paths") or [])
        deleted_relative_paths = list(payload.get("deleted_relative_paths") or [])
        if payload.get("full_reload") or (not chunk_relative_paths and not deleted_relative_paths):
            config = load_config()
            if _graph_has_data(config):
                set_job_progress(stage="graph_refresh", progress=55, detail="Đang làm mới trạng thái GraphRAG.")
                return refresh_local_resources()
            set_job_progress(stage="graph_rebuild", progress=55, detail="Đang rebuild toàn bộ graph.")
            return reload_local_resources()
        set_job_progress(stage="graph_delta_sync", progress=55, detail="Đang đồng bộ delta graph.")
        return sync_local_resources(
            chunk_relative_paths=chunk_relative_paths,
            deleted_relative_paths=deleted_relative_paths,
        )


# Chay `cli` cho luong xu ly hien tai.
def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Graph RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        question = input("\nNhập câu hỏi: ").strip()
        if question.lower() == "exit":
            break
        with GRAPH_LOCK:
            result = answer_question(question)
        print("\n=== TRẢ LỜI ===")
        print(result.answer)


# Resolve `server port` cho luong xu ly hien tai.
def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Chay `server` cho luong xu ly hien tai.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    config = load_config()
    Thread(target=warm_resources_in_background, daemon=True).start()
    print("Đang khởi tạo Graph RAG...")
    print(f"Graph RAG đang chạy tại http://{host}:{resolved_port}")
    print(f"Neo4j: {config.neo4j_uri} / database={config.neo4j_database}")
    print("Client UI: / | Admin UI: /admin | /health sẵn sàng ngay.")
    print("Nhấn Ctrl+C để dừng server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        server.server_close()


# Entry point chinh cua module nay.
def main() -> None:
    # Bước 1: nếu có cờ `--cli` thì chạy chế độ hỏi đáp trực tiếp trên terminal.
    if "--cli" in sys.argv:
        run_cli()
        return
    # Bước 2: nếu không chạy CLI thì khởi động HTTP server cho client/admin UI của GraphRAG.
    run_server()


if __name__ == "__main__":
    main()
