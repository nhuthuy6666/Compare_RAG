from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import parse_qs, urlparse

from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llamaindex_shared import (
    AdminUiConfig,
    ChatUiConfig,
    authenticate_user_with_status,
    build_admin_ui_url,
    build_chat_ui_tabs,
    build_cluster_server_urls,
    change_password_for_session,
    build_logout_cookie,
    build_query_engine,
    build_runtime_config_payload,
    build_session_cookie,
    collect_cluster_status,
    collect_sources,
    compare_cluster_answers,
    configure_models,
    create_job,
    delete_corpus_documents,
    delete_account_for_session,
    ensure_vector_index,
    get_job,
    has_role,
    has_sufficient_query_grounding,
    is_internal_cluster_request,
    list_corpus_documents,
    list_jobs,
    load_document_ids_from_payload,
    load_shared_config,
    post_json,
    read_session_from_cookie,
    register_email_user,
    render_admin_ui,
    render_chat_ui,
    render_email_verification_result,
    set_job_progress,
    should_apply_similarity_threshold,
    supports_self_service_account,
    update_runtime_scope,
    verify_email_token,
    wait_for_job,
    add_corpus_documents,
)
from llamaindex_shared.benchmark_runtime import parse_benchmark_profile_payload, runtime_overrides_signature  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".qdrant_state.json"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
CURRENT_RAG_ID = "hybrid"
COLLECTION_NAME = "ntu_hybrid_llamaindex"
RAG_LOCK = Lock()
ADMIN_LOCK = Lock()


# Nap `ui html` cho luong xu ly hien tai.
@lru_cache(maxsize=1)
def load_ui_html() -> str:
    return render_chat_ui(
        ChatUiConfig(
            current_rag_id=CURRENT_RAG_ID,
            page_title="NTU Hybrid RAG",
            brand_badge="NTU Admissions",
            brand_title="NTU Bot",
            brand_description=(
                "Giao diện người dùng tập trung vào hỏi đáp, lịch sử hội thoại và chuyển nhanh giữa Baseline, Hybrid, GraphRAG."
            ),
            header_badge="Hybrid RAG",
            header_subtitle="Hybrid retrieval | Qdrant | LlamaIndex",
            assistant_label="Hybrid NTU Bot",
            empty_title="Hybrid RAG sẵn sàng",
            empty_description="Dùng cùng một câu hỏi trên 3 tab để so sánh dense, hybrid và graph retrieval.",
            placeholder="Nhập câu hỏi tuyển sinh của bạn...",
            composer_hint="Ví dụ: Học phí ngành CNTT năm 2025 là bao nhiêu?",
            loading_message="Đang truy xuất tài liệu bằng hybrid retrieval...",
            ready_message="Đã hoàn tất.",
            storage_key="ntu_fusion_hybrid_sessions",
            admin_href=build_admin_ui_url(),
            suggestions=[
                "Điểm chuẩn năm 2025 là bao nhiêu?",
                "Hồ sơ đăng ký cần những gì?",
                "Ngành nào có nhiều học bổng nhất?",
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
    config = load_shared_config(rag_id=CURRENT_RAG_ID, collection_name=COLLECTION_NAME, overrides={})
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
            "prompt": config.prompt,
            "query_refusal_response": config.query_refusal_response,
            "graph_vector_candidates": int(os.getenv("GRAPH_VECTOR_CANDIDATES", "18")),
            "graph_neighbor_hops": int(os.getenv("GRAPH_NEIGHBOR_HOPS", "1")),
            "graph_neighbor_facts_limit": int(os.getenv("GRAPH_NEIGHBOR_FACTS_LIMIT", "12")),
        }
    )


# Lay `resources` cho luong xu ly hien tai.
@lru_cache(maxsize=16)
def get_resources(profile_name: str = "default", overrides_key: str = "{}"):
    config = load_shared_config(
        rag_id=CURRENT_RAG_ID,
        collection_name=COLLECTION_NAME,
        overrides=json.loads(overrides_key),
    )
    configure_models(config)
    index = ensure_vector_index(config, state_path=STATE_PATH, enable_hybrid=True)
    query_engine = build_query_engine(index, config, enable_hybrid=True)
    return config, query_engine


# Xu ly `query` cho luong xu ly hien tai.
def answer_query(query: str, *, profile_name: str = "default", runtime_overrides: dict | None = None) -> dict:
    config, query_engine = get_resources(profile_name, runtime_overrides_signature(runtime_overrides))
    response = query_engine.query(query)
    sources = collect_sources(response, limit=config.retrieval_top_n)
    is_grounded = has_sufficient_query_grounding(
        query,
        sources,
        similarity_threshold=config.retrieval_similarity_threshold,
        enforce_similarity_threshold=should_apply_similarity_threshold(config),
    )
    answer = str(response).strip() if is_grounded else config.query_refusal_response
    return {
        "answer": answer,
        "rewritten_query": query,
        "sources": sources,
        "benchmark_profile": profile_name,
    }


# Reload `local resources` cho luong xu ly hien tai.
def reload_local_resources() -> dict[str, str]:
    get_resources.cache_clear()
    config, _ = get_resources()
    return {
        "rag_id": CURRENT_RAG_ID,
        "collection": config.qdrant_collection,
        "message": f"Đã reload collection {config.qdrant_collection}.",
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


# Reload `cluster resources` cho luong xu ly hien tai.
def reload_cluster_resources(graph_sync_payload: dict | None = None) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    remote_targets = [rag_id for rag_id in build_cluster_server_urls() if rag_id != CURRENT_RAG_ID]
    set_job_progress(stage="local_reload", progress=20, detail="Đang reload Hybrid cục bộ.")
    local_summary = run_reload_job()
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
            request_payload = {"async_mode": True}
            if rag_id == "graph" and graph_sync_payload:
                request_payload.update(graph_sync_payload)
            job = post_json(f"{base_url}/api/admin/reload", request_payload)
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
        if status != "ok" and not (rag_id == "graph" and "Graph Neo4j" in message and "reset-graph" in message):
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
        config, _ = get_resources()
        print(f"[hybrid] Warm-up hoàn tất cho collection {config.qdrant_collection}.")
    except Exception as exc:
        print(f"[hybrid] Warm-up lỗi: {exc}")


# Chay `reload job` cho luong xu ly hien tai.
def run_reload_job() -> dict[str, str]:
    with RAG_LOCK:
        set_job_progress(stage="reindex", progress=55, detail="Đang làm mới cache và tài nguyên Hybrid.")
        return reload_local_resources()


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
    config = load_shared_config(rag_id=CURRENT_RAG_ID, collection_name=COLLECTION_NAME, overrides={})
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    try:
        collection_exists = client.collection_exists(config.qdrant_collection)
        point_count = int(client.count(collection_name=config.qdrant_collection, exact=True).count) if collection_exists else 0
    finally:
        client.close()
    return {
        "rag_id": CURRENT_RAG_ID,
        "port": resolve_server_port(),
        "backend_url": f"http://{DEFAULT_HOST}:{resolve_server_port()}",
        "health": "ok",
        "collection": config.qdrant_collection,
        "point_count": point_count,
        "graph_ready": None,
        "fact_count": None,
        "message": "Hybrid vector store sẵn sàng." if collection_exists else "Collection chưa tồn tại.",
    }


class ChatHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "NTUHybridHTTP/4.0"

    # Dieu phoi cac route GET cua HTTP handler nay.
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(load_ui_html())
            return
        if parsed.path == "/verify-email":
            self._handle_verify_email_request(parsed.query)
            return
        if parsed.path == "/admin":
            self._send_html(load_admin_html())
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
        if parsed.path == "/api/auth/register":
            self._handle_register_request()
            return
        if parsed.path == "/api/auth/change-password":
            self._handle_change_password_request()
            return
        if parsed.path == "/api/auth/delete-account":
            self._handle_delete_account_request()
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
                "self_service_enabled": supports_self_service_account(session),
            }
        )

    # Helper cho `handle_login_request` trong module nay.
    def _handle_login_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        result = authenticate_user_with_status(str(payload.get("username") or ""), str(payload.get("password") or ""))
        account = result.get("account")
        if not isinstance(account, dict):
            self._send_json(
                {"error": str(result.get("message") or "Sai tên đăng nhập hoặc mật khẩu.")},
                status=HTTPStatus.UNAUTHORIZED,
            )
            return
        self._send_json({"authenticated": True, "session": account}, headers={"Set-Cookie": build_session_cookie(account)})

    # Helper cho `handle_register_request` trong module nay.
    def _handle_register_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        try:
            result = register_email_user(
                email=str(payload.get("email") or ""),
                password=str(payload.get("password") or ""),
                display_name=str(payload.get("display_name") or ""),
                verification_base_url=self._build_absolute_url("/verify-email"),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(result, status=HTTPStatus.CREATED)

    # Helper cho `handle_logout_request` trong module nay.
    def _handle_logout_request(self) -> None:
        self._send_json({"authenticated": False}, headers={"Set-Cookie": build_logout_cookie()})

    # Helper cho `handle_change_password_request` trong module nay.
    def _handle_change_password_request(self) -> None:
        if not self._require_role("user"):
            return
        payload = self._read_json_payload()
        if payload is None:
            return
        try:
            result = change_password_for_session(
                self._get_session(),
                current_password=str(payload.get("current_password") or ""),
                new_password=str(payload.get("new_password") or ""),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result)

    # Helper cho `handle_delete_account_request` trong module nay.
    def _handle_delete_account_request(self) -> None:
        if not self._require_role("user"):
            return
        payload = self._read_json_payload()
        if payload is None:
            return
        try:
            result = delete_account_for_session(
                self._get_session(),
                password=str(payload.get("password") or ""),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result, headers={"Set-Cookie": build_logout_cookie()})

    # Helper cho `handle_verify_email_request` trong module nay.
    def _handle_verify_email_request(self, raw_query: str) -> None:
        query = parse_qs(raw_query or "")
        token = str((query.get("token") or [""])[0] or "").strip()
        try:
            result = verify_email_token(token)
            self._send_html(
                render_email_verification_result(
                    success=True,
                    message=str(result.get("message") or "Xác thực email thành công."),
                    login_href="/",
                )
            )
        except Exception as exc:
            self._send_html(
                render_email_verification_result(success=False, message=str(exc), login_href="/"),
                status=HTTPStatus.BAD_REQUEST,
            )

    # Helper cho `handle_chat_request` trong module nay.
    def _handle_chat_request(self) -> None:
        payload = self._read_json_payload()
        if payload is None:
            return
        query = str(payload.get("query") or payload.get("question") or "").strip()
        if not query:
            self._send_json({"error": "Vui lòng nhập câu hỏi."}, status=HTTPStatus.BAD_REQUEST)
            return
        profile_name, runtime_overrides = parse_benchmark_profile_payload(payload)
        try:
            with RAG_LOCK:
                response = answer_query(query, profile_name=profile_name, runtime_overrides=runtime_overrides)
        except Exception as exc:
            self._send_json({"error": f"Không thể xử lý câu hỏi. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(response)

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
            job = (
                create_job(
                    action="reload",
                    runner=run_reload_job,
                    metadata={"rag_id": CURRENT_RAG_ID, "full_reload": bool(payload.get("full_reload"))},
                )
                if internal_request
                else start_cluster_reload_job(graph_sync_payload=None, action_label="reload")
            )
            self._send_json(job, status=HTTPStatus.ACCEPTED)
            return
        try:
            summary = run_reload_job() if internal_request else _run_cluster_reload_job(graph_sync_payload=None, action_label="reload")
        except Exception as exc:
            self._send_json({"error": f"Không thể reload Hybrid RAG. Chi tiết: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
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
                local_answer_builder=lambda value: answer_query(value),
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
            get_resources.cache_clear()
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

    # Helper cho `build_absolute_url` trong module nay.
    def _build_absolute_url(self, path: str) -> str:
        forwarded_proto = str(self.headers.get("X-Forwarded-Proto") or "").strip()
        scheme = forwarded_proto or "http"
        host = str(self.headers.get("Host") or f"{DEFAULT_HOST}:{resolve_server_port()}").strip()
        normalized_path = path if str(path or "").startswith("/") else f"/{path}"
        return f"{scheme}://{host}{normalized_path}"

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


# Chay `cli` cho luong xu ly hien tai.
def run_cli() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    print("Hybrid RAG sẵn sàng. Gõ 'exit' để thoát.")
    while True:
        query = input("\nNhập câu hỏi: ").strip()
        if query.lower() == "exit":
            break
        with RAG_LOCK:
            result = answer_query(query)
        print("\n=== TRẢ LỜI ===")
        print(result["answer"])


# Resolve `server port` cho luong xu ly hien tai.
def resolve_server_port() -> int:
    return int(os.getenv("UI_PORT", str(DEFAULT_PORT)))


# Chay `server` cho luong xu ly hien tai.
def run_server(host: str = DEFAULT_HOST, port: int | None = None) -> None:
    resolved_port = port if port is not None else resolve_server_port()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    server = ThreadingHTTPServer((host, resolved_port), ChatHTTPRequestHandler)
    Thread(target=warm_resources_in_background, daemon=True).start()
    print("Đang khởi tạo Hybrid RAG...")
    print(f"Hybrid RAG đang chạy tại http://{host}:{resolved_port}")
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
    # Bước 1: nếu có cờ `--cli` thì chạy chế độ hỏi đáp trực tiếp trong terminal.
    if "--cli" in sys.argv:
        run_cli()
        return
    # Bước 2: nếu không chạy CLI thì khởi động HTTP server cho client/admin UI.
    run_server()


if __name__ == "__main__":
    main()
