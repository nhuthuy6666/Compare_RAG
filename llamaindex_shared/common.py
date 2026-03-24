from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from llamaindex_shared.corpus_utils import load_chunk_record_groups, records_to_nodes
from llamaindex_shared.openai_compatible import OpenAICompatibleEmbedding, OpenAICompatibleLLM
from llamaindex_shared.prompts import build_prompt_templates

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_BASELINE_CONFIG = PROJECT_ROOT / "extract_md" / "rag_baseline.json"


@dataclass(frozen=True)
class SharedRagConfig:
    baseline_config_path: Path
    source_chunk_root: Path
    corpus_scope: str
    llm_base_url: str
    llm_model: str
    embed_model: str
    llm_api_key: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    retrieval_top_n: int
    retrieval_similarity_threshold: float
    generation_temperature: float
    prompt: str
    query_refusal_response: str


# Đọc JSON baseline config nếu file tồn tại.
def _load_baseline_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# Hợp nhất config từ baseline JSON và biến môi trường thành một object dùng chung.
def load_shared_config(
    *,
    collection_name: str,
    baseline_config_path: Path | None = None,
) -> SharedRagConfig:
    baseline_path = baseline_config_path or DEFAULT_BASELINE_CONFIG
    baseline = _load_baseline_config(baseline_path)
    corpus = baseline.get("corpus") or {}
    models = baseline.get("models") or {}
    retrieval = baseline.get("retrieval") or {}
    generation = baseline.get("generation") or {}

    return SharedRagConfig(
        baseline_config_path=baseline_path,
        source_chunk_root=Path(
            os.getenv("CHUNK_JSONL_ROOT", str(corpus.get("chunk_root") or PROJECT_ROOT / "extract_md" / "data_chunks"))
        ),
        corpus_scope=os.getenv("CORPUS_SCOPE", str(corpus.get("scope") or "all")),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", str(models.get("chat") or "llama3.1:8b")),
        embed_model=os.getenv("EMBED_MODEL", str(models.get("embedding") or "bge-m3:latest")),
        llm_api_key=os.getenv("LLM_API_KEY", "ollama"),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection=collection_name,
        retrieval_top_n=int(os.getenv("RETRIEVAL_TOP_N", str(retrieval.get("top_n") or 6))),
        retrieval_similarity_threshold=float(
            os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", str(retrieval.get("similarity_threshold") or 0.0))
        ),
        generation_temperature=float(
            os.getenv("GENERATION_TEMPERATURE", str(generation.get("temperature") or 0.1))
        ),
        prompt=str(baseline.get("prompt") or "").strip(),
        query_refusal_response=str(
            baseline.get("query_refusal_response")
            or "Tôi chưa đủ căn cứ để trả lời câu hỏi này từ dữ liệu hiện có."
        ).strip(),
    )


# Cấu hình LlamaIndex để dùng chung LLM chat và embedding model theo config hiện tại.
def configure_models(config: SharedRagConfig) -> None:
    Settings.llm = OpenAICompatibleLLM(
        model_name=config.llm_model,
        api_key=config.llm_api_key,
        api_base=config.llm_base_url,
        timeout=180.0,
        max_tokens=1024,
        temperature=config.generation_temperature,
    )
    Settings.embed_model = OpenAICompatibleEmbedding(
        model_name=config.embed_model,
        api_key=config.llm_api_key,
        api_base=config.llm_base_url,
        timeout=300.0,
        embed_batch_size=4,
        retry_attempts=8,
        retry_delay=8.0,
    )


# Tạo cặp sync/async Qdrant client để vector store có thể dùng cho cả index và query.
def _build_qdrant_clients(config: SharedRagConfig) -> tuple[QdrantClient, AsyncQdrantClient]:
    return (
        QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key),
        AsyncQdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key),
    )


# Tạo Qdrant vector store cho dense-only hoặc hybrid retrieval.
def build_vector_store(config: SharedRagConfig, *, enable_hybrid: bool) -> QdrantVectorStore:
    client, aclient = _build_qdrant_clients(config)
    kwargs: dict[str, Any] = {
        "collection_name": config.qdrant_collection,
        "client": client,
        "aclient": aclient,
        "enable_hybrid": enable_hybrid,
    }
    if enable_hybrid:
        kwargs["fastembed_sparse_model"] = "Qdrant/bm25"
    return QdrantVectorStore(**kwargs)


# Tạo collection nếu chưa tồn tại, đồng thời bật sparse vectors khi hybrid được sử dụng.
def _ensure_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    embedding_dim: int,
    enable_hybrid: bool,
) -> None:
    if client.collection_exists(collection_name):
        return

    vectors_config: dict[str, VectorParams] | VectorParams
    vectors_config = {
        "text-dense": VectorParams(size=embedding_dim, distance=Distance.COSINE),
    }
    sparse_vectors_config = {"text-sparse-new": SparseVectorParams()} if enable_hybrid else None
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )


# Kiểm tra collection có dữ liệu hay không để quyết định có cần rebuild index.
def _collection_has_points(client: QdrantClient, collection_name: str) -> bool:
    try:
        if not client.collection_exists(collection_name):
            return False
        return int(client.count(collection_name=collection_name, exact=True).count) > 0
    except Exception:
        return False


# Xóa collection cũ khi fingerprint thay đổi và cần ingest lại từ đầu.
def _reset_collection(client: QdrantClient, collection_name: str) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)


# Đọc toàn bộ chunk records, đổi sang TextNode và gán node id ổn định cho Qdrant.
def load_nodes(config: SharedRagConfig) -> list[TextNode]:
    groups = load_chunk_record_groups(
        chunk_root=config.source_chunk_root,
        scope=config.corpus_scope,
        limit=None,
    )
    records = [record for _, group in groups for record in group]
    if not records:
        raise FileNotFoundError(
            f"Không tìm thấy chunk JSONL trong {config.source_chunk_root} (scope={config.corpus_scope})."
        )
    nodes = records_to_nodes(records)
    for node in nodes:
        original_id = node.node_id
        node.metadata["doc_id"] = original_id
        node.id_ = str(uuid.uuid5(uuid.NAMESPACE_URL, original_id))
    return nodes


# Tạo fingerprint theo model + hybrid flag + nội dung node để theo dõi lúc nào cần reindex.
def _compute_nodes_fingerprint(nodes: list[TextNode], config: SharedRagConfig, enable_hybrid: bool) -> str:
    digest = hashlib.sha256()
    digest.update(config.embed_model.encode("utf-8"))
    digest.update(str(enable_hybrid).encode("utf-8"))
    for node in nodes:
        digest.update(node.node_id.encode("utf-8"))
        digest.update(node.text.encode("utf-8"))
        digest.update(json.dumps(node.metadata, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


# Đọc file state cache của lần ingest gần nhất.
def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# Lưu thông tin state để các lần chạy sau biết có cần reindex hay không.
def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Bảo đảm vector index trong Qdrant luôn khớp với corpus/model hiện tại.
# 1) Đọc nodes và tính fingerprint.
# 2) Nếu state + collection vẫn hợp lệ thì mở lại index từ vector store.
# 3) Nếu không thì reset collection, tạo collection mới và ingest lại toàn bộ nodes.
# 4) Ghi file state mới để tái sử dụng cho lần chạy sau.
def ensure_vector_index(
    config: SharedRagConfig,
    *,
    state_path: Path,
    enable_hybrid: bool,
) -> VectorStoreIndex:
    nodes = load_nodes(config)
    fingerprint = _compute_nodes_fingerprint(nodes, config, enable_hybrid)
    state = _load_state(state_path)
    client, _ = _build_qdrant_clients(config)
    vector_store = build_vector_store(config, enable_hybrid=enable_hybrid)
    sample_embedding = Settings.embed_model.get_text_embedding(nodes[0].text)
    embedding_dim = len(sample_embedding)

    if (
        state.get("fingerprint") == fingerprint
        and state.get("collection") == config.qdrant_collection
        and _collection_has_points(client, config.qdrant_collection)
    ):
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    _reset_collection(client, config.qdrant_collection)
    _ensure_collection(
        client,
        collection_name=config.qdrant_collection,
        embedding_dim=embedding_dim,
        enable_hybrid=enable_hybrid,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=False)
    _save_state(
        state_path,
        {
            "collection": config.qdrant_collection,
            "fingerprint": fingerprint,
            "node_count": len(nodes),
            "embed_model": config.embed_model,
            "hybrid_enabled": enable_hybrid,
        },
    )
    return index


# Tạo query engine dùng chung prompt, top-k và bật thêm hybrid mode khi cần.
def build_query_engine(
    index: VectorStoreIndex,
    config: SharedRagConfig,
    *,
    enable_hybrid: bool,
) -> BaseQueryEngine:
    qa_template, refine_template = build_prompt_templates(
        shared_prompt=config.prompt,
        query_refusal_response=config.query_refusal_response,
    )
    query_kwargs: dict[str, Any] = {
        "similarity_top_k": config.retrieval_top_n,
        "text_qa_template": qa_template,
        "refine_template": refine_template,
    }
    if enable_hybrid:
        query_kwargs.update(
            {
                "vector_store_query_mode": VectorStoreQueryMode.HYBRID,
                "sparse_top_k": config.retrieval_top_n,
                "hybrid_top_k": config.retrieval_top_n,
            }
        )
    return index.as_query_engine(**query_kwargs)


# Chuyển `source_nodes` của LlamaIndex thành schema JSON gọn cho UI và benchmark.
def collect_sources(response, *, limit: int) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for node_with_score in (getattr(response, "source_nodes", None) or [])[:limit]:
        metadata = node_with_score.node.metadata or {}
        relative_path = str(metadata.get("relative_path") or metadata.get("source_file") or "unknown")
        sources.append(
            {
                "source": Path(relative_path).name,
                "relative_path": relative_path,
                "chunk_id": metadata.get("chunk_id", "?"),
                "content": node_with_score.node.get_content(),
                "score": getattr(node_with_score, "score", None),
            }
        )
    return sources
