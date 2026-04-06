from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from config import build_qdrant_client, build_vector_store, configure_models, load_config
from prompts import build_prompt_templates


QUERY_FUSION_PROMPT = """Bạn là trợ lý tạo truy vấn tìm kiếm cho hệ thống GraphRAG tư vấn tuyển sinh.
Hãy tạo {num_queries} cách diễn đạt khác nhau cho cùng một câu hỏi, giữ nguyên ý định và ưu tiên từ khóa bám sát dữ liệu tuyển sinh NTU.
Mỗi dòng dùng một truy vấn, không đánh số, không giải thích.

Câu hỏi gốc: {query}
Danh sách truy vấn:
"""


@dataclass(frozen=True)
class SourceFact:
    content: str
    relative_path: str
    heading_path: str
    score: float | None = None


@dataclass(frozen=True)
class ChatAnswer:
    question: str
    answer: str
    facts: list[SourceFact]


def _resolve_query_fusion_mode(mode: str) -> FUSION_MODES:
    normalized = mode.strip().lower()
    mapping = {
        "reciprocal_rank": FUSION_MODES.RECIPROCAL_RANK,
        FUSION_MODES.RECIPROCAL_RANK.value: FUSION_MODES.RECIPROCAL_RANK,
        FUSION_MODES.RELATIVE_SCORE.value: FUSION_MODES.RELATIVE_SCORE,
        FUSION_MODES.DIST_BASED_SCORE.value: FUSION_MODES.DIST_BASED_SCORE,
        FUSION_MODES.SIMPLE.value: FUSION_MODES.SIMPLE,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unsupported QUERY_FUSION_MODE={mode!r}. Supported values: {supported} "
            "(legacy alias `reciprocal_rank` is also accepted)."
        ) from exc


# Score sau query fusion bị đổi thang đo, nên không thể so trực tiếp với
# ngưỡng similarity gốc như ở truy vấn đơn.
def _should_apply_similarity_threshold(config) -> bool:
    return not (config.query_fusion_enabled and config.query_fusion_num_queries > 1)


## Kiểm tra collection Qdrant đã tồn tại và có dữ liệu trước khi phục vụ truy vấn.
def _ensure_collection_ready(config) -> None:
    client = build_qdrant_client(config)
    if not client.collection_exists(config.qdrant_collection):
        raise RuntimeError(
            f"Qdrant collection `{config.qdrant_collection}` chưa tồn tại. "
            "Hãy chạy `python graph_rag/src/ingest.py --reset-graph` trước."
        )
    count = int(client.count(collection_name=config.qdrant_collection, exact=True).count)
    if count == 0:
        raise RuntimeError(
            f"Qdrant collection `{config.qdrant_collection}` đang rỗng. "
            "Hãy chạy ingest lại trước khi query."
        )


## Tạo query engine dùng lại nhiều lần để tránh rebuild index cho mỗi câu hỏi.
## Engine này query trực tiếp trên tập graph facts đã lưu trong Qdrant.
@lru_cache(maxsize=8)
def get_query_engine(top_k: int | None = None, overrides_key: str = "{}"):
    overrides = json.loads(overrides_key)
    config = load_config(overrides=overrides)
    configure_models(config)
    _ensure_collection_ready(config)
    vector_store = build_vector_store(config)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    resolved_top_k = top_k or config.retrieval_top_n
    qa_template, refine_template = build_prompt_templates(
        shared_prompt=config.shared_prompt,
        query_refusal_response=config.query_refusal_response,
    )
    retriever = index.as_retriever(similarity_top_k=resolved_top_k)
    if config.query_fusion_enabled and config.query_fusion_num_queries > 1:
        retriever = QueryFusionRetriever(
            retrievers=[retriever],
            query_gen_prompt=QUERY_FUSION_PROMPT,
            mode=_resolve_query_fusion_mode(config.query_fusion_mode),
            similarity_top_k=resolved_top_k,
            num_queries=config.query_fusion_num_queries,
            use_async=False,
        )
    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=qa_template,
        refine_template=refine_template,
    )


## Chuẩn hóa `source_nodes` của LlamaIndex về danh sách graph facts hiển thị cho UI/evaluation.
def _collect_facts(response, limit: int = 5) -> list[SourceFact]:
    facts: list[SourceFact] = []
    source_nodes = getattr(response, "source_nodes", None) or []
    for node in source_nodes[:limit]:
        metadata = node.node.metadata or {}
        facts.append(
            SourceFact(
                content=node.node.get_content(),
                relative_path=str(metadata.get("relative_path") or metadata.get("source_file") or ""),
                heading_path=str(metadata.get("heading_path") or ""),
                score=getattr(node, "score", None),
            )
        )
    return facts


## Quyết định có đủ tự tin để trả lời hay nên fallback sang câu từ chối mặc định.
def _is_confident_enough(facts: list[SourceFact], threshold: float) -> bool:
    if not facts:
        return False
    if threshold <= 0:
        return True
    scores = [float(fact.score) for fact in facts if fact.score is not None]
    if not scores:
        return True
    return max(scores) >= threshold


## Xử lý một lượt hỏi đáp hoàn chỉnh theo đúng đặc trưng kiến trúc GraphRAG:
def answer_question(question: str, top_k: int | None = None, runtime_overrides: dict | None = None) -> ChatAnswer:
    config = load_config(overrides=runtime_overrides)
    resolved_top_k = top_k or config.retrieval_top_n
    response = get_query_engine(top_k=top_k, overrides_key=json.dumps(runtime_overrides or {}, ensure_ascii=False, sort_keys=True)).query(question)
    facts = _collect_facts(response, limit=resolved_top_k)
    threshold = config.retrieval_similarity_threshold if _should_apply_similarity_threshold(config) else 0.0
    if not _is_confident_enough(facts, threshold):
        return ChatAnswer(
            question=question,
            answer=config.query_refusal_response,
            facts=facts,
        )
    return ChatAnswer(
        question=question,
        answer=str(response).strip(),
        facts=facts,
    )
