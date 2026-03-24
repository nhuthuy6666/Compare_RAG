from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from llama_index.core import VectorStoreIndex

from config import build_qdrant_client, build_vector_store, configure_models, load_config
from prompts import build_prompt_templates


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
def get_query_engine(top_k: int | None = None):
    config = load_config()
    configure_models(config)
    _ensure_collection_ready(config)
    vector_store = build_vector_store(config)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    resolved_top_k = top_k or config.retrieval_top_n
    qa_template, refine_template = build_prompt_templates(
        shared_prompt=config.shared_prompt,
        query_refusal_response=config.query_refusal_response,
    )
    return index.as_query_engine(
        similarity_top_k=resolved_top_k,
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
## query trên graph facts bằng nguyên câu hỏi gốc, không thêm heuristic tăng cường truy vấn.
def answer_question(question: str, top_k: int | None = None) -> ChatAnswer:
    config = load_config()
    response = get_query_engine(top_k=top_k).query(question)
    facts = _collect_facts(response)
    if not _is_confident_enough(facts, config.retrieval_similarity_threshold):
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
