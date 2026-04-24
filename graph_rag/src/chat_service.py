from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from llama_index.core import Settings

from config import build_neo4j_driver, configure_models, load_config
from llamaindex_shared.prompts import build_prompt_templates
from neo4j_store import RetrievedFact, entity_guided_search, graph_ready, neighbor_search, vector_search


QUERY_FUSION_PROMPT = """Bạn là trợ lý tạo truy vấn tìm kiếm cho hệ thống GraphRAG tư vấn tuyển sinh.
Hãy tạo {num_queries} cách diễn đạt khác nhau cho cùng một câu hỏi, giữ nguyên ý định và ưu tiên từ khóa bám sát dữ liệu tuyển sinh NTU.
Mỗi dòng dùng một truy vấn, không đánh số, không giải thích.

Câu hỏi gốc: {query}
Danh sách truy vấn:
"""


@dataclass(frozen=True)
class SourceFact:
    """Fact nguồn đã rút gọn để trả về UI và benchmark."""

    content: str
    relative_path: str
    heading_path: str
    score: float | None


@dataclass(frozen=True)
class ChatAnswer:
    """Kết quả cuối cùng của một lượt hỏi GraphRAG."""

    question: str
    answer: str
    facts: list[SourceFact]


@dataclass(frozen=True)
class SearchPlan:
    """Biểu diễn một truy vấn con khi bật query fusion."""

    query: str
    rank: int


# Chuẩn hóa tên mode query fusion và chặn mode không được hỗ trợ.
def _resolve_query_fusion_mode(mode: str) -> str:
    """Kiểm tra mode query fusion có nằm trong danh sách hỗ trợ hay không."""

    normalized = mode.strip().lower()
    supported = {
        "reciprocal_rank",
        "relative_score",
        "dist_based_score",
        "simple",
    }
    if normalized not in supported:
        supported_text = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported QUERY_FUSION_MODE={mode!r}. Supported values: {supported_text}."
        )
    return normalized


# Khi fusion bật, score sau hợp nhất không còn cùng thang nên không áp ngưỡng như truy vấn đơn.
def _should_apply_similarity_threshold(config) -> bool:
    """Quyết định có nên dùng similarity threshold hay không."""

    return not (config.query_fusion_enabled and config.query_fusion_num_queries > 1)


# Chuẩn hóa text về lowercase một dòng để tăng độ ổn định cho xử lý truy vấn.
def _normalize_text(value: str) -> str:
    """Rút gọn khoảng trắng và đưa text về lowercase."""

    return re.sub(r"\s+", " ", value).strip().lower()


# Loại bớt ký tự đặc biệt để tạo truy vấn fulltext an toàn hơn cho Neo4j.
def _sanitize_fulltext_term(value: str) -> str:
    """Làm sạch chuỗi trước khi đưa vào fulltext query của Neo4j."""

    sanitized = re.sub(r"[^\wÀ-ỹ]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", sanitized).strip()


# Tạo câu truy vấn fulltext cho entity search từ câu hỏi người dùng.
def _build_entity_fulltext_query(query: str) -> str:
    """Tạo truy vấn fulltext theo cụm từ và token quan trọng trong câu hỏi."""

    tokens = [token for token in _sanitize_fulltext_term(query).split() if len(token) >= 3]
    if not tokens:
        return ""
    unique_tokens = list(dict.fromkeys(tokens))
    quoted_tokens = [f'"{token}"' for token in unique_tokens[:8]]
    phrase = " ".join(unique_tokens[:5]).strip()
    if phrase:
        quoted_tokens.insert(0, f'"{phrase}"')
    return " OR ".join(quoted_tokens)


# Tạo danh sách truy vấn con khi bật query fusion; nếu tắt thì chỉ giữ câu hỏi gốc.
def _generate_query_variants(question: str, config) -> list[SearchPlan]:
    """Sinh các truy vấn con phục vụ query fusion từ câu hỏi gốc."""

    if not config.query_fusion_enabled or config.query_fusion_num_queries <= 1:
        return [SearchPlan(query=question, rank=1)]

    prompt = QUERY_FUSION_PROMPT.format(num_queries=config.query_fusion_num_queries, query=question)
    try:
        response = Settings.llm.complete(prompt)
        lines = [line.strip(" -*\t") for line in response.text.splitlines() if line.strip()]
    except Exception:
        lines = []

    ordered_queries: list[str] = [question]
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned not in ordered_queries:
            ordered_queries.append(cleaned)
        if len(ordered_queries) >= config.query_fusion_num_queries:
            break
    return [SearchPlan(query=query, rank=index + 1) for index, query in enumerate(ordered_queries)]


# Hợp nhất kết quả của nhiều truy vấn con thành một ranking cuối cùng.
def _combine_variant_results(
    variant_results: list[tuple[SearchPlan, list[RetrievedFact]]],
    *,
    mode: str,
    limit: int,
) -> list[RetrievedFact]:
    """Hợp nhất kết quả của query fusion theo mode đã cấu hình."""

    combined: dict[str, dict[str, Any]] = {}
    for plan, results in variant_results:
        ranked_results = results[:limit]
        if mode == "simple":
            for item in ranked_results:
                state = combined.setdefault(item.fact_id, {"fact": item, "score": 0.0, "hits": 0})
                state["score"] += item.score
                state["hits"] += 1
            continue

        for index, item in enumerate(ranked_results, start=1):
            state = combined.setdefault(item.fact_id, {"fact": item, "score": 0.0, "hits": 0})
            if mode == "reciprocal_rank":
                state["score"] += 1.0 / (60 + index)
            elif mode == "relative_score":
                state["score"] += item.score / max(index, 1)
            elif mode == "dist_based_score":
                state["score"] += (item.score * item.score) / max(index, 1)
            state["hits"] += 1

    sorted_rows = sorted(
        combined.values(),
        key=lambda row: (row["score"], row["hits"], row["fact"].score),
        reverse=True,
    )[:limit]

    if not sorted_rows:
        return []

    max_score = max(float(row["score"]) for row in sorted_rows) or 1.0
    merged: list[RetrievedFact] = []
    for row in sorted_rows:
        fact = row["fact"]
        merged.append(
            RetrievedFact(
                fact_id=fact.fact_id,
                text=fact.text,
                relative_path=fact.relative_path,
                heading_path=fact.heading_path,
                chunk_id=fact.chunk_id,
                relation_label=fact.relation_label,
                source_entity=fact.source_entity,
                target_entity=fact.target_entity,
                source_type=fact.source_type,
                target_type=fact.target_type,
                source_fact_id=fact.source_fact_id,
                score=float(row["score"]) / max_score,
                retrieval_stage="fusion",
            )
        )
    return merged


# Khử trùng fact trùng ID giữa nhiều nguồn retrieve và giữ lại bản có score cao nhất.
def _dedupe_results(results: list[RetrievedFact], *, limit: int) -> list[RetrievedFact]:
    """Gộp fact trùng nhau và giữ lại bản có score cao nhất."""

    merged: dict[str, RetrievedFact] = {}
    for item in results:
        current = merged.get(item.fact_id)
        if current is None or item.score > current.score:
            merged[item.fact_id] = item
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


# Chạy một truy vấn duy nhất trên Neo4j bằng vector search, entity search và graph expansion.
def _run_single_query(plan: SearchPlan, config) -> list[RetrievedFact]:
    """Thực hiện một lượt retrieve đơn trên Neo4j với ba nhánh tìm kiếm."""

    driver = build_neo4j_driver(config)
    try:
        query_embedding = Settings.embed_model.get_query_embedding(plan.query)
        vector_hits = vector_search(
            driver,
            config,
            embedding=query_embedding,
            limit=max(config.graph_vector_candidates, config.retrieval_top_n * 2),
        )
        entity_hits = entity_guided_search(
            driver,
            config,
            query_text=_build_entity_fulltext_query(plan.query),
            limit=max(config.retrieval_top_n, 6),
        )
        seed_ids = [item.fact_id for item in vector_hits[: max(config.retrieval_top_n // 2, 2)]]
        neighbor_hits = neighbor_search(
            driver,
            config,
            seed_fact_ids=seed_ids,
            limit=config.graph_neighbor_facts_limit,
        )
    finally:
        driver.close()

    combined: dict[str, dict[str, Any]] = {}
    for item in vector_hits:
        state = combined.setdefault(item.fact_id, {"fact": item, "vector": 0.0, "entity": 0.0, "neighbor": 0.0})
        state["vector"] = max(state["vector"], item.score)
    for item in entity_hits:
        state = combined.setdefault(item.fact_id, {"fact": item, "vector": 0.0, "entity": 0.0, "neighbor": 0.0})
        state["entity"] = max(state["entity"], _normalize_rank_score(item.score, entity_hits))
        if item.score > state["fact"].score:
            state["fact"] = item
    for item in neighbor_hits:
        state = combined.setdefault(item.fact_id, {"fact": item, "vector": 0.0, "entity": 0.0, "neighbor": 0.0})
        state["neighbor"] = max(state["neighbor"], _normalize_rank_score(item.score, neighbor_hits))
        if item.score > state["fact"].score:
            state["fact"] = item

    ranked: list[RetrievedFact] = []
    for row in combined.values():
        final_score = min(
            1.0,
            (0.70 * row["vector"]) + (0.20 * row["entity"]) + (0.10 * row["neighbor"]),
        )
        fact = row["fact"]
        ranked.append(
            RetrievedFact(
                fact_id=fact.fact_id,
                text=fact.text,
                relative_path=fact.relative_path,
                heading_path=fact.heading_path,
                chunk_id=fact.chunk_id,
                relation_label=fact.relation_label,
                source_entity=fact.source_entity,
                target_entity=fact.target_entity,
                source_type=fact.source_type,
                target_type=fact.target_type,
                source_fact_id=fact.source_fact_id,
                score=final_score,
                retrieval_stage=fact.retrieval_stage,
            )
        )
    return _dedupe_results(ranked, limit=max(config.retrieval_top_n * 2, 8))


# Chuẩn hóa score theo ranking hiện tại để trộn nhiều nhánh retrieve vào cùng một thang tương đối.
def _normalize_rank_score(score: float, ranked_items: list[RetrievedFact]) -> float:
    """Chuẩn hóa score tương đối trong một danh sách retrieve."""

    if not ranked_items:
        return 0.0
    max_score = max(float(item.score) for item in ranked_items) or 1.0
    return min(1.0, float(score) / max_score)


# Chạy retrieval trên Neo4j bằng vector index và mở rộng neighborhood theo entity.
def retrieve_facts(question: str, config) -> list[RetrievedFact]:
    """Retrieve các fact tốt nhất cho một câu hỏi theo cấu hình hiện tại."""

    plans = _generate_query_variants(question, config)
    variant_results = [(plan, _run_single_query(plan, config)) for plan in plans]
    if len(variant_results) == 1:
        return variant_results[0][1][: config.retrieval_top_n]
    fusion_mode = _resolve_query_fusion_mode(config.query_fusion_mode)
    return _combine_variant_results(
        variant_results,
        mode=fusion_mode,
        limit=config.retrieval_top_n,
    )


# Định dạng danh sách fact thành context string để đưa vào prompt trả lời.
def _format_context(facts: list[RetrievedFact]) -> str:
    """Biến danh sách fact retrieve thành chuỗi context cho LLM."""

    lines = []
    for index, fact in enumerate(facts, start=1):
        parts = [f"[Fact {index}] {fact.text}"]
        if fact.source_entity or fact.target_entity:
            parts.append(
                f"Thực thể: {fact.source_entity or 'n/a'} -> {fact.target_entity or 'n/a'}"
            )
        if fact.heading_path:
            parts.append(f"Ngữ cảnh: {fact.heading_path}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


# Gọi LLM để tổng hợp câu trả lời cuối cùng từ context GraphRAG.
def _generate_answer(question: str, facts: list[RetrievedFact], config) -> str:
    """Sinh câu trả lời cuối cùng từ context fact đã retrieve."""

    qa_template, _ = build_prompt_templates(
        shared_prompt=config.shared_prompt,
        query_refusal_response=config.query_refusal_response,
    )
    prompt = qa_template.format(
        context_str=_format_context(facts),
        query_str=question,
    )
    response = Settings.llm.complete(prompt)
    return str(response.text or "").strip()


# Đảm bảo graph đã có dữ liệu trước khi nhận request chat.
def _ensure_graph_ready(config) -> None:
    """Báo lỗi sớm nếu Neo4j chưa có fact nào để phục vụ hỏi đáp."""

    driver = build_neo4j_driver(config)
    try:
        if not graph_ready(driver, config):
            raise RuntimeError(
                "Graph Neo4j đang rỗng. Hãy chạy `python graph_rag/src/ingest.py --reset-graph` trước."
            )
    finally:
        driver.close()


def _graph_is_ready(config) -> bool:
    driver = build_neo4j_driver(config)
    try:
        return graph_ready(driver, config)
    finally:
        driver.close()


# Cache bước warm-up để web server không phải kiểm tra Neo4j và model lặp lại quá nhiều.
@lru_cache(maxsize=8)
def warm_up_graph(overrides_key: str = "{}") -> dict[str, Any]:
    """Warm-up GraphRAG và cache kết quả kiểm tra readiness."""

    overrides = json.loads(overrides_key)
    config = load_config(overrides=overrides)
    configure_models(config)
    is_ready = _graph_is_ready(config)
    return {
        "graph_ready": is_ready,
        "retrieval_top_n": config.retrieval_top_n,
        "query_fusion_enabled": config.query_fusion_enabled,
    }


# Truy vấn GraphRAG mới: Neo4j vector search + graph expansion + LLM synthesis.
def answer_question(question: str, top_k: int | None = None, runtime_overrides: dict | None = None) -> ChatAnswer:
    """Trả lời một câu hỏi bằng pipeline GraphRAG Neo4j hiện tại."""

    config = load_config(overrides=runtime_overrides)
    if top_k is not None and top_k > 0:
        config = load_config(
            overrides={
                **(runtime_overrides or {}),
                "retrieval_top_n": int(top_k),
            }
        )
    configure_models(config)
    if not _graph_is_ready(config):
        return ChatAnswer(question=question, answer=config.query_refusal_response, facts=[])

    facts = retrieve_facts(question, config)
    threshold = config.retrieval_similarity_threshold if _should_apply_similarity_threshold(config) else 0.0
    if not facts:
        return ChatAnswer(question=question, answer=config.query_refusal_response, facts=[])
    if threshold > 0 and max(float(fact.score) for fact in facts) < threshold:
        return ChatAnswer(
            question=question,
            answer=config.query_refusal_response,
            facts=[_to_source_fact(fact) for fact in facts],
        )

    answer = _generate_answer(question, facts, config)
    if not answer:
        answer = config.query_refusal_response
    return ChatAnswer(
        question=question,
        answer=answer,
        facts=[_to_source_fact(fact) for fact in facts],
    )


# Rút gọn RetrievedFact thành SourceFact trước khi trả về UI hoặc benchmark.
def _to_source_fact(fact: RetrievedFact) -> SourceFact:
    """Chuyển RetrievedFact sang payload nhẹ hơn để trả về phía client."""

    return SourceFact(
        content=fact.text,
        relative_path=fact.relative_path,
        heading_path=fact.heading_path,
        score=round(float(fact.score), 4),
    )
