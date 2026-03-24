from __future__ import annotations

from evaluation.common import EvalExample, EvalPrediction, refusal_detected
from evaluation.metrics.generation_metrics import (
    char_similarity,
    exact_match_score,
    keyword_coverage,
    token_f1_score,
)
from evaluation.metrics.ranking_metrics_v1 import blend_scores_v1
from evaluation.metrics.retrieval_metrics import (
    average_precision,
    f1_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevant_count,
    source_hint_hit,
    source_keyword_coverage,
    source_relevance_flags,
)
from evaluation.metrics.semantic_metrics import (
    answer_relevance,
    context_relevance,
    faithfulness,
    hallucination_rate,
    semantic_similarity,
)
from evaluation.semantic import SemanticScorer


RETRIEVAL_K = 3


def _empty_result(example: EvalExample, prediction: EvalPrediction) -> dict:
    # Khi request lỗi, trả về một hàng metric “an toàn” với điểm 0 để pipeline vẫn chạy hết.
    return {
        "example_id": example.id,
        "question": example.question,
        "topic": example.topic,
        "refusal_expected": example.refusal_expected,
        "refusal_predicted": False,
        "refusal_correct": 0.0,
        "exact_match": 0.0,
        "token_precision": 0.0,
        "token_recall": 0.0,
        "token_f1": 0.0,
        "char_similarity": 0.0,
        "semantic_similarity": 0.0,
        "keyword_coverage": 0.0,
        "answer_relevance": 0.0,
        "context_relevance": 0.0,
        "faithfulness": 0.0,
        "hallucination_rate": 1.0,
        "source_hint_hit": 0.0,
        "source_keyword_coverage": 0.0,
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "precision_at_5": 0.0,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "f1_at_1": 0.0,
        "f1_at_3": 0.0,
        "f1_at_5": 0.0,
        "hit_rate_at_3": 0.0,
        "mrr": 0.0,
        "map": 0.0,
        "ndcg_at_3": 0.0,
        "ndcg_at_5": 0.0,
        "answer_quality": 0.0,
        "retrieval_quality": 0.0,
        "recall_at_k": 0.0,
        "context_precision": 0.0,
        "overall_score": 0.0,
        "latency_ms": round(prediction.latency_ms, 2),
        "source_count": len(prediction.sources),
        "error": prediction.error,
    }


def evaluate_prediction_v1(
    example: EvalExample,
    prediction: EvalPrediction,
    semantic_scorer: SemanticScorer | None = None,
) -> dict:
    # Hàm lõi V1: giữ lại metric chi tiết cũ nhưng thay công thức overall theo bộ trọng số mới.
    if prediction.error:
        return _empty_result(example, prediction)

    refusal_predicted = refusal_detected(prediction.answer)
    refusal_correct = 1.0 if refusal_predicted == example.refusal_expected else 0.0

    exact = exact_match_score(example.reference_answer, prediction.answer)
    precision, recall, token_f1 = token_f1_score(example.reference_answer, prediction.answer)
    char_sim = char_similarity(example.reference_answer, prediction.answer)
    sem_sim = semantic_similarity(example.reference_answer, prediction.answer, semantic_scorer)
    answer_keyword_cov = keyword_coverage(
        prediction.answer,
        example.answer_keywords or example.expected_keywords,
    )
    answer_rel = answer_relevance(example, prediction.answer, semantic_scorer)
    context_rel = context_relevance(example, prediction.sources, semantic_scorer)
    faith = faithfulness(prediction.answer, prediction.sources, semantic_scorer)
    hallucination = hallucination_rate(prediction.answer, prediction.sources, semantic_scorer)

    hit = source_hint_hit(prediction.sources, example.expected_source_hints)
    source_kw_cov = source_keyword_coverage(
        prediction.sources,
        example.context_keywords or example.expected_keywords,
    )

    flags = source_relevance_flags(example, prediction.sources)
    total_relevant = relevant_count(example)
    p1 = precision_at_k(flags, 1)
    p3 = precision_at_k(flags, 3)
    p5 = precision_at_k(flags, 5)
    r1 = recall_at_k(flags, total_relevant, 1)
    r3 = recall_at_k(flags, total_relevant, 3)
    r5 = recall_at_k(flags, total_relevant, 5)
    f1_1 = f1_at_k(flags, total_relevant, 1)
    f1_3 = f1_at_k(flags, total_relevant, 3)
    f1_5 = f1_at_k(flags, total_relevant, 5)
    hit3 = hit_rate_at_k(flags, 3)
    reciprocal_rank = mrr(flags)
    mean_average_precision = average_precision(flags, total_relevant)
    ndcg3 = ndcg_at_k(flags, 3)
    ndcg5 = ndcg_at_k(flags, 5)

    if example.refusal_expected:
        answer_quality = (0.6 * refusal_correct) + (0.4 * answer_rel)
        retrieval_quality = hit3
    else:
        answer_quality = (
            (0.20 * exact)
            + (0.20 * token_f1)
            + (0.20 * char_sim)
            + (0.20 * sem_sim)
            + (0.20 * answer_keyword_cov)
        )
        if refusal_predicted:
            answer_quality *= 0.25

        retrieval_quality = (
            (0.10 * hit)
            + (0.10 * source_kw_cov)
            + (0.10 * p3)
            + (0.10 * r3)
            + (0.10 * f1_3)
            + (0.15 * reciprocal_rank)
            + (0.15 * mean_average_precision)
            + (0.10 * ndcg3)
            + (0.10 * ndcg5)
        )

    overall = blend_scores_v1(
        recall_at_k=r3,
        faithfulness_score=faith,
        answer_relevance_score=answer_rel,
        context_precision_score=p3,
    )

    return {
        "example_id": example.id,
        "question": example.question,
        "topic": example.topic,
        "refusal_expected": example.refusal_expected,
        "refusal_predicted": refusal_predicted,
        "refusal_correct": round(refusal_correct, 4),
        "exact_match": round(exact, 4),
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(token_f1, 4),
        "char_similarity": round(char_sim, 4),
        "semantic_similarity": round(sem_sim, 4),
        "keyword_coverage": round(answer_keyword_cov, 4),
        "answer_relevance": round(answer_rel, 4),
        "context_relevance": round(context_rel, 4),
        "faithfulness": round(faith, 4),
        "hallucination_rate": round(hallucination, 4),
        "source_hint_hit": round(hit, 4),
        "source_keyword_coverage": round(source_kw_cov, 4),
        "precision_at_1": round(p1, 4),
        "precision_at_3": round(p3, 4),
        "precision_at_5": round(p5, 4),
        "recall_at_1": round(r1, 4),
        "recall_at_3": round(r3, 4),
        "recall_at_5": round(r5, 4),
        "f1_at_1": round(f1_1, 4),
        "f1_at_3": round(f1_3, 4),
        "f1_at_5": round(f1_5, 4),
        "hit_rate_at_3": round(hit3, 4),
        "mrr": round(reciprocal_rank, 4),
        "map": round(mean_average_precision, 4),
        "ndcg_at_3": round(ndcg3, 4),
        "ndcg_at_5": round(ndcg5, 4),
        "answer_quality": round(answer_quality, 4),
        "retrieval_quality": round(retrieval_quality, 4),
        "recall_at_k": round(r3, 4),
        "context_precision": round(p3, 4),
        "overall_score": round(overall, 4),
        "latency_ms": round(prediction.latency_ms, 2),
        "source_count": len(prediction.sources),
        "error": prediction.error or "",
    }
