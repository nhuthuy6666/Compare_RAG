from __future__ import annotations


# Công thức tổng hợp V1 ưu tiên recall, faithfulness và độ liên quan của câu trả lời.
def blend_scores_v1(
    *,
    recall_at_k: float,
    faithfulness_score: float,
    answer_relevance_score: float,
    context_precision_score: float,
) -> float:
    # Công thức tổng hợp V1 ưu tiên recall, faithfulness và độ liên quan của câu trả lời.
    return (
        (0.25 * recall_at_k)
        + (0.35 * faithfulness_score)
        + (0.25 * answer_relevance_score)
        + (0.15 * context_precision_score)
    )
