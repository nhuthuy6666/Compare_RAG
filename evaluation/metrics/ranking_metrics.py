from __future__ import annotations


def blend_scores(
    *,
    answer_quality: float,
    retrieval_quality: float,
    faithfulness_score: float,
    relevance_score: float,
    refusal_score: float,
) -> float:
    # Trộn các nhóm metric thành điểm tổng hợp cuối cùng của một câu hỏi.
    return (
        (0.30 * answer_quality)
        + (0.20 * retrieval_quality)
        + (0.20 * faithfulness_score)
        + (0.20 * relevance_score)
        + (0.10 * refusal_score)
    )
