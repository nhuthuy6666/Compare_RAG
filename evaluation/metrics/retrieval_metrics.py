from __future__ import annotations

import math

from evaluation.common import EvalExample, SourceRecord, extract_numbers, flatten_sources_text, normalize_text
from evaluation.semantic import SemanticScorer


def _source_bundle(source: SourceRecord) -> str:
    # Gom moi truong cua mot source thanh mot text normalize de so sanh cong bang giua chunk va graph fact.
    text = " ".join(
        [
            source.label,
            source.content,
            " ".join(f"{key} {value}" for key, value in source.metadata.items()),
        ]
    )
    return normalize_text(text)


def _keyword_overlap_ratio(text: str, keywords: list[str]) -> float:
    normalized = normalize_text(text)
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits / len(keywords)


def _semantic_similarity(scorer: SemanticScorer | None, left: str, right: str) -> float:
    if scorer is None or not left.strip() or not right.strip():
        return 0.0
    return scorer.cosine_text(left, right)


def _numeric_overlap_ratio(text: str, expected_numbers: list[str]) -> float:
    if not expected_numbers:
        return 0.0
    source_numbers = set(extract_numbers(text))
    if not source_numbers:
        return 0.0
    hits = sum(1 for number in expected_numbers if number in source_numbers)
    return hits / len(expected_numbers)


def source_relevance_scores(
    example: EvalExample,
    sources: list[SourceRecord],
    scorer: SemanticScorer | None = None,
) -> list[float]:
    # Cham do lien quan cua moi source bang noi dung that su, khong phu thuoc vao ten file hay kieu bieu dien.
    context_keywords = example.context_keywords or example.expected_keywords
    answer_keywords = example.answer_keywords or example.expected_keywords
    expected_numbers = list(
        dict.fromkeys(
            extract_numbers(example.reference_answer)
            + [number for keyword in answer_keywords for number in extract_numbers(keyword)]
        )
    )

    scores: list[float] = []
    for source in sources:
        bundle = _source_bundle(source)
        text = source.content or source.label or bundle
        semantic_question = _semantic_similarity(scorer, example.question, text)
        semantic_reference = _semantic_similarity(scorer, example.reference_answer, text)
        context_coverage = _keyword_overlap_ratio(bundle, context_keywords)
        answer_coverage = _keyword_overlap_ratio(bundle, answer_keywords)
        numeric_coverage = _numeric_overlap_ratio(bundle, expected_numbers)

        if expected_numbers:
            score = (
                (0.30 * semantic_reference)
                + (0.20 * semantic_question)
                + (0.25 * context_coverage)
                + (0.10 * answer_coverage)
                + (0.15 * numeric_coverage)
            )
        else:
            score = (
                (0.40 * semantic_reference)
                + (0.25 * semantic_question)
                + (0.20 * context_coverage)
                + (0.15 * answer_coverage)
            )
        scores.append(max(0.0, min(1.0, score)))
    return scores


def source_relevance_flags(
    example: EvalExample,
    sources: list[SourceRecord],
    scorer: SemanticScorer | None = None,
    threshold: float = 0.55,
) -> list[int]:
    # Giu lai co nhi phan cho cac bao cao can hit/miss, nhung nen duoc suy ra tu content score.
    return [1 if score >= threshold else 0 for score in source_relevance_scores(example, sources, scorer)]


def source_hint_hit(sources: list[SourceRecord], expected_source_hints: list[str]) -> float:
    # Chi giu cho muc diagnostic de trace dataset, khong nen la tin hieu chinh de xep hang kien truc.
    if not expected_source_hints:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    return 1.0 if any(normalize_text(hint) in bundle for hint in expected_source_hints) else 0.0


def source_keyword_coverage(sources: list[SourceRecord], expected_keywords: list[str]) -> float:
    # Do muc bao phu keyword quan trong tren toan bo tap source retrieve.
    if not expected_keywords:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in bundle)
    return hits / len(expected_keywords)


def relevant_count(example: EvalExample) -> int:
    # So nguon ground-truth duoc dung lam mau so recall; fallback ve 1 neu khong co source hint cho cau tra loi fact.
    count = len(example.expected_source_hints or [])
    if count > 0:
        return count
    return 0 if example.refusal_expected else 1


def precision_at_k(relevances: list[float], k: int) -> float:
    # Precision mem: lay trung binh relevance score trong top-k va van phat khi he tra ve it hon k source.
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / k


def recall_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    # Recall mem: tong relevance mass trong top-k chia cho so evidence ground-truth ky vong.
    if total_relevant <= 0:
        return 1.0 if not relevances[:k] else 0.0
    return min(1.0, sum(relevances[:k]) / total_relevant)


def f1_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    # Diem can bang giua precision mem va recall mem.
    precision = precision_at_k(relevances, k)
    recall = recall_at_k(relevances, total_relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(relevances: list[float], k: int, threshold: float = 0.55) -> float:
    # Chi can co it nhat mot source duoc support score tot trong top-k la dat 1.
    return 1.0 if any(score >= threshold for score in relevances[:k]) else 0.0


def mrr(relevances: list[float]) -> float:
    # Weighted reciprocal rank uu tien source dung som va co muc support cao.
    best = 0.0
    for index, score in enumerate(relevances, start=1):
        best = max(best, score / index)
    return best


def average_precision(relevances: list[float], total_relevant: int) -> float:
    # Average Precision mem tren graded relevance de giam bias do khac nhau ve hinh thuc source.
    if total_relevant <= 0:
        return 1.0 if not any(relevances) else 0.0
    precision_sum = 0.0
    for index, score in enumerate(relevances, start=1):
        if score <= 0:
            continue
        precision_sum += precision_at_k(relevances, index) * score
    return min(1.0, precision_sum / total_relevant)


def ndcg_at_k(relevances: list[float], k: int) -> float:
    # nDCG graded relevance danh gia thu tu source ngay ca khi source mang dung mot phan evidence.
    top_k = relevances[:k]
    if not top_k:
        return 0.0

    dcg = sum(score / math.log2(index + 2) for index, score in enumerate(top_k))
    ideal_relevances = sorted(relevances, reverse=True)[:k]
    ideal_dcg = sum(score / math.log2(index + 2) for index, score in enumerate(ideal_relevances))
    if ideal_dcg == 0:
        return 1.0 if not any(top_k) else 0.0
    return dcg / ideal_dcg
