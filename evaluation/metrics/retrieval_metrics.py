from __future__ import annotations

import math

from evaluation.common import EvalExample, SourceRecord, extract_numbers, flatten_sources_text, normalize_text
from evaluation.semantic import SemanticScorer


# Gom mọi trường của một source thành một text normalize để so sánh công bằng giữa chunk và graph fact.
def _source_bundle(source: SourceRecord) -> str:
    text = " ".join(
        [
            source.label,
            source.content,
            " ".join(f"{key} {value}" for key, value in source.metadata.items()),
        ]
    )
    return normalize_text(text)


# Đo tỷ lệ keyword kỳ vọng xuất hiện trong text đã normalize.
def _keyword_overlap_ratio(text: str, keywords: list[str]) -> float:
    normalized = normalize_text(text)
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits / len(keywords)


# Tính semantic similarity giữa hai chuỗi nếu semantic scorer đang được bật.
def _semantic_similarity(scorer: SemanticScorer | None, left: str, right: str) -> float:
    if scorer is None or not left.strip() or not right.strip():
        return 0.0
    return scorer.cosine_text(left, right)


# Đo tỷ lệ số liệu kỳ vọng xuất hiện trong text nguồn.
def _numeric_overlap_ratio(text: str, expected_numbers: list[str]) -> float:
    if not expected_numbers:
        return 0.0
    source_numbers = set(extract_numbers(text))
    if not source_numbers:
        return 0.0
    hits = sum(1 for number in expected_numbers if number in source_numbers)
    return hits / len(expected_numbers)


# Chấm độ liên quan của từng source dựa trên nội dung thật thay vì tên file hay dạng biểu diễn.
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


# Giữ lại cờ nhị phân hit/miss từ content score để phục vụ một số báo cáo cũ.
def source_relevance_flags(
    example: EvalExample,
    sources: list[SourceRecord],
    scorer: SemanticScorer | None = None,
    threshold: float = 0.55,
) -> list[int]:
    return [1 if score >= threshold else 0 for score in source_relevance_scores(example, sources, scorer)]


# Giữ source hint hit như tín hiệu diagnostic để trace dataset.
def source_hint_hit(sources: list[SourceRecord], expected_source_hints: list[str]) -> float:
    if not expected_source_hints:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    return 1.0 if any(normalize_text(hint) in bundle for hint in expected_source_hints) else 0.0


# Đo mức bao phủ keyword quan trọng trên toàn bộ tập source retrieve.
def source_keyword_coverage(sources: list[SourceRecord], expected_keywords: list[str]) -> float:
    # Do muc bao phu keyword quan trong tren toan bo tap source retrieve.
    if not expected_keywords:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in bundle)
    return hits / len(expected_keywords)


# Số evidence ground-truth kỳ vọng dùng làm mẫu số cho recall.
def relevant_count(example: EvalExample) -> int:
    count = len(example.expected_source_hints or [])
    if count > 0:
        return count
    return 0 if example.refusal_expected else 1


# Precision mềm: trung bình relevance score trong top-k.
def precision_at_k(relevances: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / k


# Recall mềm: tổng relevance mass trong top-k chia cho số evidence kỳ vọng.
def recall_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    # Recall mem: tong relevance mass trong top-k chia cho so evidence ground-truth ky vong.
    if total_relevant <= 0:
        return 1.0 if not relevances[:k] else 0.0
    return min(1.0, sum(relevances[:k]) / total_relevant)


# Điểm cân bằng giữa precision mềm và recall mềm.
def f1_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    # Diem can bang giua precision mem va recall mem.
    precision = precision_at_k(relevances, k)
    recall = recall_at_k(relevances, total_relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Chỉ cần có ít nhất một source đủ tốt trong top-k là hit.
def hit_rate_at_k(relevances: list[float], k: int, threshold: float = 0.55) -> float:
    # Chi can co it nhat mot source duoc support score tot trong top-k la dat 1.
    return 1.0 if any(score >= threshold for score in relevances[:k]) else 0.0


# Weighted reciprocal rank ưu tiên source đúng sớm và có mức support cao.
def mrr(relevances: list[float]) -> float:
    # Weighted reciprocal rank uu tien source dung som va co muc support cao.
    best = 0.0
    for index, score in enumerate(relevances, start=1):
        best = max(best, score / index)
    return best


# Average Precision mềm trên graded relevance để giảm bias khác kiến trúc.
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


# nDCG graded relevance đánh giá thứ tự source ngay cả khi source chỉ mang một phần evidence.
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
