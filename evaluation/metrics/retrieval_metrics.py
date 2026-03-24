from __future__ import annotations

import math

from evaluation.common import EvalExample, SourceRecord, flatten_sources_text, normalize_text


def _keyword_threshold(keywords: list[str]) -> int:
    # Tập keyword càng dài thì yêu cầu hit tối thiểu càng cao để bớt đánh dấu nhầm.
    if not keywords:
        return 1
    if len(keywords) <= 2:
        return 1
    return 2


def _source_bundle(source: SourceRecord) -> str:
    # Gom mọi trường của một source thành một text normalize để kiểm tra tính liên quan.
    text = " ".join(
        [
            source.label,
            source.content,
            " ".join(f"{key} {value}" for key, value in source.metadata.items()),
        ]
    )
    return normalize_text(text)


def source_relevance_flags(example: EvalExample, sources: list[SourceRecord]) -> list[int]:
    # Gán cờ relevant/không relevant cho từng source dựa trên source hint hoặc context keyword.
    hints = [normalize_text(item) for item in (example.expected_source_hints or []) if item]
    keywords = [normalize_text(item) for item in (example.context_keywords or example.expected_keywords or []) if item]
    threshold = _keyword_threshold(keywords)

    flags: list[int] = []
    for source in sources:
        bundle = _source_bundle(source)
        hint_match = any(hint and hint in bundle for hint in hints)
        keyword_hits = sum(1 for keyword in keywords if keyword and keyword in bundle)
        is_relevant = hint_match or keyword_hits >= threshold
        flags.append(1 if is_relevant else 0)
    return flags


def source_hint_hit(sources: list[SourceRecord], expected_source_hints: list[str]) -> float:
    # Kiểm tra ít nhất một source có chứa hint nguồn mong đợi hay không.
    if not expected_source_hints:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    return 1.0 if any(normalize_text(hint) in bundle for hint in expected_source_hints) else 0.0


def source_keyword_coverage(sources: list[SourceRecord], expected_keywords: list[str]) -> float:
    # Đo mức bao phủ keyword quan trọng ở toàn bộ tập nguồn retrieve.
    if not expected_keywords:
        return 1.0 if not sources else 0.0
    bundle = normalize_text(flatten_sources_text(sources))
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in bundle)
    return hits / len(expected_keywords)


def relevant_count(example: EvalExample) -> int:
    # Số nguồn ground-truth được dùng làm mẫu số cho recall/AP.
    return len(example.expected_source_hints or [])


def precision_at_k(flags: list[int], k: int) -> float:
    # Trong top-k có bao nhiêu phần tử là relevant.
    if k <= 0:
        return 0.0
    top_k = flags[:k]
    if not top_k:
        return 0.0
    return sum(top_k) / k


def recall_at_k(flags: list[int], total_relevant: int, k: int) -> float:
    # Top-k đã thu hồi được bao nhiêu phần trăm số nguồn relevant kỳ vọng.
    if total_relevant <= 0:
        return 1.0 if not flags[:k] else 0.0
    return min(1.0, sum(flags[:k]) / total_relevant)


def f1_at_k(flags: list[int], total_relevant: int, k: int) -> float:
    # Điểm cân bằng giữa precision@k và recall@k.
    precision = precision_at_k(flags, k)
    recall = recall_at_k(flags, total_relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(flags: list[int], k: int) -> float:
    # Chỉ cần có ít nhất một nguồn đúng trong top-k là đạt 1.
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: list[int]) -> float:
    # Nguồn đúng xuất hiện càng sớm thì điểm reciprocal rank càng cao.
    for index, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / index
    return 0.0


def average_precision(flags: list[int], total_relevant: int) -> float:
    # Average Precision thưởng cho việc đưa nhiều nguồn đúng lên vị trí đầu.
    if total_relevant <= 0:
        return 1.0 if not any(flags) else 0.0
    precision_sum = 0.0
    hits = 0
    for index, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            precision_sum += hits / index
    if not total_relevant:
        return 0.0
    return min(1.0, precision_sum / total_relevant)


def ndcg_at_k(flags: list[int], k: int) -> float:
    # nDCG đánh giá chất lượng ranking có xét trọng số giảm dần theo vị trí.
    top_k = flags[:k]
    if not top_k:
        return 0.0

    dcg = sum(flag / math.log2(index + 2) for index, flag in enumerate(top_k))
    ideal_flags = sorted(flags, reverse=True)[:k]
    ideal_dcg = sum(flag / math.log2(index + 2) for index, flag in enumerate(ideal_flags))
    if ideal_dcg == 0:
        return 1.0 if not any(top_k) else 0.0
    return dcg / ideal_dcg
