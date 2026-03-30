from __future__ import annotations

from difflib import SequenceMatcher

from evaluation.common import normalize_text, tokenize


# Chấm exact match sau khi normalize để giảm khác biệt bề mặt.
def exact_match_score(reference: str, answer: str) -> float:
    # Điểm tuyệt đối: chỉ được 1 khi answer trùng reference sau normalize.
    return 1.0 if normalize_text(reference) == normalize_text(answer) else 0.0


# So khớp ở mức token để đo độ "đúng ý" ngay cả khi wording không hoàn toàn giống nhau.
def token_f1_score(reference: str, answer: str) -> tuple[float, float, float]:
    # So khớp token-level để đo mức độ “đúng ý” ngay cả khi câu chữ không hoàn toàn giống nhau.
    reference_tokens = tokenize(reference)
    answer_tokens = tokenize(answer)
    if not reference_tokens and not answer_tokens:
        return 1.0, 1.0, 1.0
    if not reference_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    reference_counts: dict[str, int] = {}
    answer_counts: dict[str, int] = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    for token in answer_tokens:
        answer_counts[token] = answer_counts.get(token, 0) + 1

    overlap = 0
    for token, count in reference_counts.items():
        overlap += min(count, answer_counts.get(token, 0))

    precision = overlap / max(1, len(answer_tokens))
    recall = overlap / max(1, len(reference_tokens))
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


# Độ giống ký tự hữu ích cho số điện thoại, email, URL hoặc đáp án ngắn.
def char_similarity(reference: str, answer: str) -> float:
    # Độ giống ký tự hữu ích cho số điện thoại, email, URL hoặc câu trả lời ngắn.
    return SequenceMatcher(a=normalize_text(reference), b=normalize_text(answer)).ratio()


# Tính tỷ lệ keyword kỳ vọng thực sự xuất hiện trong câu trả lời.
def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    # Tính tỷ lệ keyword kỳ vọng thực sự xuất hiện trong câu trả lời.
    if not expected_keywords:
        return 1.0
    answer_norm = normalize_text(answer)
    hits = sum(1 for keyword in expected_keywords if normalize_text(keyword) in answer_norm)
    return hits / len(expected_keywords)
