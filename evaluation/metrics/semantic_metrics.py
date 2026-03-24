from __future__ import annotations

import re

from evaluation.common import EvalExample, SourceRecord, extract_numbers, normalize_text, tokenize
from evaluation.semantic import SemanticScorer


STOPWORDS = {
    "la",
    "va",
    "cua",
    "cho",
    "theo",
    "nam",
    "tren",
    "duoc",
    "voi",
    "mot",
    "cac",
    "nhung",
    "trong",
    "tai",
    "co",
    "o",
    "tu",
    "ve",
    "thi",
    "do",
    "day",
    "thi",
    "la",
}


def content_tokens(text: str) -> set[str]:
    # Lấy các token mang nhiều nội dung hơn để phục vụ kiểm tra support của claim.
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def keyword_overlap_ratio(text: str, keywords: list[str]) -> float:
    # Tỷ lệ keyword kỳ vọng xuất hiện trong text đã normalize.
    normalized = normalize_text(text)
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits / len(keywords)


def semantic_similarity(reference: str, answer: str, scorer: SemanticScorer | None) -> float:
    # So độ gần nghĩa giữa reference answer và answer bằng embedding cosine.
    if scorer is None:
        return 0.0
    return scorer.cosine_text(reference, answer)


def answer_relevance(example: EvalExample, answer: str, scorer: SemanticScorer | None) -> float:
    # Đo answer có thực sự trả lời câu hỏi hay chỉ nói chung chung.
    semantic = scorer.cosine_text(example.question, answer) if scorer else 0.0
    lexical = keyword_overlap_ratio(answer, example.answer_keywords or example.expected_keywords)
    return (0.7 * semantic) + (0.3 * lexical)


def context_relevance(example: EvalExample, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Chấm từng nguồn retrieve theo mức liên quan tới câu hỏi, rồi lấy trung bình top 3 tốt nhất.
    if not sources:
        return 0.0

    context_keywords = example.context_keywords or example.expected_keywords
    scored: list[float] = []
    for source in sources:
        text = source.content or source.label
        semantic = scorer.cosine_text(example.question, text) if scorer else 0.0
        lexical = keyword_overlap_ratio(text, context_keywords)
        scored.append((0.65 * semantic) + (0.35 * lexical))

    top_items = sorted(scored, reverse=True)[: min(3, len(scored))]
    return sum(top_items) / len(top_items)


def split_claims(answer: str) -> list[str]:
    # Tách câu trả lời dài thành các claim nhỏ để kiểm tra độ bám nguồn từng ý.
    raw_claims = re.split(r"[.!?\n;]+", answer)
    claims = [claim.strip(" -\t") for claim in raw_claims if claim.strip()]
    return [claim for claim in claims if len(tokenize(claim)) >= 3]


def _claim_supported(claim: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> bool:
    # Một claim được xem là có support khi có ít nhất một nguồn đủ gần về nghĩa và thuật ngữ.
    claim_numbers = set(extract_numbers(claim))
    claim_terms = content_tokens(claim)

    best_score = 0.0
    numeric_supported = not claim_numbers
    for source in sources:
        text = source.content or source.label
        semantic = scorer.cosine_text(claim, text) if scorer else 0.0
        source_terms = content_tokens(text)
        lexical = 0.0 if not claim_terms else len(claim_terms & source_terms) / len(claim_terms)
        if claim_numbers:
            source_numbers = set(extract_numbers(text))
            numeric_supported = numeric_supported or bool(claim_numbers & source_numbers)
        combined = (0.6 * semantic) + (0.4 * lexical)
        best_score = max(best_score, combined)

    if claim_numbers and not numeric_supported:
        return False
    return best_score >= 0.58


def faithfulness(answer: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Tỷ lệ claim trong câu trả lời được nguồn hỗ trợ.
    claims = split_claims(answer)
    if not claims:
        return 1.0 if not answer.strip() else 0.0
    supported = sum(1 for claim in claims if _claim_supported(claim, sources, scorer))
    return supported / len(claims)


def hallucination_rate(answer: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Hallucination rate là phần bổ sung của faithfulness.
    return 1.0 - faithfulness(answer, sources, scorer)
