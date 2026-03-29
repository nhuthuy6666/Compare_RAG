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
}


def content_tokens(text: str) -> set[str]:
    # Lay cac token mang nhieu noi dung hon de phuc vu kiem tra support cua claim.
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1}


def keyword_overlap_ratio(text: str, keywords: list[str]) -> float:
    # Ty le keyword ky vong xuat hien trong text da normalize.
    normalized = normalize_text(text)
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in normalized)
    return hits / len(keywords)


def semantic_similarity(reference: str, answer: str, scorer: SemanticScorer | None) -> float:
    # So do gan nghia giua reference answer va answer bang embedding cosine.
    if scorer is None:
        return 0.0
    return scorer.cosine_text(reference, answer)


def answer_relevance(example: EvalExample, answer: str, scorer: SemanticScorer | None) -> float:
    # Do answer co tra loi dung cau hoi va bam vao dap an mong doi, khong phat nang vi khac wording.
    semantic_question = scorer.cosine_text(example.question, answer) if scorer else 0.0
    semantic_reference = scorer.cosine_text(example.reference_answer, answer) if scorer else 0.0
    lexical = keyword_overlap_ratio(answer, example.answer_keywords or example.expected_keywords)
    return (0.35 * semantic_question) + (0.40 * semantic_reference) + (0.25 * lexical)


def context_relevance(example: EvalExample, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Cham tung source theo muc do giup tra loi cau hoi va reference, roi lay trung binh top 3 tot nhat.
    if not sources:
        return 0.0

    context_keywords = example.context_keywords or example.expected_keywords
    scored: list[float] = []
    for source in sources:
        text = source.content or source.label
        semantic_question = scorer.cosine_text(example.question, text) if scorer else 0.0
        semantic_reference = scorer.cosine_text(example.reference_answer, text) if scorer else 0.0
        lexical = keyword_overlap_ratio(text, context_keywords)
        scored.append((0.25 * semantic_question) + (0.40 * semantic_reference) + (0.35 * lexical))

    top_items = sorted(scored, reverse=True)[: min(3, len(scored))]
    return sum(top_items) / len(top_items)


def split_claims(answer: str) -> list[str]:
    # Tach cau tra loi dai thanh cac claim nho de kiem tra do bam nguon tung y.
    raw_claims = re.split(r"[.!?\n;]+", answer)
    claims = [claim.strip(" -\t") for claim in raw_claims if claim.strip()]
    return [claim for claim in claims if len(tokenize(claim)) >= 3]


def _claim_supported(claim: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> bool:
    # Mot claim duoc xem la co support khi co it nhat mot source du gan ve nghia va thuat ngu.
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
    return best_score >= 0.55


def faithfulness(answer: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Ty le claim trong cau tra loi duoc nguon ho tro.
    claims = split_claims(answer)
    if not claims:
        return 1.0 if not answer.strip() else 0.0
    supported = sum(1 for claim in claims if _claim_supported(claim, sources, scorer))
    return supported / len(claims)


def hallucination_rate(answer: str, sources: list[SourceRecord], scorer: SemanticScorer | None) -> float:
    # Hallucination rate la phan bo sung cua faithfulness.
    return 1.0 - faithfulness(answer, sources, scorer)
