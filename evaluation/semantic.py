from __future__ import annotations

import math
from typing import Any

import requests
from requests import RequestException

from evaluation.common import normalize_text


# Semantic scorer with cached embedding lookups.
class SemanticScorer:

    # Khởi tạo semantic scorer với endpoint, model và timeout tương ứng.
    def __init__(self, base_url: str, model: str, timeout: tuple[int, int]):

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self._cache: dict[str, list[float]] = {}

    # Lấy embedding cho một đoạn text, có normalize và cache trước khi gọi API.
    def embed(self, text: str) -> list[float]:

        normalized = normalize_text(text)
        if not normalized:
            return []
        if normalized in self._cache:
            return self._cache[normalized]

        vector = self._embed_with_fallback(text)
        self._cache[normalized] = vector
        return vector

    # Thử nhiều độ dài input và nhiều endpoint để tăng khả năng tương thích.
    def _embed_with_fallback(self, text: str) -> list[float]:

        candidates = [text[:2500], text[:1200], text[:600]]
        for candidate in candidates:
            if not candidate.strip():
                continue
            for path, payload in (
                (f"{self.base_url}/embeddings", {"model": self.model, "input": candidate}),
                (
                    self.base_url.replace("/v1", "") + "/api/embed",
                    {"model": self.model, "input": candidate},
                ),
            ):
                try:
                    response = self.session.post(path, json=payload, timeout=self.timeout)
                    response.raise_for_status()
                    return _extract_vector(response.json())
                except RequestException:
                    continue
        return []

    # Tính cosine similarity trực tiếp trên embedding của hai đoạn text.
    def cosine_text(self, left: str, right: str) -> float:

        return cosine_similarity(self.embed(left), self.embed(right))


# Tính cosine similarity an toàn, trả 0 nếu vector rỗng hoặc lệch chiều.
def cosine_similarity(left: list[float], right: list[float]) -> float:

    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, numerator / (left_norm * right_norm))


# Rút vector embedding từ nhiều dạng payload response phổ biến.
def _extract_vector(payload: dict[str, Any]) -> list[float]:

    if "data" in payload:
        return list(payload["data"][0]["embedding"])
    if "embedding" in payload:
        return list(payload["embedding"])
    if "embeddings" in payload:
        return list(payload["embeddings"][0])
    return []
