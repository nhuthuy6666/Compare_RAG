from __future__ import annotations

import math
from typing import Any

import requests
from requests import RequestException

from evaluation.common import normalize_text


class SemanticScorer:
    """Bọc logic gọi embedding service và tính semantic similarity có cache."""

    def __init__(self, base_url: str, model: str, timeout: tuple[int, int]):
        """Khởi tạo semantic scorer với endpoint, model và timeout tương ứng."""

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        """Lấy embedding cho một đoạn text, có normalize và cache trước khi gọi API."""

        normalized = normalize_text(text)
        if not normalized:
            return []
        if normalized in self._cache:
            return self._cache[normalized]

        vector = self._embed_with_fallback(text)
        self._cache[normalized] = vector
        return vector

    def _embed_with_fallback(self, text: str) -> list[float]:
        """Thử nhiều độ dài input và nhiều endpoint để tăng khả năng tương thích."""

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

    def cosine_text(self, left: str, right: str) -> float:
        """Tính cosine similarity trực tiếp trên embedding của hai đoạn text."""

        return cosine_similarity(self.embed(left), self.embed(right))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Tính cosine similarity an toàn, trả 0 nếu vector rỗng hoặc lệch chiều."""

    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, numerator / (left_norm * right_norm))


def _extract_vector(payload: dict[str, Any]) -> list[float]:
    """Rút vector embedding từ nhiều dạng payload response phổ biến."""

    if "data" in payload:
        return list(payload["data"][0]["embedding"])
    if "embedding" in payload:
        return list(payload["embedding"])
    if "embeddings" in payload:
        return list(payload["embeddings"][0])
    return []
