from __future__ import annotations

import time
from typing import Any

import requests

from evaluation.common import EvalExample, EvalPrediction, SourceRecord


# Kiểm tra app Hybrid RAG sẵn sàng nhận request hay chưa.
def healthcheck(system_config: dict[str, Any], timeout: tuple[int, int]) -> None:
    # Kiểm tra app Hybrid RAG sẵn sàng nhận request hay chưa.
    response = requests.get(
        f"{system_config['base_url'].rstrip('/')}{system_config['health_endpoint']}",
        timeout=timeout,
    )
    response.raise_for_status()


# Gọi endpoint chat của Hybrid RAG cho một câu hỏi benchmark.
def run_example(example: EvalExample, system_config: dict[str, Any], timeout: tuple[int, int]) -> EvalPrediction:
    # Gọi endpoint chat của Hybrid RAG cho một câu hỏi benchmark.
    started = time.perf_counter()
    url = f"{system_config['base_url'].rstrip('/')}{system_config['chat_endpoint']}"

    try:
        response = requests.post(
            url,
            json={
                "query": example.question,
                "benchmark_profile": system_config.get("benchmark_profile") or {},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        # Hybrid trả về danh sách chunk nên label được gắn thêm chunk_id để trace kết quả.
        sources = [
            SourceRecord(
                label=f"{item.get('source', 'hybrid_source')}#{item.get('chunk_id', '?')}",
                content=str(item.get("content") or ""),
                metadata={"chunk_id": item.get("chunk_id")},
            )
            for item in (data.get("sources") or [])
        ]
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer=str(data.get("answer") or "").strip(),
            sources=sources,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        # Giữ lỗi ở mức từng mẫu thay vì làm hỏng cả batch evaluation.
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer="",
            sources=[],
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
