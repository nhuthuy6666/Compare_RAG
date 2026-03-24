from __future__ import annotations

import time
from typing import Any

import requests

from evaluation.common import EvalExample, EvalPrediction, SourceRecord, load_env_file


def _headers(system_config: dict[str, Any]) -> dict[str, str]:
    # Ghép API key từ config hoặc file .env để gọi AnythingLLM.
    env = load_env_file(system_config.get("env_file", ""))
    api_key = (system_config.get("api_key") or env.get("ANYTHINGLLM_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Thieu ANYTHINGLLM_API_KEY cho baseline.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def healthcheck(system_config: dict[str, Any], timeout: tuple[int, int]) -> None:
    # Hỗ trợ cả baseline AnythingLLM cũ lẫn baseline local qua HTTP.
    headers = _headers(system_config) if system_config.get("workspace_slug") else None
    response = requests.get(
        f"{system_config['base_url'].rstrip('/')}{system_config['health_endpoint']}",
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()


def run_example(example: EvalExample, system_config: dict[str, Any], timeout: tuple[int, int]) -> EvalPrediction:
    # Gọi baseline cho một câu hỏi và chuẩn hóa response về EvalPrediction.
    started = time.perf_counter()
    is_anythingllm = bool(system_config.get("workspace_slug"))
    headers = _headers(system_config) if is_anythingllm else None
    endpoint = system_config["chat_endpoint"]
    if is_anythingllm:
        endpoint = endpoint.format(workspace_slug=system_config["workspace_slug"])
    url = f"{system_config['base_url'].rstrip('/')}{endpoint}"
    payload = (
        {
            "message": example.question,
            "mode": system_config.get("mode", "chat"),
        }
        if is_anythingllm
        else {"query": example.question}
    )

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if is_anythingllm:
            # Mỗi source của AnythingLLM được map sang cấu trúc chung label/content/metadata.
            sources = [
                SourceRecord(
                    label=str(item.get("title") or item.get("url") or "baseline_source"),
                    content=str(item.get("text") or ""),
                    metadata={
                        "score": item.get("score"),
                        "description": item.get("description"),
                    },
                )
                for item in (data.get("sources") or [])
            ]
            answer = str(data.get("textResponse") or "").strip()
        else:
            sources = [
                SourceRecord(
                    label=f"{item.get('source', 'baseline_source')}#{item.get('chunk_id', '?')}",
                    content=str(item.get("content") or ""),
                    metadata={
                        "chunk_id": item.get("chunk_id"),
                        "score": item.get("score"),
                        "relative_path": item.get("relative_path"),
                    },
                )
                for item in (data.get("sources") or [])
            ]
            answer = str(data.get("answer") or "").strip()
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer=answer,
            sources=sources,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        # Không làm vỡ toàn bộ lượt chạy khi một câu hỏi gặp lỗi HTTP/JSON.
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer="",
            sources=[],
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
