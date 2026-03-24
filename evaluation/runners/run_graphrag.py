from __future__ import annotations

import time
from typing import Any

import requests

from evaluation.common import EvalExample, EvalPrediction, SourceRecord


def healthcheck(system_config: dict[str, Any], timeout: tuple[int, int]) -> None:
    response = requests.get(
        f"{system_config['base_url'].rstrip('/')}{system_config['health_endpoint']}",
        timeout=timeout,
    )
    response.raise_for_status()


def _extract_stateless_answer_and_sources(payload: dict[str, Any]) -> tuple[str, list[SourceRecord]]:
    sources = [
        SourceRecord(
            label=str(item.get("source") or item.get("relative_path") or item.get("heading_path") or "graph_fact"),
            content=str(item.get("content") or ""),
            metadata={
                "heading_path": item.get("heading_path") or "",
                "score": item.get("score"),
                "relative_path": item.get("relative_path") or "",
            },
        )
        for item in (payload.get("sources") or [])
    ]
    return str(payload.get("answer") or "").strip(), sources


def _extract_session_answer_and_sources(payload: dict[str, Any]) -> tuple[str, list[SourceRecord]]:
    session = payload.get("session") or {}
    messages = session.get("messages") or []
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not assistant_messages:
        return "", []

    latest = assistant_messages[-1]
    facts = latest.get("facts") or []
    sources = [
        SourceRecord(
            label=str(fact.get("relative_path") or fact.get("heading_path") or "graph_fact"),
            content=str(fact.get("content") or ""),
            metadata={
                "heading_path": fact.get("heading_path") or "",
                "score": fact.get("score"),
                "relative_path": fact.get("relative_path") or "",
            },
        )
        for fact in facts
    ]
    return str(latest.get("content") or "").strip(), sources


def run_example(example: EvalExample, system_config: dict[str, Any], timeout: tuple[int, int]) -> EvalPrediction:
    started = time.perf_counter()
    base_url = system_config["base_url"].rstrip("/")
    stateless_endpoint = system_config.get("chat_endpoint")

    if stateless_endpoint:
        try:
            response = requests.post(
                f"{base_url}{stateless_endpoint}",
                json={"query": example.question},
                timeout=timeout,
            )
            response.raise_for_status()
            answer, sources = _extract_stateless_answer_and_sources(response.json())
            return EvalPrediction(
                system=system_config["name"],
                example_id=example.id,
                question=example.question,
                answer=answer,
                sources=sources,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            return EvalPrediction(
                system=system_config["name"],
                example_id=example.id,
                question=example.question,
                answer="",
                sources=[],
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )

    create_url = f"{base_url}{system_config['create_chat_endpoint']}"
    session_id = ""
    try:
        created = requests.post(create_url, timeout=timeout)
        created.raise_for_status()
        session_id = str(created.json()["session"]["id"])

        send_url = f"{base_url}{system_config['send_message_endpoint'].format(session_id=session_id)}"
        response = requests.post(send_url, json={"question": example.question}, timeout=timeout)
        response.raise_for_status()
        answer, sources = _extract_session_answer_and_sources(response.json())
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer=answer,
            sources=sources,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:
        return EvalPrediction(
            system=system_config["name"],
            example_id=example.id,
            question=example.question,
            answer="",
            sources=[],
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )
    finally:
        if session_id:
            delete_endpoint = system_config.get("delete_chat_endpoint")
            if delete_endpoint:
                try:
                    requests.delete(
                        f"{base_url}{delete_endpoint.format(session_id=session_id)}",
                        timeout=timeout,
                    )
                except Exception:
                    pass
