from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import VectorStoreIndex  # noqa: E402

from evaluation.common import EvalPrediction, SourceRecord, ensure_dir  # noqa: E402
from evaluation.dataset.loader import load_examples  # noqa: E402
from evaluation.metrics.rag_metrics_v1 import evaluate_prediction_v1  # noqa: E402
from evaluation.semantic import SemanticScorer  # noqa: E402
from llamaindex_shared.common import (  # noqa: E402
    build_query_engine,
    collect_sources,
    configure_models,
    load_nodes,
    load_shared_config,
)


DEFAULT_RESULTS_DIR = "evaluation/results/ground_truth_baseline_no_fusion"

BASELINE_NO_FUSION_OVERRIDES = {
    "retrieval_top_n": 6,
    "query_fusion_enabled": False,
    "generation_temperature": 0.0,
    "generation_top_p": 1.0,
    "max_output_tokens": 1024,
    "llm_seed": 17,
}


# Khai báo và parse tham số CLI cho lượt đánh giá ground-truth riêng của baseline no-fusion.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone ground-truth evaluation for baseline RAG without fusion."
    )
    parser.add_argument("--split", choices=("all", "dev", "held_out_test"), default="all", help="Dataset split.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N examples.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Separate output dir for this evaluation.")
    parser.add_argument(
        "--semantic-base-url",
        default="http://127.0.0.1:11434/v1",
        help="Embedding endpoint used by the semantic scorer.",
    )
    parser.add_argument("--semantic-model", default="bge-m3:latest", help="Embedding model for semantic scoring.")
    parser.add_argument("--connect-timeout", type=int, default=10, help="HTTP connect timeout in seconds.")
    parser.add_argument("--request-timeout", type=int, default=180, help="HTTP request timeout in seconds.")
    parser.add_argument("--invalid-answer-threshold", type=float, default=0.45, help="Threshold for invalid answers.")
    parser.add_argument(
        "--hallucination-threshold",
        type=float,
        default=0.5,
        help="Threshold for hallucination case detection.",
    )
    parser.add_argument(
        "--severe-hallucination-threshold",
        type=float,
        default=0.8,
        help="Threshold for severe hallucination detection.",
    )
    parser.add_argument("--recall-threshold", type=float, default=0.5, help="Threshold for retrieval failure.")
    parser.add_argument(
        "--context-precision-threshold",
        type=float,
        default=0.4,
        help="Threshold for retrieval failure via context precision.",
    )
    parser.add_argument(
        "--grounding-gap-faithfulness-threshold",
        type=float,
        default=0.5,
        help="Faithfulness threshold for grounding-gap detection.",
    )
    parser.add_argument(
        "--grounding-gap-answer-threshold",
        type=float,
        default=0.6,
        help="Answer relevance threshold for grounding-gap detection.",
    )
    return parser.parse_args()


# Tính trung bình của một cột metric trên toàn bộ mẫu và làm tròn 4 chữ số.
def mean(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return round(statistics.fmean(float(row.get(field, 0.0) or 0.0) for row in rows), 4)


# In console an toàn kể cả khi terminal Windows không hỗ trợ Unicode đầy đủ.
def emit_console(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


# Dựng baseline RAG no-fusion trong bộ nhớ để tránh phụ thuộc Qdrant/HTTP server.
def build_baseline_query_engine() -> object:
    config = load_shared_config(
        rag_id="baseline",
        collection_name="ntu_ground_truth_baseline_no_fusion",
        overrides=BASELINE_NO_FUSION_OVERRIDES,
    )
    configure_models(config)
    nodes = load_nodes(config)
    index = VectorStoreIndex(nodes=nodes, show_progress=False)
    query_engine = build_query_engine(index, config, enable_hybrid=False)
    return config, query_engine, len(nodes)


# Chạy một câu hỏi qua baseline no-fusion và chuẩn hóa sources về schema evaluator.
def run_query(
    *,
    query_engine,
    question: str,
    retrieval_top_n: int,
) -> tuple[str, list[SourceRecord]]:
    response = query_engine.query(question)
    raw_sources = collect_sources(response, limit=retrieval_top_n)
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
        for item in raw_sources
    ]
    return str(response).strip(), sources


# Parse một giá trị bất kỳ về bool theo cách dùng chung trong evaluator.
def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() == "true"


# Gắn nhãn các kiểu lỗi chính của RAG từ metric ground-truth trên từng mẫu.
def detect_failure_tags(row: dict, args: argparse.Namespace) -> list[str]:
    tags: list[str] = []
    refusal_expected = to_bool(row.get("refusal_expected"))
    refusal_predicted = to_bool(row.get("refusal_predicted"))
    refusal_correct = float(row.get("refusal_correct", 0.0) or 0.0)
    answer_quality = float(row.get("answer_quality", 0.0) or 0.0)
    answer_relevance = float(row.get("answer_relevance", 0.0) or 0.0)
    faithfulness = float(row.get("faithfulness", 0.0) or 0.0)
    hallucination = float(row.get("hallucination_rate", 0.0) or 0.0)
    recall = float(row.get("recall_at_k", 0.0) or 0.0)
    context_precision = float(row.get("context_precision", 0.0) or 0.0)

    if refusal_expected and refusal_correct < 1.0:
        tags.append("missed_refusal")
    if not refusal_expected and refusal_predicted:
        tags.append("false_refusal")
    if (not refusal_expected and answer_quality < args.invalid_answer_threshold) or (
        refusal_expected and refusal_correct < 1.0
    ):
        tags.append("invalid")
    if hallucination >= args.severe_hallucination_threshold:
        tags.append("severe_hallucination")
    elif hallucination >= args.hallucination_threshold:
        tags.append("hallucination")
    if recall < args.recall_threshold or context_precision < args.context_precision_threshold:
        tags.append("retrieval_failure")
    if (
        answer_relevance >= args.grounding_gap_answer_threshold
        and faithfulness < args.grounding_gap_faithfulness_threshold
    ):
        tags.append("grounding_gap")
    return tags


# Tính tỷ lệ các mẫu thỏa một điều kiện bất kỳ trên toàn bộ tập đánh giá.
def rate(rows: list[dict], predicate) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


# Gom metric theo topic để tìm ra nhóm câu hỏi mà baseline no-fusion yếu nhất.
def topic_breakdown(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("topic") or "unknown")].append(row)

    items: list[dict] = []
    for topic, topic_rows in sorted(grouped.items()):
        items.append(
            {
                "topic": topic,
                "samples": len(topic_rows),
                "invalid_rate": rate(topic_rows, lambda row: "invalid" in detect_failure_tags(row, args)),
                "hallucination_case_rate": rate(
                    topic_rows,
                    lambda row: float(row.get("hallucination_rate", 0.0) or 0.0) >= args.hallucination_threshold,
                ),
                "retrieval_failure_rate": rate(
                    topic_rows,
                    lambda row: (
                        float(row.get("recall_at_k", 0.0) or 0.0) < args.recall_threshold
                        or float(row.get("context_precision", 0.0) or 0.0) < args.context_precision_threshold
                    ),
                ),
                "mean_overall_score": mean(topic_rows, "overall_score"),
            }
        )
    items.sort(key=lambda item: (item["invalid_rate"], item["hallucination_case_rate"], -item["mean_overall_score"]), reverse=True)
    return items


# Chọn ra các ví dụ lỗi nặng nhất để đưa vào báo cáo Markdown cuối cùng.
def worst_examples(rows: list[dict], predictions_by_id: dict[str, dict], args: argparse.Namespace, top_n: int = 10) -> list[dict]:
    ranked: list[dict] = []
    for row in rows:
        tags = detect_failure_tags(row, args)
        if not tags:
            continue
        example_id = str(row.get("example_id") or "")
        prediction = predictions_by_id.get(example_id) or {}
        answer = str(prediction.get("answer") or "").replace("\n", " ").strip()
        if len(answer) > 220:
            answer = answer[:217].rstrip() + "..."
        severity = (
            0.35 * (1.0 - float(row.get("answer_quality", 0.0) or 0.0))
            + 0.30 * float(row.get("hallucination_rate", 0.0) or 0.0)
            + 0.20 * (1.0 - float(row.get("faithfulness", 0.0) or 0.0))
            + 0.15 * (1.0 - float(row.get("recall_at_k", 0.0) or 0.0))
        )
        ranked.append(
            {
                "example_id": example_id,
                "topic": row.get("topic", ""),
                "question": row.get("question", ""),
                "answer_preview": answer,
                "failure_tags": ", ".join(tags),
                "severity_score": round(severity, 4),
                "answer_quality": round(float(row.get("answer_quality", 0.0) or 0.0), 4),
                "faithfulness": round(float(row.get("faithfulness", 0.0) or 0.0), 4),
                "hallucination_rate": round(float(row.get("hallucination_rate", 0.0) or 0.0), 4),
                "recall_at_k": round(float(row.get("recall_at_k", 0.0) or 0.0), 4),
            }
        )
    return sorted(ranked, key=lambda item: item["severity_score"], reverse=True)[:top_n]


# Render báo cáo Markdown duy nhất cho toàn bộ lượt đánh giá ground-truth.
def build_report(
    *,
    split: str,
    sample_count: int,
    node_count: int,
    summary: dict,
    topic_rows: list[dict],
    worst_rows: list[dict],
) -> str:
    lines = [
        "# Ground-Truth Evaluation for Baseline No Fusion",
        "",
        f"- System: `baseline`",
        f"- Profile: `shared_no_fusion_v1`",
        f"- Split: `{split}`",
        f"- Samples: `{sample_count}`",
        f"- Corpus nodes indexed in-memory: `{node_count}`",
        "",
        "## Summary",
        "",
        f"- Overall score: `{summary['overall_score']}`",
        f"- Faithfulness: `{summary['faithfulness']}`",
        f"- Hallucination rate: `{summary['hallucination_rate']}`",
        f"- Recall@k: `{summary['recall_at_k']}`",
        f"- Context precision: `{summary['context_precision']}`",
        f"- Invalid rate: `{summary['invalid_rate']}`",
        f"- Hallucination case rate: `{summary['hallucination_case_rate']}` | severe: `{summary['severe_hallucination_rate']}`",
        f"- Retrieval failure rate: `{summary['retrieval_failure_rate']}`",
        f"- False refusal rate: `{summary['false_refusal_rate']}` | missed refusal rate: `{summary['missed_refusal_rate']}`",
        f"- Grounding gap rate: `{summary['grounding_gap_rate']}`",
        "",
        "## Worst Topics",
        "",
    ]
    for item in topic_rows[:8]:
        lines.append(
            f"- {item['topic']}: invalid={item['invalid_rate']}, hallucination={item['hallucination_case_rate']}, retrieval_failure={item['retrieval_failure_rate']}, overall={item['mean_overall_score']}"
        )
    lines.extend(["", "## Worst Examples", ""])
    for item in worst_rows:
        lines.append(
            f"- {item['example_id']} [{item['topic']}]: tags={item['failure_tags']}, severity={item['severity_score']}, faithfulness={item['faithfulness']}, hallucination={item['hallucination_rate']}, recall@k={item['recall_at_k']}"
        )
        lines.append(f"  Q: {item['question']}")
        if item["answer_preview"]:
            lines.append(f"  A: {item['answer_preview']}")
    return "\n".join(lines) + "\n"


# Entry point của pipeline ground-truth riêng cho baseline no-fusion.
# 1. Đọc CLI và chuẩn bị thư mục output riêng biệt.
# 2. Nạp dataset theo split yêu cầu và giới hạn số mẫu nếu có.
# 3. Dựng baseline no-fusion trong bộ nhớ và khởi tạo semantic scorer.
# 4. Lần lượt chạy từng câu hỏi, thu answer/sources và chấm metric theo ground-truth.
# 5. Tổng hợp summary, topic breakdown, worst examples rồi render một file report.md duy nhất.
# 6. Ghi report ra thư mục kết quả riêng và in lại đường dẫn để người dùng mở trực tiếp.
def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    timeout = (args.connect_timeout, args.request_timeout)

    examples = load_examples("evaluation/dataset/testset.json", split=args.split)
    if args.limit:
        examples = examples[: args.limit]

    emit_console(f"Building in-memory baseline no-fusion index for {len(examples)} examples...")
    config, query_engine, node_count = build_baseline_query_engine()
    semantic_scorer = SemanticScorer(
        base_url=args.semantic_base_url,
        model=args.semantic_model,
        timeout=timeout,
    )

    metric_rows: list[dict] = []
    predictions_payload: list[dict] = []

    for index, example in enumerate(examples, start=1):
        emit_console(f"[baseline-no-fusion] {index}/{len(examples)} - {example.id}")
        started = time.perf_counter()
        answer, sources = run_query(
            query_engine=query_engine,
            question=example.question,
            retrieval_top_n=config.retrieval_top_n,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        prediction = EvalPrediction(
            system="baseline",
            example_id=example.id,
            question=example.question,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            error="",
        )
        row = evaluate_prediction_v1(example, prediction, semantic_scorer=semantic_scorer)
        metric_rows.append(row)
        predictions_payload.append(
            {
                "example_id": example.id,
                "question": example.question,
                "answer": answer,
                "latency_ms": round(latency_ms, 2),
                "sources": [
                    {
                        "label": source.label,
                        "content": source.content,
                        "metadata": source.metadata,
                    }
                    for source in sources
                ],
            }
        )

    predictions_by_id = {item["example_id"]: item for item in predictions_payload}
    summary = {
        "system": "baseline",
        "profile_name": "shared_no_fusion_v1",
        "split": args.split,
        "samples": len(metric_rows),
        "overall_score": mean(metric_rows, "overall_score"),
        "answer_quality": mean(metric_rows, "answer_quality"),
        "retrieval_quality": mean(metric_rows, "retrieval_quality"),
        "faithfulness": mean(metric_rows, "faithfulness"),
        "hallucination_rate": mean(metric_rows, "hallucination_rate"),
        "answer_relevance": mean(metric_rows, "answer_relevance"),
        "recall_at_k": mean(metric_rows, "recall_at_k"),
        "context_precision": mean(metric_rows, "context_precision"),
        "latency_ms": mean(metric_rows, "latency_ms"),
        "invalid_rate": rate(metric_rows, lambda row: "invalid" in detect_failure_tags(row, args)),
        "hallucination_case_rate": rate(
            metric_rows,
            lambda row: float(row.get("hallucination_rate", 0.0) or 0.0) >= args.hallucination_threshold,
        ),
        "severe_hallucination_rate": rate(
            metric_rows,
            lambda row: float(row.get("hallucination_rate", 0.0) or 0.0) >= args.severe_hallucination_threshold,
        ),
        "retrieval_failure_rate": rate(
            metric_rows,
            lambda row: (
                float(row.get("recall_at_k", 0.0) or 0.0) < args.recall_threshold
                or float(row.get("context_precision", 0.0) or 0.0) < args.context_precision_threshold
            ),
        ),
        "false_refusal_rate": rate(
            metric_rows,
            lambda row: (not to_bool(row.get("refusal_expected"))) and to_bool(row.get("refusal_predicted")),
        ),
        "missed_refusal_rate": rate(
            metric_rows,
            lambda row: to_bool(row.get("refusal_expected")) and (not to_bool(row.get("refusal_predicted"))),
        ),
        "grounding_gap_rate": rate(metric_rows, lambda row: "grounding_gap" in detect_failure_tags(row, args)),
    }

    topic_rows = topic_breakdown(metric_rows, args)
    worst_rows = worst_examples(metric_rows, predictions_by_id, args, top_n=10)
    report = build_report(
        split=args.split,
        sample_count=len(metric_rows),
        node_count=node_count,
        summary=summary,
        topic_rows=topic_rows,
        worst_rows=worst_rows,
    )

    (results_dir / "report.md").write_text(report, encoding="utf-8")

    emit_console(report)
    emit_console(f"Saved: {results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
