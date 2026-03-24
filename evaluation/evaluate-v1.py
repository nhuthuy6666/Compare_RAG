from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import (  # noqa: E402
    EvalPrediction,
    dataclass_to_dict,
    ensure_dir,
    load_structured_config,
    write_csv,
    write_json,
)
from evaluation.dataset.loader import load_examples  # noqa: E402
from evaluation.metrics.rag_metrics_v1 import evaluate_prediction_v1  # noqa: E402
from evaluation.runners import run_baseline, run_graphrag, run_hybrid  # noqa: E402
from evaluation.semantic import SemanticScorer  # noqa: E402


RUNNERS: dict[str, tuple[Callable, Callable]] = {
    "baseline": (run_baseline.healthcheck, run_baseline.run_example),
    "hybrid": (run_hybrid.healthcheck, run_hybrid.run_example),
    "graphrag": (run_graphrag.healthcheck, run_graphrag.run_example),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3 RAG systems via HTTP with V1 scoring.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument(
        "--system",
        choices=("all", "baseline", "hybrid", "graphrag"),
        default="all",
        help="System to evaluate.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N questions.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


DETAIL_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "V1Core",
        [
            ("overall_score", "overall"),
            ("recall_at_k", "recall@k"),
            ("faithfulness", "faithfulness"),
            ("answer_relevance", "answer_relevancy"),
            ("context_precision", "context_precision"),
        ],
    ),
    (
        "Generation",
        [
            ("answer_quality", "answer_quality"),
            ("exact_match", "exact_match"),
            ("token_f1", "token_f1"),
            ("char_similarity", "char_similarity"),
            ("semantic_similarity", "semantic_similarity"),
            ("keyword_coverage", "keyword_coverage"),
        ],
    ),
    (
        "Retrieval",
        [
            ("retrieval_quality", "retrieval_quality"),
            ("source_hint_hit", "source_hint_hit"),
            ("source_keyword_coverage", "source_keyword_coverage"),
            ("precision_at_3", "precision@3"),
            ("recall_at_3", "recall@3"),
            ("f1_at_3", "f1@3"),
            ("hit_rate_at_3", "hit@3"),
            ("mrr", "mrr"),
            ("map", "map"),
            ("ndcg_at_3", "ndcg@3"),
            ("ndcg_at_5", "ndcg@5"),
        ],
    ),
    (
        "System",
        [
            ("latency_ms", "latency_ms"),
            ("errors", "errors"),
        ],
    ),
]


def aggregate_metrics(system: str, rows: list[dict]) -> dict:
    numeric_fields = [
        "refusal_correct",
        "exact_match",
        "token_f1",
        "char_similarity",
        "semantic_similarity",
        "keyword_coverage",
        "answer_relevance",
        "context_relevance",
        "faithfulness",
        "hallucination_rate",
        "source_hint_hit",
        "source_keyword_coverage",
        "precision_at_1",
        "precision_at_3",
        "precision_at_5",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "f1_at_1",
        "f1_at_3",
        "f1_at_5",
        "hit_rate_at_3",
        "mrr",
        "map",
        "ndcg_at_3",
        "ndcg_at_5",
        "answer_quality",
        "retrieval_quality",
        "recall_at_k",
        "context_precision",
        "overall_score",
        "latency_ms",
    ]
    summary = {"system": system, "samples": len(rows)}
    for field in numeric_fields:
        summary[field] = mean([float(row[field]) for row in rows])
    summary["errors"] = sum(1 for row in rows if row.get("error"))
    return summary


def print_detailed_summary(system_name: str, summary: dict, metric_rows: list[dict]) -> None:
    topic_scores: dict[str, list[float]] = {}
    for row in metric_rows:
        topic_scores.setdefault(str(row["topic"]), []).append(float(row["overall_score"]))

    best_topics = sorted(
        ((topic, statistics.fmean(scores)) for topic, scores in topic_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    worst_topics = sorted(
        ((topic, statistics.fmean(scores)) for topic, scores in topic_scores.items()),
        key=lambda item: item[1],
    )[:3]

    print(f"\n[{system_name}] Detailed Metrics", flush=True)
    for group_name, fields in DETAIL_GROUPS:
        formatted = " | ".join(
            f"{label}={float(summary.get(field, 0)):.4f}" if field != "errors" else f"{label}={int(summary.get(field, 0))}"
            for field, label in fields
        )
        print(f"  {group_name}: {formatted}", flush=True)

    if best_topics:
        best_text = ", ".join(f"{topic}={score:.4f}" for topic, score in best_topics)
        print(f"  Best topics: {best_text}", flush=True)
    if worst_topics:
        worst_text = ", ".join(f"{topic}={score:.4f}" for topic, score in worst_topics)
        print(f"  Worst topics: {worst_text}", flush=True)


def run_system(
    *,
    system_name: str,
    system_config: dict,
    examples: list,
    timeout: tuple[int, int],
    outputs_dir: Path,
    results_dir: Path,
    semantic_scorer: SemanticScorer | None,
) -> dict:
    healthcheck, runner = RUNNERS[system_name]
    healthcheck(system_config, timeout)

    predictions: list[EvalPrediction] = []
    metric_rows: list[dict] = []
    for index, example in enumerate(examples, start=1):
        print(f"[{system_name}] {index}/{len(examples)} - {example.id}", flush=True)
        prediction = runner(example, system_config, timeout)
        predictions.append(prediction)
        row = {"system": system_name}
        row.update(evaluate_prediction_v1(example, prediction, semantic_scorer=semantic_scorer))
        metric_rows.append(row)

    summary = aggregate_metrics(system_name, metric_rows)
    output_payload = {
        "system": system_name,
        "summary": summary,
        "predictions": dataclass_to_dict(predictions),
    }

    output_path = write_json(outputs_dir / f"{system_name}_outputs.json", output_payload)
    metrics_path = write_csv(results_dir / f"{system_name}_metrics.csv", metric_rows)
    print_detailed_summary(system_name, summary, metric_rows)
    print(f"  Saved: {metrics_path}", flush=True)
    print(f"  Saved: {output_path}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    config = load_structured_config(args.config)

    examples = load_examples(config["dataset_path"])
    if args.limit:
        examples = examples[: args.limit]

    outputs_dir = ensure_dir(config["outputs_dir"])
    results_dir = ensure_dir(config["results_dir"])
    timeout = (
        int(config.get("connect_timeout_seconds", 10)),
        int(config.get("request_timeout_seconds", 180)),
    )
    semantic_config = config.get("semantic") or {}
    semantic_scorer = None
    if semantic_config.get("enabled", True):
        semantic_scorer = SemanticScorer(
            base_url=str(semantic_config.get("base_url") or "http://127.0.0.1:11434/v1"),
            model=str(semantic_config.get("model") or "bge-m3:latest"),
            timeout=timeout,
        )

    systems = list(RUNNERS.keys()) if args.system == "all" else [args.system]
    summaries: list[dict] = []
    for system_name in systems:
        summary = run_system(
            system_name=system_name,
            system_config=config[system_name],
            examples=examples,
            timeout=timeout,
            outputs_dir=outputs_dir,
            results_dir=results_dir,
            semantic_scorer=semantic_scorer,
        )
        summaries.append(summary)

    write_csv(results_dir / "comparison.csv", summaries)
    comparison_json = write_json(results_dir / "comparison.json", summaries)
    comparison_csv = results_dir / "comparison.csv"

    print("\nSummary", flush=True)
    for summary in summaries:
        print(
            f"- {summary['system']}: overall={summary['overall_score']}, "
            f"recall@k={summary['recall_at_k']}, faithfulness={summary['faithfulness']}, "
            f"answer_relevancy={summary['answer_relevance']}, context_precision={summary['context_precision']}, "
            f"latency_ms={summary['latency_ms']}, errors={summary['errors']}",
            flush=True,
        )
    print(f"\nSaved: {comparison_csv}", flush=True)
    print(f"Saved: {comparison_json}", flush=True)


if __name__ == "__main__":
    main()
