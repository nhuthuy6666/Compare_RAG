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
from evaluation.compare import build_comparison_report  # noqa: E402
from evaluation.metrics.rag_metrics_v1 import evaluate_prediction_v1  # noqa: E402
from evaluation.policy import (  # noqa: E402
    load_benchmark_policy,
    load_locked_profiles,
    load_profile_candidates,
    mode_budget,
    resolve_profile_payload,
    resolve_mode,
    resolve_split,
)
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
    parser.add_argument("--split", choices=("all", "dev", "held_out_test"), default=None, help="Dataset split to use.")
    parser.add_argument("--mode", choices=("controlled", "best_tuned"), default=None, help="Benchmark mode.")
    parser.add_argument("--profile-source", choices=("locked", "candidate"), default="locked", help="Profile source.")
    parser.add_argument("--profile-name", default="", help="Explicit profile name when using candidate source.")
    parser.add_argument("--seed", type=int, default=0, help="Recorded seed for repeat runs.")
    parser.add_argument("--run-label", default="", help="Optional label for this run.")
    parser.add_argument("--outputs-dir", default=None, help="Override outputs dir for this run.")
    parser.add_argument("--results-dir", default=None, help="Override results dir for this run.")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep intermediate CSV/JSON artifacts. By default only comparison.md is written.",
    )
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
            ("latency_budget_violation_rate", "latency_budget_violation_rate"),
            ("errors", "errors"),
        ],
    ),
]


def aggregate_metrics(system: str, rows: list[dict], *, metadata: dict[str, object], budget: dict[str, object]) -> dict:
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
    summary = {"system": system, "samples": len(rows), **metadata}
    for field in numeric_fields:
        summary[field] = mean([float(row[field]) for row in rows])
    summary["errors"] = sum(1 for row in rows if row.get("error"))
    summary["profile_name"] = str(metadata.get("profile_name") or "default")
    summary["latency_budget_ms"] = float(budget.get("latency_budget_ms") or 0.0)
    summary["latency_budget_violation_rate"] = mean([float(row.get("latency_budget_violation", 0.0)) for row in rows])
    summary["retrieval_budget_top_n"] = int(budget.get("retrieval_top_n") or 0)
    summary["evaluation_k"] = int(budget.get("evaluation_k") or 0)
    summary["max_output_tokens"] = int(budget.get("max_output_tokens") or 0)
    summary["token_budget_status"] = str(budget.get("token_budget_status") or "unknown")
    return summary


def aggregate_strength_buckets(system: str, rows: list[dict], *, metadata: dict[str, object], budget: dict[str, object]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        bucket = str(row.get("strength_bucket") or "neutral")
        grouped.setdefault(bucket, []).append(row)

    summaries: list[dict] = []
    for bucket, bucket_rows in sorted(grouped.items()):
        bucket_summary = aggregate_metrics(system, bucket_rows, metadata=metadata, budget=budget)
        bucket_summary["strength_bucket"] = bucket
        summaries.append(bucket_summary)
    return summaries


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

    print(
        f"\n[{system_name}] Detailed Metrics | mode={summary['mode']} | split={summary['split']} | profile={summary['profile_name']}",
        flush=True,
    )
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
    keep_artifacts: bool,
    semantic_scorer: SemanticScorer | None,
    metadata: dict[str, object],
    budget: dict[str, object],
) -> dict:
    healthcheck, runner = RUNNERS[system_name]
    healthcheck(system_config, timeout)

    predictions: list[EvalPrediction] = []
    metric_rows: list[dict] = []
    latency_budget_ms = float(budget.get("latency_budget_ms") or 0.0)
    for index, example in enumerate(examples, start=1):
        print(f"[{system_name}] {index}/{len(examples)} - {example.id}", flush=True)
        prediction = runner(example, system_config, timeout)
        predictions.append(prediction)
        row = {
            "system": system_name,
            "strength_bucket": example.strength_bucket,
            "split": example.split,
            **metadata,
        }
        row.update(evaluate_prediction_v1(example, prediction, semantic_scorer=semantic_scorer))
        row["latency_budget_ms"] = latency_budget_ms
        row["latency_budget_violation"] = (
            1.0 if latency_budget_ms > 0 and float(row.get("latency_ms") or 0.0) > latency_budget_ms else 0.0
        )
        metric_rows.append(row)

    summary = aggregate_metrics(system_name, metric_rows, metadata=metadata, budget=budget)
    bucket_summaries = aggregate_strength_buckets(system_name, metric_rows, metadata=metadata, budget=budget)
    print_detailed_summary(system_name, summary, metric_rows)
    if keep_artifacts:
        output_payload = {
            "system": system_name,
            "metadata": metadata,
            "budget": budget,
            "summary": summary,
            "strength_breakdown": bucket_summaries,
            "predictions": dataclass_to_dict(predictions),
        }
        output_path = write_json(outputs_dir / f"{system_name}_outputs.json", output_payload)
        metrics_path = write_csv(results_dir / f"{system_name}_metrics.csv", metric_rows)
        print(f"  Saved: {metrics_path}", flush=True)
        print(f"  Saved: {output_path}", flush=True)
    return {
        "summary": summary,
        "bucket_summaries": bucket_summaries,
    }


def main() -> None:
    args = parse_args()
    config = load_structured_config(args.config)
    policy = load_benchmark_policy(config)
    locked_profiles = load_locked_profiles(policy)
    profile_candidates = load_profile_candidates(policy)
    resolved_mode = resolve_mode(policy, args.mode)
    resolved_split = resolve_split(policy, args.split)
    budget = mode_budget(policy, resolved_mode)

    examples = load_examples(config["dataset_path"], split=resolved_split)
    if args.limit:
        examples = examples[: args.limit]

    results_dir = ensure_dir(args.results_dir or config["results_dir"])
    outputs_dir = ensure_dir(args.outputs_dir or config["outputs_dir"]) if args.keep_artifacts else Path(config["outputs_dir"])
    timeout = (
        int(budget.get("connect_timeout_seconds") or config.get("connect_timeout_seconds", 10)),
        int(budget.get("request_timeout_seconds") or config.get("request_timeout_seconds", 180)),
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
    if args.profile_source == "candidate" and args.system == "all" and not args.profile_name:
        raise ValueError("When --profile-source candidate is used with --system all, --profile-name is required.")
    summaries: list[dict] = []
    strength_rows: list[dict] = []
    for system_name in systems:
        profile_payload = resolve_profile_payload(
            locked_profiles=locked_profiles,
            profile_candidates=profile_candidates,
            mode=resolved_mode,
            system_name=system_name,
            source=args.profile_source,
            profile_name=args.profile_name or None,
        )
        metadata = {
            "mode": resolved_mode,
            "split": resolved_split,
            "seed": args.seed,
            "run_label": args.run_label,
            "profile_name": str(profile_payload.get("profile_name") or "default"),
            "profile_source": args.profile_source,
        }
        system_config = dict(config[system_name])
        system_config["benchmark_profile"] = {
            "profile_name": metadata["profile_name"],
            "runtime_overrides": dict(profile_payload.get("runtime_overrides") or {}),
        }
        result = run_system(
            system_name=system_name,
            system_config=system_config,
            examples=examples,
            timeout=timeout,
            outputs_dir=outputs_dir,
            results_dir=results_dir,
            keep_artifacts=args.keep_artifacts,
            semantic_scorer=semantic_scorer,
            metadata=metadata,
            budget=budget,
        )
        summaries.append(result["summary"])
        strength_rows.extend(result["bucket_summaries"])

    report_rows = [{key: str(value) for key, value in row.items()} for row in summaries]
    report_strength_rows = [{key: str(value) for key, value in row.items()} for row in strength_rows]
    comparison_md = results_dir / "comparison.md"
    comparison_md.write_text(build_comparison_report(report_rows, report_strength_rows), encoding="utf-8")

    if args.keep_artifacts:
        comparison_csv = write_csv(results_dir / "comparison.csv", summaries)
        comparison_json = write_json(results_dir / "comparison.json", summaries)
        strength_csv = write_csv(results_dir / "strength_breakdown.csv", strength_rows)

    print("\nSummary", flush=True)
    print(
        f"- mode={resolved_mode}, split={resolved_split}, seed={args.seed}, run_label={args.run_label or 'default'}",
        flush=True,
    )
    for summary in summaries:
        print(
            f"- {summary['system']}: overall={summary['overall_score']}, "
            f"recall@k={summary['recall_at_k']}, faithfulness={summary['faithfulness']}, "
            f"answer_relevancy={summary['answer_relevance']}, context_precision={summary['context_precision']}, "
            f"latency_ms={summary['latency_ms']}, latency_budget_violation_rate={summary['latency_budget_violation_rate']}, "
            f"profile={summary['profile_name']}, errors={summary['errors']}",
            flush=True,
        )
    print(f"\nSaved: {comparison_md}", flush=True)
    if args.keep_artifacts:
        print(f"Saved: {comparison_csv}", flush=True)
        print(f"Saved: {comparison_json}", flush=True)
        print(f"Saved: {strength_csv}", flush=True)


if __name__ == "__main__":
    main()
