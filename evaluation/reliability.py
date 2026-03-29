from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_structured_config, resolve_path  # noqa: E402
from evaluation.dataset.loader import load_examples  # noqa: E402


RNG = random.Random(42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap reliability reports from evaluation results.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument(
        "--split",
        choices=("auto", "all", "dev", "held_out_test"),
        default="auto",
        help="Dataset split to align against. Default auto-reads from comparison.json when available.",
    )
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_missing_metric_files(results_dir: Path, systems: list[str]) -> list[Path]:
    missing: list[Path] = []
    for system in systems:
        metric_path = results_dir / f"{system}_metrics.csv"
        if not metric_path.exists():
            missing.append(metric_path)
    return missing


def resolve_reliability_split(results_dir: Path, requested_split: str) -> str:
    if requested_split != "auto":
        return requested_split

    comparison_path = results_dir / "comparison.json"
    if not comparison_path.exists():
        return "all"

    try:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "all"

    if not isinstance(payload, list) or not payload:
        return "all"

    splits = {str(row.get("split") or "").strip() for row in payload if str(row.get("split") or "").strip()}
    if len(splits) == 1:
        return next(iter(splits))
    return "all"


def align_rows_to_examples(system: str, rows: list[dict[str, str]], example_ids: list[str]) -> list[dict[str, str]]:
    # Căn metric rows theo đúng thứ tự example_id của evaluate-v1 để pairwise bootstrap không lệch mẫu.
    indexed_rows: dict[str, dict[str, str]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"{system}: metric row thiếu example_id nên không thể align reliability.")
        if example_id in indexed_rows:
            duplicate_ids.append(example_id)
            continue
        indexed_rows[example_id] = row

    if duplicate_ids:
        duplicates = ", ".join(sorted(set(duplicate_ids)))
        raise ValueError(f"{system}: metric CSV có example_id bị trùng: {duplicates}")

    expected_ids = set(example_ids)
    actual_ids = set(indexed_rows)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        details: list[str] = []
        if missing_ids:
            details.append(f"missing={', '.join(missing_ids[:5])}")
        if extra_ids:
            details.append(f"extra={', '.join(extra_ids[:5])}")
        suffix = " ..." if len(missing_ids) > 5 or len(extra_ids) > 5 else ""
        raise ValueError(f"{system}: metric CSV không khớp dataset ({'; '.join(details)}{suffix})")

    return [indexed_rows[example_id] for example_id in example_ids]


def to_float(value: str | float | int | None) -> float:
    # Parse số an toàn cho các giá trị đọc từ CSV.
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def bootstrap_ci(values: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    # Bootstrap mean và khoảng tin cậy 95% cho một metric của một hệ.
    if not values:
        return 0.0, 0.0, 0.0
    means: list[float] = []
    n = len(values)
    for _ in range(rounds):
        sample = [values[RNG.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    low = means[max(0, int(0.025 * len(means)) - 1)]
    high = means[min(len(means) - 1, int(0.975 * len(means)) - 1)]
    return statistics.fmean(values), low, high


def bootstrap_diff_ci(left: list[float], right: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    # Bootstrap chênh lệch trung bình giữa hai hệ trên cùng tập câu hỏi.
    if not left or not right or len(left) != len(right):
        return 0.0, 0.0, 0.0
    diffs: list[float] = []
    n = len(left)
    for _ in range(rounds):
        idxs = [RNG.randrange(n) for _ in range(n)]
        diffs.append(statistics.fmean(left[i] - right[i] for i in idxs))
    diffs.sort()
    observed = statistics.fmean(l - r for l, r in zip(left, right))
    low = diffs[max(0, int(0.025 * len(diffs)) - 1)]
    high = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)) - 1)]
    return observed, low, high


def confidence_label(low: float, high: float) -> str:
    # Nếu CI không cắt qua 0 thì xem khác biệt là ổn định hơn.
    if low > 0 or high < 0:
        return "stable"
    return "overlap_zero"


def main() -> None:
    # 1) Nạp config, dataset và metric CSV của từng hệ.
    args = parse_args()
    config = load_structured_config(args.config)
    results_dir = resolve_path(config["results_dir"])
    resolved_split = resolve_reliability_split(results_dir, args.split)
    examples = load_examples(config["dataset_path"], split=resolved_split)
    example_ids = [example.id for example in examples]
    topic_counts = Counter(example.topic for example in examples)

    systems = ["baseline", "hybrid", "graphrag"]
    missing_metric_files = find_missing_metric_files(results_dir, systems)
    if missing_metric_files:
        comparison_md = results_dir / "comparison.md"
        if comparison_md.exists():
            print(f"Saved: {comparison_md}")
        print("Neu ban chi can comparison.md thi khong can chay lai.")
        print("Khong the chay reliability.py vi cac file metric CSV khong con trong evaluation/results_v1.")
        print("Mac dinh evaluate-v1.py hien chi giu comparison.md de folder ket qua gon hon.")
        print("Neu can reliability report, hay chay lai evaluate-v1.py kem --keep-artifacts roi chay lai reliability.py.")
        print("Cac file dang thieu:")
        for path in missing_metric_files:
            print(f"- {path}")
        return

    rows_by_system = {
        system: align_rows_to_examples(
            system,
            load_csv_rows(results_dir / f"{system}_metrics.csv"),
            example_ids,
        )
        for system in systems
    }

    # 2) Tính bootstrap 95% CI cho các metric tổng quan quan trọng của từng hệ.
    summary_rows: list[dict[str, object]] = []
    for system, rows in rows_by_system.items():
        overall_vals = [to_float(row["overall_score"]) for row in rows]
        faith_vals = [to_float(row["faithfulness"]) for row in rows]
        retrieval_vals = [to_float(row["retrieval_quality"]) for row in rows]
        answer_vals = [to_float(row["answer_quality"]) for row in rows]
        overall_mean, overall_low, overall_high = bootstrap_ci(overall_vals)
        faith_mean, faith_low, faith_high = bootstrap_ci(faith_vals)
        retrieval_mean, retrieval_low, retrieval_high = bootstrap_ci(retrieval_vals)
        answer_mean, answer_low, answer_high = bootstrap_ci(answer_vals)
        summary_rows.append(
            {
                "system": system,
                "samples": len(rows),
                "overall_mean": round(overall_mean, 4),
                "overall_ci_low": round(overall_low, 4),
                "overall_ci_high": round(overall_high, 4),
                "answer_mean": round(answer_mean, 4),
                "answer_ci_low": round(answer_low, 4),
                "answer_ci_high": round(answer_high, 4),
                "retrieval_mean": round(retrieval_mean, 4),
                "retrieval_ci_low": round(retrieval_low, 4),
                "retrieval_ci_high": round(retrieval_high, 4),
                "faithfulness_mean": round(faith_mean, 4),
                "faithfulness_ci_low": round(faith_low, 4),
                "faithfulness_ci_high": round(faith_high, 4),
                }
            )

    # 3) Tổng hợp điểm theo topic để nhìn xem hệ mạnh/yếu ở nhóm câu hỏi nào.
    topic_report_rows: list[dict[str, object]] = []
    for system, rows in rows_by_system.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[row["topic"]].append(to_float(row["overall_score"]))
        for topic, expected_n in sorted(topic_counts.items()):
            vals = grouped.get(topic, [])
            mean = statistics.fmean(vals) if vals else 0.0
            topic_report_rows.append(
                {
                    "system": system,
                    "topic": topic,
                    "samples": len(vals),
                    "expected_samples": expected_n,
                    "overall_mean": round(mean, 4),
                }
            )

    # 4) So sánh từng cặp hệ bằng bootstrap diff và win-rate theo từng câu.
    pairwise_rows: list[dict[str, object]] = []
    for left, right in [("baseline", "hybrid"), ("baseline", "graphrag"), ("hybrid", "graphrag")]:
        left_rows = rows_by_system[left]
        right_rows = rows_by_system[right]
        left_scores = [to_float(row["overall_score"]) for row in left_rows]
        right_scores = [to_float(row["overall_score"]) for row in right_rows]
        diff, low, high = bootstrap_diff_ci(left_scores, right_scores)
        win_rate = sum(1 for l, r in zip(left_scores, right_scores) if l > r) / max(1, len(left_scores))
        pairwise_rows.append(
            {
                "left_system": left,
                "right_system": right,
                "mean_diff": round(diff, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "win_rate": round(win_rate, 4),
                "stability": confidence_label(low, high),
            }
        )

    summary_csv = results_dir / "reliability_summary.csv"
    topic_csv = results_dir / "topic_breakdown.csv"
    pairwise_csv = results_dir / "pairwise_differences.csv"

    # 5) Ghi các bảng reliability ra CSV để tiện phân tích tiếp bằng spreadsheet.
    for path, rows in [
        (summary_csv, summary_rows),
        (topic_csv, topic_report_rows),
        (pairwise_csv, pairwise_rows),
    ]:
        headers: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in headers:
                    headers.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    # 6) Sinh báo cáo Markdown tóm tắt CI, pairwise difference và độ phủ chủ đề.
    warning_topics = [topic for topic, count in topic_counts.items() if count < 3]
    lines = [
        "# Reliability Report",
        "",
        f"- Split: {resolved_split}",
        f"- Tong so cau hoi: {len(examples)}",
        f"- So nhom chu de: {len(topic_counts)}",
        f"- Nhom co it hon 3 cau: {', '.join(warning_topics) if warning_topics else 'khong co'}",
        "",
        "## Bootstrap 95% CI",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['system']}: overall={row['overall_mean']} [{row['overall_ci_low']}, {row['overall_ci_high']}], "
            f"retrieval={row['retrieval_mean']} [{row['retrieval_ci_low']}, {row['retrieval_ci_high']}], "
            f"faithfulness={row['faithfulness_mean']} [{row['faithfulness_ci_low']}, {row['faithfulness_ci_high']}]"
        )
    lines.append("")
    lines.append("## Pairwise Differences")
    for row in pairwise_rows:
        lines.append(
            f"- {row['left_system']} - {row['right_system']}: diff={row['mean_diff']} "
            f"[{row['ci_low']}, {row['ci_high']}], win_rate={row['win_rate']}, {row['stability']}"
        )
    lines.append("")
    lines.append("## Topic Coverage")
    for topic, count in sorted(topic_counts.items()):
        lines.append(f"- {topic}: {count} cau")

    # 7) Ghi file báo cáo và in ra terminal.
    report_path = results_dir / "reliability.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
