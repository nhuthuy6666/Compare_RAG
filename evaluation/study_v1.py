from __future__ import annotations

import argparse
import csv
import random
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import ensure_dir, load_structured_config, write_csv  # noqa: E402
from evaluation.policy import load_benchmark_policy, resolve_mode, resolve_split  # noqa: E402


RNG = random.Random(42)


# Khai báo và parse tham số CLI cho repeated evaluation study.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated evaluation study with CI and significance tests.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument(
        "--mode",
        choices=("controlled", "controlled_with_fusion", "controlled_no_fusion", "best_tuned"),
        default=None,
        help="Benchmark mode.",
    )
    parser.add_argument("--split", choices=("dev", "held_out_test"), default=None, help="Dataset split.")
    parser.add_argument("--runs", type=int, default=None, help="Override number of repeated runs.")
    parser.add_argument("--label", default="", help="Optional study label.")
    return parser.parse_args()


# Đọc CSV thành list dict để tổng hợp study.
def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Parse một giá trị số từ CSV hoặc runtime về float an toàn.
def to_float(value: str | float | int | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


# Tính mean và khoảng tin cậy 95% bằng bootstrap trên nhiều lần chạy.
def bootstrap_ci(values: list[float], rounds: int = 2000) -> tuple[float, float, float]:
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


# Tính p-value pairwise bằng permutation test trên cùng tập mẫu.
def permutation_paired_pvalue(left: list[float], right: list[float], rounds: int = 5000) -> float:
    if not left or not right or len(left) != len(right):
        return 1.0
    diffs = [l - r for l, r in zip(left, right)]
    observed = abs(statistics.fmean(diffs))
    extreme = 0
    for _ in range(rounds):
        signed = [diff if RNG.random() >= 0.5 else -diff for diff in diffs]
        if abs(statistics.fmean(signed)) >= observed:
            extreme += 1
    return (extreme + 1) / (rounds + 1)


# Gắn nhãn ổn định khi CI không cắt 0 và p-value đủ nhỏ.
def confidence_label(ci_low: float, ci_high: float, p_value: float) -> str:
    if (ci_low > 0 or ci_high < 0) and p_value < 0.05:
        return "stable"
    return "uncertain"


# Entry point của repeated study.
# 1. Đọc config/policy rồi resolve mode, split, số lần chạy và seed schedule.
# 2. Gọi `evaluate-v1.py` lặp lại nhiều lần và lưu mỗi run vào thư mục riêng.
# 3. Tổng hợp chuỗi điểm theo hệ, tính CI và pairwise significance.
# 4. Ghi CSV và report Markdown cho phân tích độ ổn định.
def main() -> None:
    args = parse_args()
    config = load_structured_config(args.config)
    policy = load_benchmark_policy(config)
    mode = resolve_mode(policy, args.mode)
    split = resolve_split(policy, args.split)
    repeats = policy.get("repeated_runs") or {}
    runs = int(args.runs or repeats.get("num_runs") or 3)
    seeds = list(repeats.get("seed_values") or [13, 29, 47])
    if len(seeds) < runs:
        seeds.extend(seeds[-1:] * (runs - len(seeds)))

    study_label = args.label or f"{mode}_{split}"
    study_root = ensure_dir(Path("evaluation") / "studies_v1" / study_label)
    run_rows: list[dict[str, object]] = []
    series_by_system: dict[str, list[float]] = {system: [] for system in ("baseline", "hybrid", "graphrag")}
    paired_by_system: dict[str, list[float]] = {system: [] for system in ("baseline", "hybrid", "graphrag")}

    for run_index in range(1, runs + 1):
        seed = int(seeds[run_index - 1])
        run_root = ensure_dir(study_root / f"run_{run_index:02d}")
        outputs_dir = run_root / "outputs"
        results_dir = run_root / "results"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "evaluate-v1.py"),
            "--config",
            args.config,
            "--mode",
            mode,
            "--split",
            split,
            "--seed",
            str(seed),
            "--run-label",
            f"{study_label}_run_{run_index:02d}",
            "--outputs-dir",
            str(outputs_dir),
            "--results-dir",
            str(results_dir),
            "--keep-artifacts",
        ]
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)

        comparison_rows = load_csv_rows(results_dir / "comparison.csv")
        for row in comparison_rows:
            system = row["system"]
            score = to_float(row["overall_score"])
            run_rows.append(
                {
                    "run_index": run_index,
                    "seed": seed,
                    "system": system,
                    "mode": mode,
                    "split": split,
                    "overall_score": round(score, 4),
                }
            )
            series_by_system[system].append(score)

        for system in paired_by_system:
            metric_rows = load_csv_rows(results_dir / f"{system}_metrics.csv")
            metric_rows.sort(key=lambda row: row["example_id"])
            paired_by_system[system].extend(to_float(row["overall_score"]) for row in metric_rows)

    summary_rows: list[dict[str, object]] = []
    for system, values in series_by_system.items():
        mean_value, ci_low, ci_high = bootstrap_ci(values)
        std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "system": system,
                "mode": mode,
                "split": split,
                "runs": runs,
                "mean_overall": round(mean_value, 4),
                "std_overall": round(std_value, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
            }
        )

    pairwise_rows: list[dict[str, object]] = []
    for left, right in [("baseline", "hybrid"), ("baseline", "graphrag"), ("hybrid", "graphrag")]:
        diffs = [l - r for l, r in zip(paired_by_system[left], paired_by_system[right])]
        diff_mean, ci_low, ci_high = bootstrap_ci(diffs)
        p_value = permutation_paired_pvalue(paired_by_system[left], paired_by_system[right])
        pairwise_rows.append(
            {
                "left_system": left,
                "right_system": right,
                "mode": mode,
                "split": split,
                "mean_diff": round(diff_mean, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "p_value": round(p_value, 4),
                "stability": confidence_label(ci_low, ci_high, p_value),
            }
        )

    write_csv(study_root / "study_runs.csv", run_rows)
    write_csv(study_root / "study_summary.csv", summary_rows)
    write_csv(study_root / "study_pairwise.csv", pairwise_rows)

    lines = [
        "# Repeated Evaluation Study",
        "",
        f"- Mode: `{mode}`",
        f"- Split: `{split}`",
        f"- Runs: `{runs}`",
        f"- Seeds: `{', '.join(str(int(seeds[index])) for index in range(runs))}`",
        f"- Seed Effective: `{str(bool((policy.get('repeated_runs') or {}).get('seed_effective', False))).lower()}`",
        "",
        "## Summary",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['system']}: mean={row['mean_overall']} std={row['std_overall']} "
            f"95% CI=[{row['ci_low']}, {row['ci_high']}]"
        )
    lines.append("")
    lines.append("## Pairwise Significance")
    for row in pairwise_rows:
        lines.append(
            f"- {row['left_system']} - {row['right_system']}: diff={row['mean_diff']} "
            f"95% CI=[{row['ci_low']}, {row['ci_high']}], p={row['p_value']}, {row['stability']}"
        )

    report_path = study_root / "study.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
