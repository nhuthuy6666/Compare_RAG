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


# Khai báo và parse tham số CLI cho báo cáo reliability.
def parse_args() -> argparse.Namespace:
    """Khai báo và parse tham số cho báo cáo reliability."""

    parser = argparse.ArgumentParser(description="Compute bootstrap reliability reports from evaluation results.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument(
        "--split",
        choices=("auto", "all", "dev", "held_out_test"),
        default="auto",
        help="Dataset split to align against. Default auto-reads from comparison.json when available.",
    )
    return parser.parse_args()


# Đọc một file CSV thành list dict để các bước bootstrap dùng chung.
def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Đọc CSV thành danh sách dict để xử lý thống nhất."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Liệt kê các file metric CSV còn thiếu trước khi bắt đầu bootstrap.
def find_missing_metric_files(results_dir: Path, systems: list[str]) -> list[Path]:
    """Liệt kê các file metric CSV còn thiếu cho từng hệ."""

    missing: list[Path] = []
    for system in systems:
        metric_path = results_dir / f"{system}_metrics.csv"
        if not metric_path.exists():
            missing.append(metric_path)
    return missing


# Resolve split dùng cho reliability; `auto` sẽ cố suy ra từ `comparison.json`.
def resolve_reliability_split(results_dir: Path, requested_split: str) -> str:
    """Tự suy ra split cần dùng khi người dùng chọn `auto`."""

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


# Căn metric rows theo thứ tự example chuẩn để so sánh pairwise không lệch mẫu.
def align_rows_to_examples(system: str, rows: list[dict[str, str]], example_ids: list[str]) -> list[dict[str, str]]:
    """Căn metric row theo đúng thứ tự example để bootstrap pairwise không lệch mẫu."""

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


# Parse một giá trị số lấy từ CSV về float an toàn.
def to_float(value: str | float | int | None) -> float:
    """Parse số an toàn cho dữ liệu đọc từ CSV."""

    try:
        return float(value or 0)
    except ValueError:
        return 0.0


# Tính mean và khoảng tin cậy 95% bằng bootstrap cho một dãy điểm.
def bootstrap_ci(values: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    """Ước lượng mean và khoảng tin cậy 95% cho một metric."""

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


# Tính khoảng tin cậy bootstrap cho chênh lệch giữa hai hệ đã align theo example.
def bootstrap_diff_ci(left: list[float], right: list[float], rounds: int = 2000) -> tuple[float, float, float]:
    """Ước lượng khoảng tin cậy cho chênh lệch trung bình giữa hai hệ."""

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


# Gắn nhãn ổn định khi khoảng tin cậy của chênh lệch không cắt qua 0.
def confidence_label(low: float, high: float) -> str:
    """Gán nhãn ổn định nếu khoảng tin cậy không cắt qua 0."""

    if low > 0 or high < 0:
        return "stable"
    return "overlap_zero"


# Entry point của báo cáo reliability.
# 1. Đọc config, resolve split và nạp dataset làm chuẩn align.
# 2. Kiểm tra các metric CSV cần thiết đã tồn tại hay chưa.
# 3. Căn metric row của từng hệ theo example rồi tính bootstrap CI và pairwise diff.
# 4. Ghi CSV/report Markdown để phục vụ phân tích độ tin cậy.
def main() -> None:
    """Điểm vào chính của script reliability.

    Các bước:
    1. Đọc config, xác định split và nạp metric CSV của từng hệ.
    2. Căn metric row theo thứ tự câu hỏi để tránh lệch mẫu khi bootstrap.
    3. Tính CI cho metric tổng quan, breakdown theo topic và pairwise difference.
    4. Ghi các bảng CSV và report Markdown để phục vụ phân tích độ tin cậy.
    """

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
        print("Nếu bạn chỉ cần comparison.md thì không cần chạy lại.")
        print("Không thể chạy reliability.py vì các file metric CSV không còn trong evaluation/results_v1.")
        print("Mặc định evaluate-v1.py hiện chỉ giữ comparison.md để folder kết quả gọn hơn.")
        print("Nếu cần reliability report, hãy chạy lại evaluate-v1.py kèm --keep-artifacts rồi chạy lại reliability.py.")
        print("Các file đang thiếu:")
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

    topic_report_rows: list[dict[str, object]] = []
    for system, rows in rows_by_system.items():
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            grouped[row["topic"]].append(to_float(row["overall_score"]))
        for topic, expected_n in sorted(topic_counts.items()):
            vals = grouped.get(topic, [])
            mean_value = statistics.fmean(vals) if vals else 0.0
            topic_report_rows.append(
                {
                    "system": system,
                    "topic": topic,
                    "samples": len(vals),
                    "expected_samples": expected_n,
                    "overall_mean": round(mean_value, 4),
                }
            )

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

    warning_topics = [topic for topic, count in topic_counts.items() if count < 3]
    lines = [
        "# Reliability Report",
        "",
        f"- Split: {resolved_split}",
        f"- Tổng số câu hỏi: {len(examples)}",
        f"- Số nhóm chủ đề: {len(topic_counts)}",
        f"- Nhóm có ít hơn 3 câu: {', '.join(warning_topics) if warning_topics else 'không có'}",
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
        lines.append(f"- {topic}: {count} câu")

    report_path = results_dir / "reliability.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
