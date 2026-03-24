from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import load_structured_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown comparison table from evaluation results.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    # Đọc bảng CSV so sánh đã được evaluate-v1.py ghi ra.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    # Parse số an toàn để script không hỏng vì ô rỗng hoặc dữ liệu lỗi.
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def render_summary_table(rows: list[dict[str, str]]) -> list[str]:
    # Dựng bảng tổng hợp điểm chung cho toàn bộ hệ.
    lines = [
        "# So sánh 3 RAG",
        "",
        "| Hạng | Hệ thống | Overall | Answer | Retrieval | Faithfulness | MRR | Semantic | Latency (ms) | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {system} | {overall:.4f} | {answer:.4f} | {retrieval:.4f} | {faithfulness:.4f} | {mrr:.4f} | {semantic:.4f} | {latency:.2f} | {errors} |".format(
                rank=rank,
                system=row["system"],
                overall=to_float(row, "overall_score"),
                answer=to_float(row, "answer_quality"),
                retrieval=to_float(row, "retrieval_quality"),
                faithfulness=to_float(row, "faithfulness"),
                mrr=to_float(row, "mrr"),
                semantic=to_float(row, "semantic_similarity"),
                latency=to_float(row, "latency_ms"),
                errors=row.get("errors", "0"),
            )
        )
    return lines


def render_strength_tables(rows: list[dict[str, str]]) -> list[str]:
    # Dựng thêm các bảng con theo từng nhóm thế mạnh để dễ đọc độ công bằng của benchmark.
    if not rows:
        return []

    bucket_labels = {
        "dense_shared": "Dense Shared",
        "graph": "Graph",
        "neutral": "Neutral",
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("strength_bucket", "neutral"), []).append(row)

    lines = ["", "## Breakdown Theo Nhóm Câu Hỏi", ""]
    for bucket in ("dense_shared", "graph", "neutral"):
        bucket_rows = grouped.get(bucket, [])
        if not bucket_rows:
            continue
        bucket_rows.sort(key=lambda row: to_float(row, "overall_score"), reverse=True)
        lines.extend(
            [
                f"### {bucket_labels.get(bucket, bucket)}",
                "",
                "| Hệ thống | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bucket_rows:
            lines.append(
                "| {system} | {samples} | {overall:.4f} | {recall:.4f} | {faithfulness:.4f} | {answer:.4f} | {context_precision:.4f} |".format(
                    system=row["system"],
                    samples=row.get("samples", "0"),
                    overall=to_float(row, "overall_score"),
                    recall=to_float(row, "recall_at_k"),
                    faithfulness=to_float(row, "faithfulness"),
                    answer=to_float(row, "answer_relevance"),
                    context_precision=to_float(row, "context_precision"),
                )
            )
        lines.append("")
    return lines


def main() -> None:
    # 1) Nạp config V1 và tìm đến file summary đã tổng hợp theo từng hệ.
    args = parse_args()
    config = load_structured_config(args.config)
    comparison_path = resolve_path(Path(config["results_dir"]) / "comparison.csv")
    strength_path = resolve_path(Path(config["results_dir"]) / "strength_breakdown.csv")
    rows = load_rows(comparison_path)
    strength_rows = load_rows(strength_path) if strength_path.exists() else []
    # 2) Sắp xếp hệ theo overall_score giảm dần để làm bảng xếp hạng.
    rows.sort(key=lambda row: to_float(row, "overall_score"), reverse=True)

    # 3) Dựng report Markdown gồm bảng tổng và breakdown theo bucket.
    lines = render_summary_table(rows)
    lines.extend(render_strength_tables(strength_rows))

    # 4) Ghi report Markdown ra thư mục results và đồng thời in ra màn hình.
    report_path = resolve_path(Path(config["results_dir"]) / "comparison.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
