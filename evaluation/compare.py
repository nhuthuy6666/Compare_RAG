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
    """Khai báo và parse tham số cho script dựng lại báo cáo Markdown."""

    parser = argparse.ArgumentParser(description="Render markdown comparison table from evaluation results.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    """Đọc CSV thành danh sách dict để các hàm render dùng chung."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float:
    """Parse số an toàn từ một cột trong CSV."""

    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def render_metadata(rows: list[dict[str, str]]) -> list[str]:
    """Render phần metadata đầu báo cáo như mode, split, budget và seed."""

    if not rows:
        return ["# So sánh 3 RAG", "", "Chưa có dữ liệu để hiển thị."]

    row = rows[0]
    return [
        "# So sánh 3 RAG",
        "",
        f"- Mode: `{row.get('mode', 'unknown')}`",
        f"- Split: `{row.get('split', 'unknown')}`",
        f"- Run Label: `{row.get('run_label', '') or 'default'}`",
        f"- Seed: `{row.get('seed', '0')}`",
        f"- Token Budget: `{row.get('token_budget_status', 'unknown')}` | max_output_tokens=`{row.get('max_output_tokens', '0')}`",
        f"- Retrieval Budget: top_n=`{row.get('retrieval_budget_top_n', '0')}`, k=`{row.get('evaluation_k', '0')}`",
        f"- Latency Budget: `{row.get('latency_budget_ms', '0')}` ms",
        "",
    ]


def render_summary_table(rows: list[dict[str, str]]) -> list[str]:
    """Render bảng xếp hạng tổng quan của các hệ."""

    lines = [
        "| Hạng | Hệ thống | Profile | Overall | Answer | Retrieval | Faithfulness | MRR | Semantic | Latency (ms) | Budget Violation | Errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {system} | {profile} | {overall:.4f} | {answer:.4f} | {retrieval:.4f} | {faithfulness:.4f} | {mrr:.4f} | {semantic:.4f} | {latency:.2f} | {budget_violation:.4f} | {errors} |".format(
                rank=rank,
                system=row["system"],
                profile=row.get("profile_name", "default"),
                overall=to_float(row, "overall_score"),
                answer=to_float(row, "answer_quality"),
                retrieval=to_float(row, "retrieval_quality"),
                faithfulness=to_float(row, "faithfulness"),
                mrr=to_float(row, "mrr"),
                semantic=to_float(row, "semantic_similarity"),
                latency=to_float(row, "latency_ms"),
                budget_violation=to_float(row, "latency_budget_violation_rate"),
                errors=row.get("errors", "0"),
            )
        )
    return lines


def render_strength_tables(rows: list[dict[str, str]]) -> list[str]:
    """Render breakdown theo từng nhóm câu hỏi nếu có dữ liệu bucket."""

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
                "| Hệ thống | Profile | Samples | Overall | Recall@k | Faithfulness | Answer_Relevancy | Context_Precision |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in bucket_rows:
            lines.append(
                "| {system} | {profile} | {samples} | {overall:.4f} | {recall:.4f} | {faithfulness:.4f} | {answer:.4f} | {context_precision:.4f} |".format(
                    system=row["system"],
                    profile=row.get("profile_name", "default"),
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


def build_comparison_report(rows: list[dict[str, str]], strength_rows: list[dict[str, str]] | None = None) -> str:
    """Ghép toàn bộ các phần thành nội dung cuối của `comparison.md`."""

    sorted_rows = sorted(rows, key=lambda row: to_float(row, "overall_score"), reverse=True)
    lines = render_metadata(sorted_rows)
    lines.extend(render_summary_table(sorted_rows))
    lines.extend(render_strength_tables(strength_rows or []))
    return "\n".join(lines)


def main() -> None:
    """Điểm vào chính của script render báo cáo.

    Các bước:
    1. Đọc config để xác định thư mục kết quả hiện tại.
    2. Kiểm tra xem `comparison.csv` và `strength_breakdown.csv` có tồn tại hay không.
    3. Nếu có dữ liệu, dựng lại `comparison.md` từ CSV.
    4. Nếu không có artifact nhưng đã có `comparison.md`, chỉ in hướng dẫn trạng thái hiện tại.
    """

    args = parse_args()
    config = load_structured_config(args.config)
    comparison_path = resolve_path(Path(config["results_dir"]) / "comparison.csv")
    strength_path = resolve_path(Path(config["results_dir"]) / "strength_breakdown.csv")
    report_path = resolve_path(Path(config["results_dir"]) / "comparison.md")

    if not comparison_path.exists():
        if report_path.exists():
            print(f"Saved: {report_path}")
            print("comparison.csv không tồn tại vì evaluate-v1.py đã được chạy ở chế độ gọn chỉ giữ comparison.md.")
            print("Nếu bạn chỉ cần comparison.md thì không cần chạy lại.")
            print("Nếu cần artifact CSV/JSON để rebuild hoặc chạy thêm script phụ, hãy run evaluate-v1.py kèm --keep-artifacts.")
            return
        print("Chưa tìm thấy comparison.csv hoặc comparison.md.")
        print("Hãy chạy lại evaluate-v1.py. Nếu cần artifact trung gian, thêm --keep-artifacts.")
        return

    rows = load_rows(comparison_path)
    strength_rows = load_rows(strength_path) if strength_path.exists() else []

    report = build_comparison_report(rows, strength_rows)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
