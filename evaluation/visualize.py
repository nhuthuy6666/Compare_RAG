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
    parser = argparse.ArgumentParser(description="Render ASCII visualization from evaluation results.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    # Đọc summary CSV để render bản xem ASCII.
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bar(value: float, width: int = 24) -> str:
    # Chuyển một điểm 0-1 thành thanh ASCII đơn giản để nhìn tương quan nhanh.
    value = max(0.0, min(1.0, value))
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def to_float(row: dict[str, str], key: str) -> float:
    # Parse số an toàn từ CSV.
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def main() -> None:
    # 1) Đọc file comparison.csv đã được evaluate-v1.py tạo ra.
    args = parse_args()
    config = load_structured_config(args.config)
    comparison_path = resolve_path(Path(config["results_dir"]) / "comparison.csv")
    rows = load_rows(comparison_path)
    # 2) Sắp xếp hệ theo overall_score để phần hiển thị nhất quán với bảng xếp hạng.
    rows.sort(key=lambda row: to_float(row, "overall_score"), reverse=True)

    # 3) Tạo các dòng ASCII bar cho từng hệ và một vài metric quan trọng.
    lines = ["ASCII view"]
    for row in rows:
        overall = to_float(row, "overall_score")
        answer = to_float(row, "answer_quality")
        retrieval = to_float(row, "retrieval_quality")
        faithfulness = to_float(row, "faithfulness")
        lines.append(f"{row['system']:<10} overall   {bar(overall)} {overall:.4f}")
        lines.append(f"{'':<10} answer    {bar(answer)} {answer:.4f}")
        lines.append(f"{'':<10} retrieval {bar(retrieval)} {retrieval:.4f}")
        lines.append(f"{'':<10} faithful  {bar(faithfulness)} {faithfulness:.4f}")
        lines.append("")

    # 4) Ghi file text để có thể mở nhanh mà không cần Markdown renderer.
    output_path = resolve_path(Path(config["results_dir"]) / "comparison_ascii.txt")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
