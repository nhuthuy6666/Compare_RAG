from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import ensure_dir, load_structured_config, resolve_path  # noqa: E402
from evaluation.dataset.loader import load_examples  # noqa: E402
from evaluation.policy import load_benchmark_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Khai báo và parse tham số cho workflow human judgment."""

    parser = argparse.ArgumentParser(description="Prepare or report human judgment workflow.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument("command", choices=("prepare", "report"), help="Action to perform.")
    return parser.parse_args()


def _annotation_dir(policy: dict) -> Path:
    """Resolve thư mục annotation từ policy và đảm bảo nó tồn tại."""

    human = policy.get("human_judgment") or {}
    return ensure_dir(resolve_path(human.get("annotation_dir") or "evaluation/annotations/human_v1"))


def _load_output_predictions(path: Path) -> dict[str, dict]:
    """Nạp file outputs JSON và index prediction theo `example_id`."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions") or []
    return {str(item["example_id"]): item for item in predictions}


def _source_preview(prediction: dict) -> str:
    """Tạo preview ngắn của tối đa 3 nguồn đầu để người chấm đọc nhanh."""

    sources = prediction.get("sources") or []
    previews: list[str] = []
    for index, source in enumerate(sources[:3], start=1):
        content = str(source.get("content") or "").replace("\n", " ").strip()
        previews.append(f"[{index}] {content[:280]}")
    return "\n".join(previews)


def prepare_packets(config: dict, policy: dict) -> None:
    """Chuẩn bị packet CSV và template CSV cho hai annotator."""

    annotation_dir = _annotation_dir(policy)
    examples = load_examples(config["dataset_path"], split="held_out_test")
    example_map = {example.id: example for example in examples}
    outputs_dir = resolve_path(config["outputs_dir"])

    packet_rows: list[dict[str, str]] = []
    for system in ("baseline", "hybrid", "graphrag"):
        output_path = outputs_dir / f"{system}_outputs.json"
        if not output_path.exists():
            continue
        predictions = _load_output_predictions(output_path)
        for example_id, example in example_map.items():
            prediction = predictions.get(example_id)
            if not prediction:
                continue
            packet_rows.append(
                {
                    "example_id": example_id,
                    "system": system,
                    "question": example.question,
                    "reference_answer": example.reference_answer,
                    "model_answer": str(prediction.get("answer") or "").strip(),
                    "source_preview": _source_preview(prediction),
                }
            )

    packet_path = annotation_dir / "annotation_packet.csv"
    with packet_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["example_id", "system", "question", "reference_answer", "model_answer", "source_preview"],
        )
        writer.writeheader()
        writer.writerows(packet_rows)

    template_headers = ["example_id", "system", "answer_correctness", "evidence_relevance", "notes"]
    for annotator in ("annotator_a.csv", "annotator_b.csv"):
        template_path = annotation_dir / annotator
        with template_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=template_headers)
            writer.writeheader()
            for row in packet_rows:
                writer.writerow(
                    {
                        "example_id": row["example_id"],
                        "system": row["system"],
                        "answer_correctness": "",
                        "evidence_relevance": "",
                        "notes": "",
                    }
                )

    print(f"Prepared annotation packet: {packet_path}")
    print(f"Prepared annotator templates in: {annotation_dir}")


def _load_annotations(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Đọc file annotation và index theo cặp `(example_id, system)`."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(str(row.get("example_id") or ""), str(row.get("system") or "")): row for row in rows}


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Tính Cohen's kappa cho hai danh sách nhãn có cùng độ dài."""

    if not labels_a or not labels_b or len(labels_a) != len(labels_b):
        return 0.0
    observed = sum(1 for left, right in zip(labels_a, labels_b) if left == right) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b))
        for label in set(counts_a) | set(counts_b)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def _emit_console(text: str) -> None:
    """In console an toàn kể cả khi terminal không hỗ trợ Unicode đầy đủ."""

    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def report_judgments(config: dict, policy: dict) -> None:
    """Tổng hợp annotation của hai người chấm và xuất báo cáo agreement."""

    annotation_dir = _annotation_dir(policy)
    annotator_a = _load_annotations(annotation_dir / "annotator_a.csv")
    annotator_b = _load_annotations(annotation_dir / "annotator_b.csv")
    shared_keys = sorted(set(annotator_a) & set(annotator_b))

    lines = [
        "# Human Judgment Report",
        "",
        f"- Status: `{(policy.get('human_judgment') or {}).get('status', 'pending')}`",
        f"- Required Annotators: `{int((policy.get('human_judgment') or {}).get('required_annotators', 2))}`",
        f"- Shared Rows: `{len(shared_keys)}`",
    ]

    field_specs = [
        ("answer_correctness", "Answer Correctness"),
        ("evidence_relevance", "Evidence Relevance"),
    ]
    report_rows: list[dict[str, object]] = []
    any_completed = False
    for field_name, label in field_specs:
        labels_a: list[str] = []
        labels_b: list[str] = []
        for key in shared_keys:
            left = str((annotator_a.get(key) or {}).get(field_name) or "").strip()
            right = str((annotator_b.get(key) or {}).get(field_name) or "").strip()
            if not left or not right:
                continue
            labels_a.append(left)
            labels_b.append(right)
        if labels_a and labels_b:
            any_completed = True
            kappa = _cohen_kappa(labels_a, labels_b)
            agreement = sum(1 for left, right in zip(labels_a, labels_b) if left == right) / len(labels_a)
            report_rows.append(
                {
                    "field": field_name,
                    "shared_labeled_rows": len(labels_a),
                    "observed_agreement": round(agreement, 4),
                    "cohen_kappa": round(kappa, 4),
                }
            )
            lines.append(f"- {label}: agreement={agreement:.4f}, kappa={kappa:.4f}, rows={len(labels_a)}")
        else:
            lines.append(f"- {label}: pending annotations")

    if not any_completed:
        lines.append("")
        lines.append("Chưa có đủ hai bộ nhãn hoàn chỉnh để tính inter-annotator agreement.")

    report_path = annotation_dir / "judgment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if report_rows:
        with (annotation_dir / "judgment_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
    _emit_console("\n".join(lines))


def main() -> None:
    """Điểm vào chính của workflow human judgment.

    Các bước:
    1. Đọc config và policy để xác định thư mục annotation.
    2. Nếu chọn `prepare`, sinh packet và template cho annotator.
    3. Nếu chọn `report`, đọc annotation hiện có và tính agreement.
    """

    args = parse_args()
    config = load_structured_config(args.config)
    policy = load_benchmark_policy(config)
    if args.command == "prepare":
        prepare_packets(config, policy)
        return
    report_judgments(config, policy)


if __name__ == "__main__":
    main()
