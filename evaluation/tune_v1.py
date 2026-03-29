from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.common import ensure_dir, load_structured_config  # noqa: E402
from evaluation.policy import load_benchmark_policy, load_locked_profiles, load_profile_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Khai báo và parse tham số cho quy trình tuning khóa profile trên dev."""

    parser = argparse.ArgumentParser(description="Run the locked tuning protocol on dev split only.")
    parser.add_argument("--config", default="evaluation/config_v1.yaml", help="Path to config file.")
    parser.add_argument("--mode", choices=("best_tuned",), default="best_tuned", help="Profile set to evaluate on dev.")
    parser.add_argument("--label", default="protocol_dev", help="Output label.")
    parser.add_argument("--write-locks", action="store_true", help="Persist selected profiles back to locked manifest.")
    return parser.parse_args()


def _comparison_path(results_dir: Path) -> Path:
    """Trả về đường dẫn file summary JSON của một run candidate."""

    return results_dir / "comparison.json"


def _candidate_names(profile_candidates: dict, system_name: str) -> list[str]:
    """Lấy danh sách tên candidate profile của một hệ trong mode best_tuned."""

    systems = ((((profile_candidates.get("best_tuned") or {}).get("systems")) or {}))
    candidates = (((systems.get(system_name) or {}).get("candidates")) or {})
    return list(candidates.keys())


def _load_summary(path: Path) -> dict:
    """Nạp summary JSON của một run candidate và chuẩn hóa về dict."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return dict(payload[0])
    raise RuntimeError(f"Unexpected comparison payload: {path}")


def main() -> None:
    """Điểm vào chính của tuning protocol.

    Các bước:
    1. Đọc config, policy và kiểm tra ràng buộc tuning chỉ được chạy trên dev.
    2. Lần lượt chạy các candidate profile cho từng hệ với nhiều seed.
    3. Tính điểm trung bình, chọn candidate tốt nhất và cập nhật manifest locked nếu được yêu cầu.
    4. Ghi báo cáo tóm tắt toàn bộ quy trình tuning.
    """

    args = parse_args()
    config = load_structured_config(args.config)
    policy = load_benchmark_policy(config)
    locked_profiles = load_locked_profiles(policy)
    profile_candidates = load_profile_candidates(policy)
    tuning = policy.get("tuning_protocol") or {}
    split = str(tuning.get("split") or "dev")
    if split != "dev":
        raise RuntimeError("Tuning protocol must point to dev split only.")
    if bool(tuning.get("allow_test_tuning", False)):
        raise RuntimeError("Policy violation: allow_test_tuning must remain false.")

    max_candidates = int(tuning.get("max_candidates_per_system", 3))
    repeats = int(tuning.get("repeats_per_candidate", 3))
    seeds = list(tuning.get("seed_values") or [13, 29, 47])[:repeats]

    output_root = ensure_dir(Path("evaluation") / "tuning_v1" / args.label)
    system_reports: list[dict[str, object]] = []

    for system_name in ("baseline", "hybrid", "graphrag"):
        candidate_names = _candidate_names(profile_candidates, system_name)[:max_candidates]
        if not candidate_names:
            continue

        candidate_reports: list[dict[str, object]] = []
        for candidate_name in candidate_names:
            run_scores: list[float] = []
            for run_index, seed in enumerate(seeds, start=1):
                candidate_root = ensure_dir(output_root / system_name / candidate_name / f"run_{run_index:02d}")
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "evaluation" / "evaluate-v1.py"),
                    "--config",
                    args.config,
                    "--system",
                    system_name,
                    "--mode",
                    args.mode,
                    "--split",
                    split,
                    "--profile-source",
                    "candidate",
                    "--profile-name",
                    candidate_name,
                    "--seed",
                    str(seed),
                    "--run-label",
                    f"{args.label}_{system_name}_{candidate_name}_run{run_index}",
                    "--outputs-dir",
                    str(candidate_root / "outputs"),
                    "--results-dir",
                    str(candidate_root / "results"),
                    "--keep-artifacts",
                ]
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
                summary = _load_summary(_comparison_path(candidate_root / "results"))
                run_scores.append(float(summary.get("overall_score") or 0.0))

            mean_score = statistics.fmean(run_scores) if run_scores else 0.0
            candidate_reports.append(
                {
                    "candidate_name": candidate_name,
                    "mean_overall_score": round(mean_score, 4),
                    "scores": [round(score, 4) for score in run_scores],
                }
            )

        candidate_reports.sort(key=lambda item: (-float(item["mean_overall_score"]), str(item["candidate_name"])))
        best = candidate_reports[0]
        selected_payload = (
            ((((profile_candidates.get("best_tuned") or {}).get("systems")) or {}).get(system_name) or {})
            .get("candidates", {})
            .get(str(best["candidate_name"]), {})
        )
        locked_profiles["best_tuned"]["systems"][system_name] = {
            "profile_name": str(best["candidate_name"]),
            "locked": True,
            "runtime_overrides": dict(selected_payload.get("runtime_overrides") or {}),
            "selection_note": (
                f"Selected on dev via tune_v1 with mean overall_score={float(best['mean_overall_score']):.4f}; "
                f"scores={best['scores']}"
            ),
        }
        system_reports.append(
            {
                "system": system_name,
                "selected_profile": best["candidate_name"],
                "mean_overall_score": best["mean_overall_score"],
                "candidates": candidate_reports,
            }
        )

    if args.write_locks:
        locked_path = Path(policy.get("locked_profiles_path") or "evaluation/locked_profiles_v1.json")
        locked_path.write_text(json.dumps(locked_profiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Tuning Protocol Report",
        "",
        f"- Mode: `{args.mode}`",
        f"- Split: `{split}`",
        f"- Max Candidates Per System: `{max_candidates}`",
        f"- Repeats Per Candidate: `{repeats}`",
        f"- Seed Values: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Locked Profiles Snapshot: `{Path(policy.get('locked_profiles_path') or 'evaluation/locked_profiles_v1.json')}`",
        "",
        "## Selection",
    ]
    for report in system_reports:
        lines.append(
            f"- {report['system']}: `{report['selected_profile']}` (mean overall={float(report['mean_overall_score']):.4f})"
        )
    lines.extend(
        [
            "",
            "## Notes",
            f"- {str(tuning.get('note') or '').strip()}",
            "- Repo hiện đã khóa quy trình: tuning chỉ được chạy trên dev split.",
            "- Kết quả held_out_test chỉ nên chạy sau khi profile đã khóa xong.",
            (
                "- Locked profile manifest đã được cập nhật."
                if args.write_locks
                else "- Chưa ghi đè locked profile manifest. Dùng --write-locks để khóa profile vào manifest."
            ),
        ]
    )
    report_path = output_root / "tuning_protocol.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
