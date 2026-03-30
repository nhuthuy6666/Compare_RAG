from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "evaluation/config_v1.yaml"
DEFAULT_WITH_FUSION_MODE = "controlled_with_fusion"
DEFAULT_NO_FUSION_MODE = "controlled_no_fusion"
DEFAULT_WITH_FUSION_RESULTS = "evaluation/results_shared/with_fusion"
DEFAULT_NO_FUSION_RESULTS = "evaluation/results_shared/no_fusion"
DEFAULT_WITH_FUSION_OUTPUTS = "evaluation/outputs_shared/with_fusion"
DEFAULT_NO_FUSION_OUTPUTS = "evaluation/outputs_shared/no_fusion"


# Khai báo và parse tham số CLI cho script chạy 2 báo cáo shared profile.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared-mode comparisons for all 3 systems with fusion and without fusion."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to config file.")
    parser.add_argument("--split", choices=("all", "dev", "held_out_test"), default="all", help="Dataset split to use.")
    parser.add_argument("--seed", type=int, default=0, help="Recorded seed for repeat runs.")
    parser.add_argument("--with-fusion-mode", default=DEFAULT_WITH_FUSION_MODE, help="Benchmark mode for the fusion run.")
    parser.add_argument("--no-fusion-mode", default=DEFAULT_NO_FUSION_MODE, help="Benchmark mode for the no-fusion run.")
    parser.add_argument("--with-fusion-results-dir", default=DEFAULT_WITH_FUSION_RESULTS, help="Results dir for the fusion run.")
    parser.add_argument("--no-fusion-results-dir", default=DEFAULT_NO_FUSION_RESULTS, help="Results dir for the no-fusion run.")
    parser.add_argument("--with-fusion-outputs-dir", default=DEFAULT_WITH_FUSION_OUTPUTS, help="Outputs dir for the fusion run when --keep-artifacts is enabled.")
    parser.add_argument("--no-fusion-outputs-dir", default=DEFAULT_NO_FUSION_OUTPUTS, help="Outputs dir for the no-fusion run when --keep-artifacts is enabled.")
    parser.add_argument("--with-fusion-run-label", default="", help="Optional run label for the fusion run.")
    parser.add_argument("--no-fusion-run-label", default="", help="Optional run label for the no-fusion run.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep intermediate CSV/JSON artifacts for both runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    return parser.parse_args()


# Dựng lệnh `evaluate-v1.py` hoàn chỉnh cho một lượt chạy fusion hoặc no-fusion.
def build_command(
    *,
    config: str,
    split: str,
    seed: int,
    mode: str,
    results_dir: str,
    outputs_dir: str,
    run_label: str,
    keep_artifacts: bool,
) -> list[str]:
    command = [
        sys.executable,
        "evaluation/evaluate-v1.py",
        "--config",
        config,
        "--system",
        "all",
        "--mode",
        mode,
        "--split",
        split,
        "--seed",
        str(seed),
        "--results-dir",
        results_dir,
    ]
    if run_label:
        command.extend(["--run-label", run_label])
    if keep_artifacts:
        command.extend(["--keep-artifacts", "--outputs-dir", outputs_dir])
    return command


# In lệnh ra console và thực thi nếu không ở chế độ dry-run.
def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = subprocess.list2cmdline(command)
    print(printable, flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


# Entry point của script chạy 2 báo cáo shared profile.
# 1. Đọc CLI và dựng nhãn run mặc định cho 2 chế độ.
# 2. Tạo command cho lượt chạy `with_fusion` và `no_fusion`.
# 3. Chạy tuần tự hai lệnh benchmark và in đường dẫn báo cáo cuối cùng.
def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    with_fusion_label = args.with_fusion_run_label or f"shared_fusion_{args.split}"
    no_fusion_label = args.no_fusion_run_label or f"shared_no_fusion_{args.split}"

    with_fusion_command = build_command(
        config=args.config,
        split=args.split,
        seed=args.seed,
        mode=args.with_fusion_mode,
        results_dir=args.with_fusion_results_dir,
        outputs_dir=args.with_fusion_outputs_dir,
        run_label=with_fusion_label,
        keep_artifacts=args.keep_artifacts,
    )
    no_fusion_command = build_command(
        config=args.config,
        split=args.split,
        seed=args.seed,
        mode=args.no_fusion_mode,
        results_dir=args.no_fusion_results_dir,
        outputs_dir=args.no_fusion_outputs_dir,
        run_label=no_fusion_label,
        keep_artifacts=args.keep_artifacts,
    )

    print("Fusion comparison run:", flush=True)
    run_command(with_fusion_command, dry_run=args.dry_run)
    print("\nNo-fusion comparison run:", flush=True)
    run_command(no_fusion_command, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\nSaved: {project_root / args.with_fusion_results_dir / 'comparison.md'}", flush=True)
        print(f"Saved: {project_root / args.no_fusion_results_dir / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
