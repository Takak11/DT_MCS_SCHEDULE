import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd, cwd, dry_run=False):
    cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"\n>>> {cmd_str}")
    if dry_run:
        return
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd))
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}) after {dt:.1f}s: {cmd_str}")
    print(f"<<< done in {dt:.1f}s")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run full DT-MCS pipeline: generate dataset -> train -> DAgger -> plot"
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path")
    parser.add_argument("--workdir", type=str, default=".", help="Project working directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")

    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-dagger", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")

    parser.add_argument("--base-dataset", type=str, default="expert_dataset.pkl")
    parser.add_argument("--dagger-dataset", type=str, default="expert_dataset_dagger.pkl")
    parser.add_argument("--init-ckpt", type=str, default="dt_mcs_best.pth")
    parser.add_argument("--log-path", type=str, default="train_log_v2.csv")

    parser.add_argument("--dagger-iters", type=int, default=3)
    parser.add_argument("--dagger-collect-episodes", type=int, default=100)
    parser.add_argument("--dagger-beta-start", type=float, default=0.9)
    parser.add_argument("--dagger-beta-end", type=float, default=0.3)
    parser.add_argument("--dagger-finetune-epochs", type=int, default=30)
    parser.add_argument("--dagger-finetune-aux-reward-weight", type=float, default=0.05)
    parser.add_argument("--dagger-eval-seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--dagger-start-seed", type=int, default=7000)
    parser.add_argument("--dagger-bootstrap-episodes", type=int, default=1000)
    parser.add_argument("--dagger-target-return", type=float, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    cwd = Path(args.workdir).resolve()
    py = str(args.python)

    if not cwd.exists():
        raise FileNotFoundError(f"workdir not found: {cwd}")

    print("Pipeline order: generate_DT_dataset -> train -> DAgger -> plot")
    print(f"Working directory: {cwd}")
    print(f"Python: {py}")

    if not args.skip_generate:
        _run([py, "generate_DT_dataset.py"], cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip generate stage")

    if not args.skip_train:
        _run([py, "train.py"], cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip train stage")

    if not args.skip_dagger:
        dagger_cmd = [
            py,
            "dagger_train.py",
            "--base-dataset",
            args.base_dataset,
            "--dagger-dataset",
            args.dagger_dataset,
            "--init-ckpt",
            args.init_ckpt,
            "--log-path",
            args.log_path,
            "--iters",
            str(args.dagger_iters),
            "--collect-episodes",
            str(args.dagger_collect_episodes),
            "--beta-start",
            str(args.dagger_beta_start),
            "--beta-end",
            str(args.dagger_beta_end),
            "--finetune-epochs",
            str(args.dagger_finetune_epochs),
            "--finetune-aux-reward-weight",
            str(args.dagger_finetune_aux_reward_weight),
            "--eval-seeds",
            args.dagger_eval_seeds,
            "--start-seed",
            str(args.dagger_start_seed),
            "--bootstrap-episodes",
            str(args.dagger_bootstrap_episodes),
        ]
        if args.dagger_target_return is not None:
            dagger_cmd.extend(["--target-return", str(args.dagger_target_return)])
        _run(dagger_cmd, cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip dagger stage")

    if not args.skip_plot:
        _run([py, "plot.py"], cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip plot stage")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
