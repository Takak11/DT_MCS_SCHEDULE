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
        description="Run full DT-MCS pipeline: generate dataset -> train -> evaluate -> plot"
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable path")
    parser.add_argument("--workdir", type=str, default=".", help="Project working directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print commands")

    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    cwd = Path(args.workdir).resolve()
    py = str(args.python)

    if not cwd.exists():
        raise FileNotFoundError(f"workdir not found: {cwd}")

    print("Pipeline order: generate_DT_dataset -> train -> evaluate -> plot")
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

    if not args.skip_evaluate:
        _run([py, "evaluate.py"], cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip evaluate stage")

    if not args.skip_plot:
        _run([py, "plot.py"], cwd=cwd, dry_run=args.dry_run)
    else:
        print("\n>>> skip plot stage")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
