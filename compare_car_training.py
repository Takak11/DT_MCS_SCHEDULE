import argparse
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import train as train_module
from config import CONFIG


def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_eval_seeds(seed_text):
    if seed_text is None or seed_text.strip() == "":
        return None
    return [int(x.strip()) for x in seed_text.split(",") if x.strip() != ""]


def _run_single(label, use_car, dataset_path, log_dir, epochs, eval_seeds, seed):
    log_path = log_dir / f"train_log_{label}.csv"
    print(f"\n=== [{label}] use_car_module={use_car} ===")
    print(f"log: {log_path}")

    old_use_car = bool(CONFIG.get("use_car_module", True))
    old_epochs = int(train_module.EPOCHS)
    old_eval_seeds = list(train_module.EVAL_SEEDS)

    try:
        CONFIG["use_car_module"] = bool(use_car)
        train_module.EPOCHS = int(epochs)
        if eval_seeds is not None:
            train_module.EVAL_SEEDS = list(eval_seeds)

        _set_global_seed(seed)
        train_module.train(
            dataset_path=str(dataset_path),
            log_path=str(log_path),
            append_log=False,
        )
    finally:
        CONFIG["use_car_module"] = old_use_car
        train_module.EPOCHS = old_epochs
        train_module.EVAL_SEEDS = old_eval_seeds

    # Keep separate checkpoints for later manual evaluation.
    ckpt_map = {
        "dt_mcs_best.pth": f"dt_mcs_best_{label}.pth",
        "dt_mcs_best_success.pth": f"dt_mcs_best_success_{label}.pth",
        "dt_mcs_best_val.pth": f"dt_mcs_best_val_{label}.pth",
    }
    for src, dst in ckpt_map.items():
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, log_dir / dst)

    return log_path


def _plot_compare(log_no_car, log_car, out_path):
    df0 = pd.read_csv(log_no_car).copy()
    df0["setting"] = "NO-CAR"
    df1 = pd.read_csv(log_car).copy()
    df1["setting"] = "CAR"
    df = pd.concat([df0, df1], ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"NO-CAR": "#d62728", "CAR": "#1f77b4"}

    for setting, g in df.groupby("setting"):
        axes[0].plot(g["epoch"], g["success_rate"], label=setting, color=colors[setting], lw=2)
        axes[1].plot(g["epoch"], g["avg_wait_steps"], label=setting, color=colors[setting], lw=2)
        axes[2].plot(g["epoch"], g["val_loss"], label=setting, color=colors[setting], lw=2)

    axes[0].set_title("Success Rate")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Percent (%)")

    axes[1].set_title("Average Wait (All Waiting)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Steps")

    axes[2].set_title("Validation Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L1 Loss")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    plt.suptitle("DT Training Comparison: CAR vs NO-CAR", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train twice (NO-CAR and CAR) and plot both performance curves."
    )
    parser.add_argument("--dataset-path", type=str, default="expert_dataset.pkl")
    parser.add_argument("--log-dir", type=str, default="result")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    eval_seeds = _parse_eval_seeds(args.eval_seeds)
    log_no_car = log_dir / "train_log_no_car.csv"
    log_car = log_dir / "train_log_car.csv"

    if not args.skip_train:
        log_no_car = _run_single(
            label="no_car",
            use_car=False,
            dataset_path=dataset_path,
            log_dir=log_dir,
            epochs=args.epochs,
            eval_seeds=eval_seeds,
            seed=args.seed,
        )
        log_car = _run_single(
            label="car",
            use_car=True,
            dataset_path=dataset_path,
            log_dir=log_dir,
            epochs=args.epochs,
            eval_seeds=eval_seeds,
            seed=args.seed,
        )
    else:
        if not log_no_car.exists() or not log_car.exists():
            raise FileNotFoundError(
                "skip-train enabled, but train_log_no_car.csv/train_log_car.csv not found in log-dir."
            )

    out_path = log_dir / "car_vs_no_car_training.png"
    _plot_compare(log_no_car, log_car, out_path)


if __name__ == "__main__":
    main()
