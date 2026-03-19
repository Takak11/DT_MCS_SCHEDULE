import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import CONFIG


plt.style.use("seaborn-v0_8-muted")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

prefix = CONFIG.get("result_path", "")


def _load_trajectories(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "trajectories" in data:
        return data["trajectories"]
    return data


def _draw_phase_boundaries(ax, df):
    if "phase" not in df.columns or len(df) <= 1:
        return
    phase = df["phase"].astype(str).fillna("DT")
    change_idx = np.where(phase.values[1:] != phase.values[:-1])[0] + 1
    for idx in change_idx:
        x = df["epoch"].iloc[idx]
        ax.axvline(x=x, color="#7f7f7f", linestyle="--", alpha=0.5, linewidth=1)


def plot_training_and_service_metrics(log_path="train_log_v2.csv"):
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Cannot find {log_path}. Please run training first.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1) Loss
    if {"train_loss", "val_loss"}.issubset(df.columns):
        sns.lineplot(data=df, x="epoch", y="train_loss", ax=axes[0], label="Train Loss", color="#1f77b4")
        sns.lineplot(data=df, x="epoch", y="val_loss", ax=axes[0], label="Val Loss", color="#ff7f0e")
        axes[0].set_title("Loss", fontsize=14)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("L1 Loss (Log Scale)")
    else:
        axes[0].set_visible(False)

    # 2) Distance error
    if "dist_error_meters" in df.columns:
        sns.lineplot(data=df, x="epoch", y="dist_error_meters", ax=axes[1], color="#2ca02c", lw=2)
        axes[1].axhline(y=2000, color="r", linestyle="--", label="MCS Radius (2km)")
        axes[1].set_title("Distance Error", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Meters")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    # 3) Learning rate
    if "learning_rate" in df.columns:
        sns.lineplot(data=df, x="epoch", y="learning_rate", ax=axes[2], color="#9467bd")
        axes[2].set_title("LR Schedule", fontsize=14)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
    else:
        axes[2].set_visible(False)

    # 4) Success rate
    if "success_rate" in df.columns:
        sns.lineplot(data=df, x="epoch", y="success_rate", ax=axes[3], color="#17becf", lw=2)
        if "success_rate_std" in df.columns:
            low = (df["success_rate"] - df["success_rate_std"]).clip(lower=0)
            high = (df["success_rate"] + df["success_rate_std"]).clip(upper=100)
            axes[3].fill_between(df["epoch"], low, high, color="#17becf", alpha=0.2, label="+/-1 std")
            axes[3].legend()
        axes[3].set_title("Success Rate", fontsize=14)
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Percent (%)")
        axes[3].set_ylim(60, 100)
        _draw_phase_boundaries(axes[3], df)
    else:
        axes[3].set_visible(False)

    # 5) Average waiting time (canonical metric: steps)
    if "avg_wait_steps" in df.columns:
        sns.lineplot(data=df, x="epoch", y="avg_wait_steps", ax=axes[4], color="#d62728", lw=2)
        if "avg_wait_steps_std" in df.columns:
            low = (df["avg_wait_steps"] - df["avg_wait_steps_std"]).clip(lower=0)
            high = df["avg_wait_steps"] + df["avg_wait_steps_std"]
            axes[4].fill_between(df["epoch"], low, high, color="#d62728", alpha=0.2, label="+/-1 std")
        step_minutes = CONFIG.get("minutes_per_step", 24.0 * 60.0 / max(1, CONFIG.get("max_steps", 200)))
        target_wait_steps = int(CONFIG.get("wait_target_steps", 2))
        axes[4].axhline(y=target_wait_steps, color="r", linestyle="--", label=f"Target ({target_wait_steps} steps)")
        axes[4].set_title(f"Average Wait (1 step = {step_minutes:.1f} min)", fontsize=14)
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Steps")
        axes[4].legend()
        _draw_phase_boundaries(axes[4], df)
    else:
        axes[4].set_visible(False)

    # 6) Active distance error
    if "active_dist_error_meters" in df.columns:
        sns.lineplot(data=df, x="epoch", y="active_dist_error_meters", ax=axes[5], color="#8c564b", lw=2)
        axes[5].set_title("Active Distance Error", fontsize=14)
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("Meters")
    else:
        axes[5].set_visible(False)

    plt.suptitle("Decision Transformer Training Metrics", fontsize=18, y=0.98)
    out_path = prefix + "dt_performance_report.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved training report to {out_path}")


def visualize_trajectory(dataset_path="expert_dataset.pkl", episode_idx=0, mcs_num=20, ev_slots=50):
    print(f"Loading dataset: {dataset_path}")
    try:
        trajectories = _load_trajectories(dataset_path)
    except FileNotFoundError:
        print(f"Cannot find file: {dataset_path}")
        return

    if episode_idx < 0 or episode_idx >= len(trajectories):
        print(f"Episode index out of range: {episode_idx} (total={len(trajectories)})")
        return

    ep = trajectories[episode_idx]
    states = ep.get("observations", ep.get("states"))
    if states is None:
        print("Dataset format mismatch: no 'observations'/'states' field.")
        return

    seq_len = states.shape[0]
    print(f"Visualizing episode {episode_idx}, steps={seq_len}")

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, mcs_num))

    # EV heat points
    ev_start_idx = mcs_num * 2
    all_ev_lats, all_ev_lons = [], []
    for step in range(0, seq_len, 5):
        for i in range(ev_slots):
            idx = ev_start_idx + i * 3
            lat, lon = states[step, idx], states[step, idx + 1]
            if lat != 0.0 and lon != 0.0:
                all_ev_lats.append(lat)
                all_ev_lons.append(lon)

    if all_ev_lats:
        plt.scatter(all_ev_lons, all_ev_lats, c="red", s=15, alpha=0.8, label="EV Requests")

    # MCS trajectories
    for i in range(mcs_num):
        lats = states[:, i * 2]
        lons = states[:, i * 2 + 1]
        valid_mask = (lats != 0.0) & (lons != 0.0)
        v_lats, v_lons = lats[valid_mask], lons[valid_mask]
        if len(v_lats) == 0:
            continue
        plt.plot(v_lons, v_lats, color=colors[i], alpha=0.7, linewidth=2, zorder=2)
        plt.scatter(v_lons[0], v_lats[0], color=colors[i], marker="s", s=80, edgecolor="black", zorder=3)
        plt.scatter(v_lons[-1], v_lats[-1], color=colors[i], marker="*", s=150, edgecolor="black", zorder=4)

    plt.title(f"Expert Scheduling Trajectory (Episode {episode_idx})", fontsize=16)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys(), loc="upper right")
    plt.tight_layout()

    save_path = f"{prefix}dataset_traj_ep{episode_idx}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory plot: {save_path}")


def visualize_ev_demand(dataset_path="expert_dataset.pkl", episode_idx=0, mcs_num=20, ev_slots=50):
    print(f"Loading dataset: {dataset_path}")
    try:
        trajectories = _load_trajectories(dataset_path)
    except FileNotFoundError:
        print(f"Cannot find file: {dataset_path}")
        return

    if episode_idx < 0 or episode_idx >= len(trajectories):
        print(f"Episode index out of range: {episode_idx} (total={len(trajectories)})")
        return

    ep = trajectories[episode_idx]
    states = ep.get("observations", ep.get("states"))
    if states is None:
        print("Dataset format mismatch: no 'observations'/'states' field.")
        return

    seq_len = states.shape[0]
    ev_start_idx = mcs_num * 2
    all_lats, all_lons, all_socs = [], [], []
    waiting_counts_per_step = []

    for step in range(seq_len):
        step_waiting_count = 0
        for i in range(ev_slots):
            idx = ev_start_idx + i * 3
            lat = states[step, idx]
            lon = states[step, idx + 1]
            soc = states[step, idx + 2]
            if lat != 0.0 and lon != 0.0:
                step_waiting_count += 1
                all_lats.append(lat)
                all_lons.append(lon)
                all_socs.append(soc)
        waiting_counts_per_step.append(step_waiting_count)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"EV Demand Analysis (Episode {episode_idx})", fontsize=16, fontweight="bold", y=1.05)

    ax1 = plt.subplot(1, 3, 1)
    if all_lats:
        sns.kdeplot(x=all_lons, y=all_lats, cmap="Reds", fill=True, bw_adjust=0.5, ax=ax1, alpha=0.8)
        ax1.scatter(all_lons, all_lats, c="black", s=5, alpha=0.1)
    ax1.set_title("Spatial Distribution")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(range(seq_len), waiting_counts_per_step, color="#1f77b4", lw=2)
    ax2.fill_between(range(seq_len), waiting_counts_per_step, color="#1f77b4", alpha=0.2)
    ax2.axhline(y=mcs_num, color="red", linestyle="--", label=f"MCS Capacity ({mcs_num})")
    ax2.set_title("Concurrent Waiting Requests")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("EV Count")
    ax2.legend()

    ax3 = plt.subplot(1, 3, 3)
    if all_socs:
        sns.histplot(all_socs, bins=20, color="#2ca02c", kde=True, ax=ax3)
    ax3.set_title("SOC at Request Time")
    ax3.set_xlabel("State of Charge (SOC)")
    ax3.set_ylabel("Frequency")

    plt.tight_layout()
    save_path = f"{prefix}ev_dataset_analysis_ep{episode_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved EV demand plot: {save_path}")


if __name__ == "__main__":
    plot_training_and_service_metrics()
    visualize_trajectory(episode_idx=100)
    visualize_ev_demand(episode_idx=100)
