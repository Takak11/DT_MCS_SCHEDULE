import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG

# 设置全局样式
plt.style.use('seaborn-v0_8-muted')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

prefix = CONFIG['result_path']


def plot_training_and_service_metrics(log_path="train_log_v2.csv"):
    # 1. load log
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Cannot find {log_path}. Please run training first.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1) Loss
    sns.lineplot(data=df, x='epoch', y='train_loss', ax=axes[0], label='Train Loss', color='#1f77b4')
    sns.lineplot(data=df, x='epoch', y='val_loss', ax=axes[0], label='Val Loss', color='#ff7f0e')
    axes[0].set_title("Loss", fontsize=14)
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 Loss (Log Scale)")

    # 2) Distance error
    if 'dist_error_meters' in df.columns:
        sns.lineplot(data=df, x='epoch', y='dist_error_meters', ax=axes[1], color='#2ca02c', lw=2)
        axes[1].axhline(y=2000, color='r', linestyle='--', label='MCS Radius (2km)')
        axes[1].set_title("Distance Error", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Meters")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    # 3) Learning rate
    sns.lineplot(data=df, x='epoch', y='learning_rate', ax=axes[2], color='#9467bd')
    axes[2].set_title("LR Schedule", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")

    # 4) Success rate over training
    if 'success_rate' in df.columns:
        sns.lineplot(data=df, x='epoch', y='success_rate', ax=axes[3], color='#17becf', lw=2)
        if 'success_rate_std' in df.columns:
            low = (df['success_rate'] - df['success_rate_std']).clip(lower=0)
            high = (df['success_rate'] + df['success_rate_std']).clip(upper=100)
            axes[3].fill_between(df['epoch'], low, high, color='#17becf', alpha=0.2, label='±1 std')
        axes[3].set_title("Success Rate", fontsize=14)
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Percent (%)")
        axes[3].set_ylim(60, 100)
        if 'success_rate_std' in df.columns:
            axes[3].legend()
    else:
        axes[3].set_visible(False)

    # 5) Waiting-time curve over training
    if 'avg_wait_minutes' in df.columns:
        sns.lineplot(data=df, x='epoch', y='avg_wait_minutes', ax=axes[4], color='#d62728', lw=2)
        if 'avg_wait_minutes_std' in df.columns:
            low = (df['avg_wait_minutes'] - df['avg_wait_minutes_std']).clip(lower=0)
            high = df['avg_wait_minutes'] + df['avg_wait_minutes_std']
            axes[4].fill_between(df['epoch'], low, high, color='#d62728', alpha=0.2, label='±1 std')
        axes[4].set_title("Average Wait Time (Served)", fontsize=14)
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Minutes")
        target_wait_minutes = CONFIG.get('wait_target_steps', 2) * CONFIG.get('minutes_per_step', 24.0 * 60.0 / max(1, CONFIG.get('max_steps', 200)))
        axes[4].axhline(y=target_wait_minutes, color='r', linestyle='--', label=f"Target ({target_wait_minutes:.1f} min)")
        axes[4].legend()
    elif 'avg_wait_steps' in df.columns:
        sns.lineplot(data=df, x='epoch', y='avg_wait_steps', ax=axes[4], color='#d62728', lw=2)
        if 'avg_wait_steps_std' in df.columns:
            low = (df['avg_wait_steps'] - df['avg_wait_steps_std']).clip(lower=0)
            high = df['avg_wait_steps'] + df['avg_wait_steps_std']
            axes[4].fill_between(df['epoch'], low, high, color='#d62728', alpha=0.2, label='±1 std')
        axes[4].set_title("Average Wait Steps (Served)", fontsize=14)
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("Steps")
        target_wait_steps = CONFIG.get('wait_target_steps', 2)
        axes[4].axhline(y=target_wait_steps, color='r', linestyle='--', label=f"Target ({target_wait_steps} steps)")
        axes[4].legend()
    else:
        axes[4].set_visible(False)

    # 6) Wait-within-target rate over training
    if 'wait_within_target_rate' in df.columns:
        sns.lineplot(data=df, x='epoch', y='wait_within_target_rate', ax=axes[5], color='#8c564b', lw=2)
        if 'wait_within_target_rate_std' in df.columns:
            low = (df['wait_within_target_rate'] - df['wait_within_target_rate_std']).clip(lower=0)
            high = (df['wait_within_target_rate'] + df['wait_within_target_rate_std']).clip(upper=100)
            axes[5].fill_between(df['epoch'], low, high, color='#8c564b', alpha=0.2, label='卤1 std')
        target_steps = int(CONFIG.get('wait_target_steps', 2))
        axes[5].set_title(f"W<={target_steps} Steps Rate", fontsize=14)
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("Percent (%)")
        axes[5].set_ylim(0, 100)
        if 'wait_within_target_rate_std' in df.columns:
            axes[5].legend()
    else:
        axes[5].set_visible(False)

    plt.suptitle("Decision Transformer Training Metrics", fontsize=18, y=0.98)
    plt.savefig(prefix + "dt_performance_report.png", dpi=300, bbox_inches='tight')
    print("Saved training report to dt_performance_report.png")

def visualize_trajectory(dataset_path="expert_dataset.pkl", episode_idx=0, mcs_num=20, ev_slots=50):
    print(f"正在加载数据集: {dataset_path} ...")
    try:
        with open(dataset_path, "rb") as f:
            # 兼容不同的保存格式 (直接是列表，或者字典里包着 trajectories)
            data = pickle.load(f)
            trajectories = data['trajectories'] if isinstance(data, dict) and 'trajectories' in data else data
    except FileNotFoundError:
        print(f"找不到文件 {dataset_path}！请检查路径。")
        return

    if episode_idx >= len(trajectories):
        print(f"索引越界, 数据集只有 {len(trajectories)} 个回合。")
        return

    ep = trajectories[episode_idx]

    # 提取状态矩阵 (假设键名为 'observations' 或 'states')
    states = ep.get('observations', ep.get('states'))
    if states is None:
        print("数据集格式不匹配，找不到 'observations' 字段。")
        return

    # states 形状: [seq_len, state_dim]
    seq_len = states.shape[0]
    print(f"提取 Episode {episode_idx} | 总步数: {seq_len}")

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, mcs_num))

    # --- 1. 绘制求救的 EV (背景散点) ---
    # EV 数据紧跟在 MCS 数据后面：mcs_num * 2 之后
    ev_start_idx = mcs_num * 2
    # 我们抽取整个回合中出现过的所有 EV 坐标
    all_ev_lats, all_ev_lons = [], []
    for step in range(0, seq_len, 5):  # 每隔 5 步采样一次 EV 避免太密集
        for i in range(ev_slots):
            idx = ev_start_idx + i * 3
            lat, lon = states[step, idx], states[step, idx + 1]
            if lat != 0.0 and lon != 0.0:  # 排除 Padding 的 0
                all_ev_lats.append(lat)
                all_ev_lons.append(lon)

    plt.scatter(all_ev_lons, all_ev_lats, c='red', s=15, alpha=0.8, label='求救 EV 热点')

    # --- 2. 绘制 MCS 移动轨迹 ---
    for i in range(mcs_num):
        lats = states[:, i * 2]
        lons = states[:, i * 2 + 1]

        # 过滤掉可能的 Padding 0 坐标（如果在某些 step 车辆隐身了）
        valid_mask = (lats != 0.0) & (lons != 0.0)
        v_lats, v_lons = lats[valid_mask], lons[valid_mask]

        if len(v_lats) == 0: continue

        # 画轨迹线
        plt.plot(v_lons, v_lats, color=colors[i], alpha=0.7, linewidth=2, zorder=2)

        # 画起点 (方形)
        plt.scatter(v_lons[0], v_lats[0], color=colors[i], marker='s', s=80, edgecolor='black', zorder=3)
        # 画终点 (五角星)
        plt.scatter(v_lons[-1], v_lats[-1], color=colors[i], marker='*', s=150, edgecolor='black', zorder=4)

    plt.title(f"二分图专家算法调度轨迹 Episode {episode_idx}", fontsize=16)
    plt.xlabel("经度 (Longitude)", fontsize=12)
    plt.ylabel("纬度 (Latitude)", fontsize=12)

    # 优化图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()
    save_path = f"{prefix}dataset_traj_ep{episode_idx}.png"
    plt.savefig(save_path, dpi=300)
    print(f"轨迹图已生成: {save_path}")


def visualize_ev_demand(dataset_path="expert_dataset.pkl", episode_idx=0, mcs_num=20, ev_slots=50):
    print(f"正在加载数据集: {dataset_path} ...")
    try:
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
            trajectories = data['trajectories'] if isinstance(data, dict) and 'trajectories' in data else data
    except FileNotFoundError:
        print(f"找不到文件 {dataset_path}！")
        return

    if episode_idx >= len(trajectories):
        print(f"索引越界！")
        return

    ep = trajectories[episode_idx]
    states = ep.get('observations', ep.get('states'))

    if states is None:
        print("找不到状态数据。")
        return

    seq_len = states.shape[0]

    # --- 数据提取与预处理 ---
    # EV 特征起始索引
    ev_start_idx = mcs_num * 2

    all_lats, all_lons, all_socs = [], [], []
    waiting_counts_per_step = []

    # 遍历每个时间步，提取 EV 信息
    for step in range(seq_len):
        step_waiting_count = 0
        for i in range(ev_slots):
            idx = ev_start_idx + i * 3
            lat = states[step, idx]
            lon = states[step, idx + 1]
            soc = states[step, idx + 2]

            # 过滤掉由于 Padding 补的 0 (假设成都市的经纬度绝不可能是 0)
            if lat != 0.0 and lon != 0.0:
                step_waiting_count += 1
                all_lats.append(lat)
                all_lons.append(lon)
                all_socs.append(soc)

        waiting_counts_per_step.append(step_waiting_count)

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(f"EV 充电需求态势分析 (Episode {episode_idx})", fontsize=16, fontweight='bold', y=1.05)

    # 1. EV 空间求救热力图 (Spatial Heatmap)
    ax1 = plt.subplot(1, 3, 1)
    # 使用 seaborn 画核密度估计(KDE)热力图
    sns.kdeplot(x=all_lons, y=all_lats, cmap="Reds", fill=True, bw_adjust=0.5, ax=ax1, alpha=0.8)
    ax1.scatter(all_lons, all_lats, c='black', s=5, alpha=0.1)  # 叠加散点
    ax1.set_title("EV 求救空间热力分布", fontsize=14)
    ax1.set_xlabel("经度")
    ax1.set_ylabel("纬度")

    # 2. 同时等待救援的 EV 数量随时间的变化 (Demand over Time)
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(range(seq_len), waiting_counts_per_step, color='#1f77b4', lw=2)
    ax2.fill_between(range(seq_len), waiting_counts_per_step, color='#1f77b4', alpha=0.2)
    ax2.axhline(y=mcs_num, color='red', linestyle='--', label=f'MCS 总运力基准 ({mcs_num}辆)')
    ax2.set_title("随时间变化的求救并发量", fontsize=14)
    ax2.set_xlabel("时间步 (Step)")
    ax2.set_ylabel("同时求救的 EV 数量 (辆)")
    ax2.legend()

    # 3. 发出求救时的 SOC 分布 (SOC Distribution)
    ax3 = plt.subplot(1, 3, 3)
    sns.histplot(all_socs, bins=20, color='#2ca02c', kde=True, ax=ax3)
    ax3.set_title("EV 发出求救时的电量 (SOC) 分布", fontsize=14)
    ax3.set_xlabel("State of Charge (SOC)")
    ax3.set_ylabel("频次")

    plt.tight_layout()
    save_path = f"{prefix}ev_dataset_analysis_ep{episode_idx}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"EV态势图已生成: {save_path}")


# 请确保 mcs_num 和 ev_slots 与你生成该数据集时的配置一致
visualize_ev_demand(dataset_path="expert_dataset.pkl", episode_idx=200, mcs_num=20, ev_slots=50)
# 假设你的 mcs 数量是 20，设定的等待 EV 槽位是 50
# 可以修改 episode_idx 看看不同回合的情况
visualize_trajectory(dataset_path="expert_dataset.pkl", episode_idx=200, mcs_num=20, ev_slots=50)
plot_training_and_service_metrics()
