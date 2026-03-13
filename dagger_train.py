import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from config import CONFIG
from env import ChargingEnv
from DT import DecisionTransformer
from generate_DT_dataset import (
    assignment_memory,
    expert_get_action_with_commitment,
    generate_offline_dataset,
    get_state_vector,
)
import train as train_module


def _sanitize_action_matrix(env, action_matrix):
    """Clamp policy actions into valid map bounds and replace non-finite values."""
    mcs_list = list(env.mcs.values())
    arr = np.asarray(action_matrix, dtype=np.float32).reshape(len(mcs_list), 2).copy()
    lat_min, lat_max = env.cfg["SOUTH"], env.cfg["NORTH"]
    lon_min, lon_max = env.cfg["WEST"], env.cfg["EAST"]

    for i, mcs in enumerate(mcs_list):
        lat, lon = float(arr[i, 0]), float(arr[i, 1])
        if not np.isfinite(lat) or not np.isfinite(lon):
            lat, lon = float(mcs.pos[0]), float(mcs.pos[1])
        lat = float(np.clip(lat, lat_min, lat_max))
        lon = float(np.clip(lon, lon_min, lon_max))
        arr[i, 0], arr[i, 1] = lat, lon
    return arr


def _load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "trajectories" in data:
        return list(data["trajectories"])
    return list(data)


def _save_dataset(path, trajectories):
    with open(path, "wb") as f:
        pickle.dump(trajectories, f)


def _bootstrap_dataset(base_dataset_path, dagger_dataset_path, bootstrap_episodes):
    if not base_dataset_path.exists():
        print(
            f"Base dataset not found: {base_dataset_path}. "
            f"Generating {bootstrap_episodes} expert episodes first."
        )
        generate_offline_dataset(
            episodes=bootstrap_episodes,
            save_path=str(base_dataset_path),
            base_seed=42,
            epsilon=float(CONFIG.get("expert_epsilon", 0.0)),
        )
    if not dagger_dataset_path.exists():
        shutil.copy2(base_dataset_path, dagger_dataset_path)
        print(f"Initialized DAgger dataset: {dagger_dataset_path} <- {base_dataset_path}")


def _prepare_scalers_and_dims(dataset_path):
    ds = train_module.ExpertDataset(str(dataset_path), train_module.CONTEXT_LEN)
    return {
        "state_mean": ds.state_mean,
        "state_std": ds.state_std,
        "action_mean": ds.action_mean,
        "action_std": ds.action_std,
        "rtg_scale": ds.rtg_scale,
        "state_dim": ds.state_dim,
        "action_dim": ds.action_dim,
    }


def _load_policy(device, state_dim, action_dim, ckpt_path):
    model = DecisionTransformer(
        state_dim=state_dim, action_dim=action_dim, max_length=CONFIG["max_steps"]
    ).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        load_res = model.load_state_dict(ckpt, strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print("Warning: checkpoint partially loaded due to architecture mismatch.")
    else:
        print(f"Warning: checkpoint not found: {ckpt_path}, using randomly initialized policy.")
    model.eval()
    return model


def _collect_one_episode(
    model,
    env,
    s_mean,
    s_std,
    a_mean,
    a_std,
    rtg_scale,
    device,
    beta,
    target_return,
):
    assignment_memory.clear()
    env.reset()

    states_hist, actions_hist, rtgs_hist, timesteps_hist = [], [], [], []
    current_rtg = target_return

    obs_seq, expert_action_seq, reward_seq = [], [], []
    final_info = {}

    context_len = train_module.CONTEXT_LEN
    mcs_num = env.cfg["mcs_num"]

    with torch.no_grad():
        for t in range(env.cfg["max_steps"]):
            raw_state = get_state_vector(env)
            obs_seq.append(raw_state.astype(np.float32))

            norm_state = (raw_state - s_mean) / s_std
            states_hist.append(norm_state)
            rtgs_hist.append([current_rtg / rtg_scale])
            timesteps_hist.append(t)

            if len(actions_hist) < len(states_hist):
                actions_hist.append(np.zeros(len(a_mean), dtype=np.float32))

            s_input = np.array(states_hist[-context_len:])
            a_input = np.array(actions_hist[-context_len:])
            rtg_input = np.array(rtgs_hist[-context_len:])
            t_input = np.array(timesteps_hist[-context_len:])

            pad_len = context_len - len(s_input)
            attn_mask = (
                np.concatenate([np.zeros(pad_len), np.ones(len(s_input))])
                if pad_len > 0
                else np.ones(context_len)
            )
            if pad_len > 0:
                s_input = np.pad(s_input, ((pad_len, 0), (0, 0)), mode="constant")
                a_input = np.pad(a_input, ((pad_len, 0), (0, 0)), mode="constant")
                rtg_input = np.pad(rtg_input, ((pad_len, 0), (0, 0)), mode="constant")
                t_input = np.pad(t_input, (pad_len, 0), mode="constant")

            s_tensor = torch.FloatTensor(s_input).unsqueeze(0).to(device)
            a_tensor = torch.FloatTensor(a_input).unsqueeze(0).to(device)
            rtg_tensor = torch.FloatTensor(rtg_input).unsqueeze(0).to(device)
            t_tensor = torch.LongTensor(t_input).unsqueeze(0).to(device)
            m_tensor = torch.BoolTensor(attn_mask).unsqueeze(0).to(device)

            action_preds = model(s_tensor, a_tensor, rtg_tensor, t_tensor, attention_mask=m_tensor)
            pred_action_norm = action_preds[0, -1].cpu().numpy()
            model_action = (pred_action_norm * a_std + a_mean).reshape(mcs_num, 2)
            model_action = _sanitize_action_matrix(env, model_action)

            expert_action = expert_get_action_with_commitment(
                env, epsilon=float(CONFIG.get("expert_epsilon", 0.0))
            )
            expert_action = _sanitize_action_matrix(env, expert_action)
            run_expert = (np.random.rand() < beta)
            exec_action = expert_action if run_expert else model_action
            exec_action = _sanitize_action_matrix(env, exec_action)

            expert_action_seq.append(np.asarray(expert_action, dtype=np.float32))

            exec_action_flat = np.asarray(exec_action, dtype=np.float32).reshape(-1)
            exec_action_norm = (exec_action_flat - a_mean) / a_std
            actions_hist[-1] = exec_action_norm

            _, reward, done, info = env.step(exec_action)
            reward_seq.append(float(reward))
            final_info = info
            current_rtg -= reward

            if done:
                break

    returns_to_go = np.zeros_like(reward_seq, dtype=np.float32)
    curr_rtg = 0.0
    for idx in reversed(range(len(reward_seq))):
        curr_rtg += reward_seq[idx]
        returns_to_go[idx] = curr_rtg

    trajectory = {
        "observations": np.array(obs_seq, dtype=np.float32),
        "actions": np.array(expert_action_seq, dtype=np.float32),
        "rewards": np.array(reward_seq, dtype=np.float32),
        "returns_to_go": returns_to_go,
    }
    metrics = {
        "success_rate": float(final_info.get("success_rate", env._calculate_success_rate())),
        "served_total": int(final_info.get("served_total", env.stats.get("served_mcs", 0) + env.stats.get("served_fcs", 0))),
        "total_requests": int(env.stats.get("total_requests", 0)),
    }
    return trajectory, metrics


def _collect_dagger_rollouts(
    model,
    collect_episodes,
    beta,
    start_seed,
    target_return,
    scalers,
    device,
):
    collect_cfg = dict(CONFIG)
    collect_cfg["verbose_dataset_load"] = False
    env = ChargingEnv(collect_cfg)

    trajectories = []
    success_rates = []
    served = 0
    total_requests = 0
    for ep in range(collect_episodes):
        env.seed(start_seed + ep)
        traj, info = _collect_one_episode(
            model=model,
            env=env,
            s_mean=scalers["state_mean"],
            s_std=scalers["state_std"],
            a_mean=scalers["action_mean"],
            a_std=scalers["action_std"],
            rtg_scale=scalers["rtg_scale"],
            device=device,
            beta=beta,
            target_return=target_return,
        )
        trajectories.append(traj)
        success_rates.append(info["success_rate"])
        served += info["served_total"]
        total_requests += info["total_requests"]

    summary = {
        "episodes": collect_episodes,
        "beta": beta,
        "success_rate_mean": float(np.mean(success_rates)) if success_rates else 0.0,
        "service_rate_mean": float(100.0 * served / total_requests) if total_requests > 0 else 100.0,
    }
    return trajectories, summary


def _run_finetune(dataset_path, init_ckpt_path, finetune_epochs, eval_seeds):
    old_epochs = train_module.EPOCHS
    old_eval_seeds = list(train_module.EVAL_SEEDS)
    try:
        train_module.EPOCHS = finetune_epochs
        train_module.EVAL_SEEDS = list(eval_seeds)
        train_module.train(dataset_path=str(dataset_path), init_ckpt=str(init_ckpt_path))
    finally:
        train_module.EPOCHS = old_epochs
        train_module.EVAL_SEEDS = old_eval_seeds


def _beta_for_iter(iter_idx, num_iters, beta_start, beta_end):
    if num_iters <= 1:
        return beta_end
    alpha = iter_idx / float(num_iters - 1)
    return beta_start + alpha * (beta_end - beta_start)


def run_dagger(
    base_dataset_path,
    dagger_dataset_path,
    init_ckpt_path,
    num_iters,
    collect_episodes,
    beta_start,
    beta_end,
    finetune_epochs,
    eval_seeds,
    start_seed,
    target_return,
    bootstrap_episodes,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dataset_path = Path(base_dataset_path)
    dagger_dataset_path = Path(dagger_dataset_path)
    init_ckpt_path = Path(init_ckpt_path)

    _bootstrap_dataset(base_dataset_path, dagger_dataset_path, bootstrap_episodes)

    for it in range(num_iters):
        beta = _beta_for_iter(it, num_iters, beta_start, beta_end)

        scalers = _prepare_scalers_and_dims(dagger_dataset_path)
        model = _load_policy(
            device=device,
            state_dim=scalers["state_dim"],
            action_dim=scalers["action_dim"],
            ckpt_path=init_ckpt_path,
        )

        new_trajs, summary = _collect_dagger_rollouts(
            model=model,
            collect_episodes=collect_episodes,
            beta=beta,
            start_seed=start_seed + it * 10000,
            target_return=target_return,
            scalers=scalers,
            device=device,
        )
        all_trajs = _load_dataset(dagger_dataset_path)
        all_trajs.extend(new_trajs)
        _save_dataset(dagger_dataset_path, all_trajs)
        print(
            f"[DAgger {it + 1}/{num_iters}] "
            f"beta={summary['beta']:.3f} | "
            f"collect SR={summary['success_rate_mean']:.2f}% | "
            f"collect Service={summary['service_rate_mean']:.2f}% | "
            f"dataset_size={len(all_trajs)}"
        )

        _run_finetune(
            dataset_path=dagger_dataset_path,
            init_ckpt_path=init_ckpt_path,
            finetune_epochs=finetune_epochs,
            eval_seeds=eval_seeds,
        )
        # train() writes newest business-best checkpoint here.
        init_ckpt_path = Path("dt_mcs_best.pth")

    print(f"DAgger done. Aggregated dataset: {dagger_dataset_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DAgger fine-tuning for DT-MCS.")
    parser.add_argument("--base-dataset", type=str, default="expert_dataset.pkl")
    parser.add_argument("--dagger-dataset", type=str, default="expert_dataset_dagger.pkl")
    parser.add_argument("--init-ckpt", type=str, default="dt_mcs_best.pth")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--collect-episodes", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=0.8)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--eval-seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--start-seed", type=int, default=7000)
    parser.add_argument("--target-return", type=float, default=train_module.EVAL_TARGET_RETURN)
    parser.add_argument("--bootstrap-episodes", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip() != ""]
    run_dagger(
        base_dataset_path=args.base_dataset,
        dagger_dataset_path=args.dagger_dataset,
        init_ckpt_path=args.init_ckpt,
        num_iters=args.iters,
        collect_episodes=args.collect_episodes,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        finetune_epochs=args.finetune_epochs,
        eval_seeds=eval_seeds,
        start_seed=args.start_seed,
        target_return=args.target_return,
        bootstrap_episodes=args.bootstrap_episodes,
    )
