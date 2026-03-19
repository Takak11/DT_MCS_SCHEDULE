import argparse
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch

import train as train_module
from DT import DecisionTransformer
from car_module import apply_constraint_aware_reranking
from config import CONFIG
from env import ChargingEnv
from generate_DT_dataset import assignment_memory, expert_get_action_with_commitment, get_state_vector

ROOT_CKPT_NAMES = [
    "dt_mcs_best.pth",
    "dt_mcs_best_success.pth",
    "dt_mcs_best_val.pth",
    "dt_mcs_best_business.pth",
]


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _copy_latest_best(dst_path):
    src = Path("dt_mcs_best.pth")
    if not src.exists():
        raise FileNotFoundError("dt_mcs_best.pth not found after training phase.")
    shutil.copy2(src, dst_path)


def _backup_root_checkpoints(backup_dir):
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ROOT_CKPT_NAMES:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, backup_dir / name)


def _restore_root_checkpoints(backup_dir):
    for name in ROOT_CKPT_NAMES:
        src = backup_dir / name
        dst = Path(name)
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(ckpt_path, scalers, device):
    model = DecisionTransformer(
        state_dim=len(scalers["state_mean"]),
        action_dim=len(scalers["action_mean"]),
        max_length=CONFIG["max_steps"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def _pad_context(seq, context_len, pad_value=0.0, pad_2d=True):
    pad_len = context_len - len(seq)
    if pad_len <= 0:
        return np.array(seq[-context_len:]), np.ones(context_len, dtype=np.float32)
    arr = np.array(seq)
    if pad_2d:
        arr = np.pad(arr, ((pad_len, 0), (0, 0)), mode="constant", constant_values=pad_value)
    else:
        arr = np.pad(arr, (pad_len, 0), mode="constant", constant_values=pad_value)
    mask = np.concatenate([np.zeros(pad_len, dtype=np.float32), np.ones(len(seq), dtype=np.float32)])
    return arr, mask


def _generate_single_model_history_traj(
    model,
    scalers,
    cfg,
    seed,
    target_return,
    context_len,
    use_car_rollout,
    device,
):
    env_cfg = dict(cfg)
    env_cfg["verbose_dataset_load"] = False
    env = ChargingEnv(env_cfg)
    env.seed(seed)
    env.reset()
    assignment_memory.clear()

    s_mean = scalers["state_mean"]
    s_std = scalers["state_std"]
    a_mean = scalers["action_mean"]
    a_std = scalers["action_std"]
    rtg_scale = float(scalers["rtg_scale"])

    obs_seq = []
    expert_action_seq = []
    reward_seq = []

    states_hist = []
    actions_hist = []
    rtgs_hist = []
    timesteps_hist = []
    current_rtg = float(target_return)

    with torch.no_grad():
        for t in range(env_cfg["max_steps"]):
            raw_state = get_state_vector(env)
            norm_state = (raw_state - s_mean) / s_std

            # Label uses expert action on current model-induced state.
            expert_action = expert_get_action_with_commitment(env, epsilon=0.0)
            expert_action_seq.append(np.asarray(expert_action, dtype=np.float32))
            obs_seq.append(np.asarray(raw_state, dtype=np.float32))

            states_hist.append(norm_state)
            rtgs_hist.append([current_rtg / max(1e-6, rtg_scale)])
            timesteps_hist.append(t)
            if len(actions_hist) < len(states_hist):
                actions_hist.append(np.zeros(len(a_mean), dtype=np.float32))

            s_input, attn_mask = _pad_context(states_hist, context_len, pad_2d=True)
            a_input, _ = _pad_context(actions_hist, context_len, pad_2d=True)
            rtg_input, _ = _pad_context(rtgs_hist, context_len, pad_2d=True)
            t_input, _ = _pad_context(timesteps_hist, context_len, pad_2d=False)

            s_tensor = torch.FloatTensor(s_input).unsqueeze(0).to(device)
            a_tensor = torch.FloatTensor(a_input).unsqueeze(0).to(device)
            rtg_tensor = torch.FloatTensor(rtg_input).unsqueeze(0).to(device)
            t_tensor = torch.LongTensor(t_input).unsqueeze(0).to(device)
            m_tensor = torch.BoolTensor(attn_mask).unsqueeze(0).to(device)

            pred = model(s_tensor, a_tensor, rtg_tensor, t_tensor, attention_mask=m_tensor)
            pred_norm = pred[0, -1].cpu().numpy()
            actions_hist[-1] = pred_norm

            action_matrix = (pred_norm * a_std + a_mean).reshape(env_cfg["mcs_num"], 2)
            if use_car_rollout:
                action_matrix = apply_constraint_aware_reranking(env, action_matrix)

            _, reward, done, _ = env.step(action_matrix)
            reward_seq.append(float(reward))
            current_rtg -= float(reward)
            if done:
                break

    returns_to_go = np.zeros(len(reward_seq), dtype=np.float32)
    running = 0.0
    for i in range(len(reward_seq) - 1, -1, -1):
        running += reward_seq[i]
        returns_to_go[i] = running

    return {
        "observations": np.asarray(obs_seq, dtype=np.float32),
        "actions": np.asarray(expert_action_seq, dtype=np.float32),
        "rewards": np.asarray(reward_seq, dtype=np.float32),
        "returns_to_go": returns_to_go,
    }


def generate_model_history_dataset(
    ckpt_path,
    scaler_path,
    out_path,
    episodes,
    seed_base,
    target_return,
    context_len,
    use_car_rollout,
    device,
):
    scalers = _load_pickle(scaler_path)
    model = _build_model(ckpt_path, scalers, device)

    data = []
    print(
        f"Generate model-history dataset: episodes={episodes}, seed_base={seed_base}, "
        f"use_car_rollout={use_car_rollout}"
    )
    for ep in range(episodes):
        traj = _generate_single_model_history_traj(
            model=model,
            scalers=scalers,
            cfg=CONFIG,
            seed=seed_base + ep,
            target_return=target_return,
            context_len=context_len,
            use_car_rollout=use_car_rollout,
            device=device,
        )
        data.append(traj)
        if (ep + 1) % 20 == 0 or (ep + 1) == episodes:
            print(f"  generated {ep + 1}/{episodes}")

    _save_pickle(data, out_path)
    return out_path


def build_mixed_dataset(base_dataset_path, model_dataset_path, out_path, total_episodes, mix_ratio, seed):
    base = _load_pickle(base_dataset_path)
    model_hist = _load_pickle(model_dataset_path)
    rng = np.random.default_rng(seed)

    model_num = int(round(total_episodes * mix_ratio))
    model_num = max(1, min(model_num, total_episodes))
    base_num = max(0, total_episodes - model_num)

    base_idx = rng.choice(len(base), size=base_num, replace=(base_num > len(base)))
    model_idx = rng.choice(len(model_hist), size=model_num, replace=(model_num > len(model_hist)))

    mixed = [base[i] for i in base_idx] + [model_hist[i] for i in model_idx]
    rng.shuffle(mixed)
    _save_pickle(mixed, out_path)
    print(
        f"Built mixed dataset: total={len(mixed)}, base={base_num}, "
        f"model_hist={model_num}, mix_ratio={mix_ratio:.2f}"
    )
    return out_path


def run_train_phase(dataset_path, log_path, epochs, init_ckpt, target_return, use_car_module):
    old_epochs = train_module.EPOCHS
    old_target = train_module.EVAL_TARGET_RETURN
    old_use_car = bool(CONFIG.get("use_car_module", True))
    try:
        train_module.EPOCHS = int(epochs)
        train_module.EVAL_TARGET_RETURN = float(target_return)
        CONFIG["use_car_module"] = bool(use_car_module)
        train_module.train(
            dataset_path=str(dataset_path),
            init_ckpt=str(init_ckpt) if init_ckpt is not None else None,
            log_path=str(log_path),
            append_log=False,
        )
    finally:
        train_module.EPOCHS = old_epochs
        train_module.EVAL_TARGET_RETURN = old_target
        CONFIG["use_car_module"] = old_use_car


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Performative mixed training inspired by arXiv:2405.14219. "
            "Phase-0 (expert-only) + iterative mixed phases (expert + model-history)."
        )
    )
    parser.add_argument("--base-dataset", type=str, default="expert_dataset.pkl")
    parser.add_argument("--work-dir", type=str, default="result/performative_mix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-return", type=float, default=182500.0)
    parser.add_argument("--context-len", type=int, default=50)

    parser.add_argument("--early-epochs", type=int, default=15)
    parser.add_argument("--mix-iters", type=int, default=3)
    parser.add_argument("--mix-epochs", type=int, default=8)
    parser.add_argument("--mix-total-episodes", type=int, default=2000)
    parser.add_argument("--mix-ratio", type=float, default=0.35)

    parser.add_argument("--rollout-use-car", action="store_true")
    parser.add_argument("--disable-car-module", action="store_true")
    parser.add_argument("--skip-early", action="store_true")
    parser.add_argument("--keep-root-ckpt", action="store_true")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base_dataset = Path(args.base_dataset)
    if not base_dataset.exists():
        raise FileNotFoundError(f"base dataset not found: {base_dataset}")

    current_ckpt = None
    root_ckpt_backup_dir = work_dir / "_root_ckpt_backup"
    _backup_root_checkpoints(root_ckpt_backup_dir)
    train_use_car = not bool(args.disable_car_module)
    print(
        f"PMIX mode | train_use_car_module={train_use_car} | "
        f"rollout_use_car={bool(args.rollout_use_car)}"
    )

    try:
        # Phase-0: early training on expert-only dataset.
        if not args.skip_early:
            print("\n=== Phase-0: Expert-only training ===")
            phase0_log = work_dir / "train_log_phase0.csv"
            run_train_phase(
                dataset_path=base_dataset,
                log_path=phase0_log,
                epochs=args.early_epochs,
                init_ckpt=None,
                target_return=args.target_return,
                use_car_module=train_use_car,
            )
            phase0_ckpt = work_dir / "model_phase0_best.pth"
            _copy_latest_best(phase0_ckpt)
            current_ckpt = phase0_ckpt
        else:
            latest = work_dir / "model_phase0_best.pth"
            if not latest.exists():
                raise FileNotFoundError("skip-early enabled but model_phase0_best.pth not found.")
            current_ckpt = latest

        # Mixed phases: inject model-generated action histories.
        for it in range(1, args.mix_iters + 1):
            print(f"\n=== Mixed Iteration {it}/{args.mix_iters} ===")
            model_num = max(1, int(round(args.mix_total_episodes * args.mix_ratio)))
            model_data_path = work_dir / f"model_history_iter{it}.pkl"
            mixed_data_path = work_dir / f"mixed_dataset_iter{it}.pkl"
            mix_log = work_dir / f"train_log_mix_iter{it}.csv"
            mix_ckpt = work_dir / f"model_mix_iter{it}_best.pth"

            generate_model_history_dataset(
                ckpt_path=current_ckpt,
                scaler_path="scaler_params.pkl",
                out_path=model_data_path,
                episodes=model_num,
                seed_base=args.seed * 1000 + it * 100,
                target_return=args.target_return,
                context_len=args.context_len,
                use_car_rollout=bool(args.rollout_use_car),
                device=device,
            )

            build_mixed_dataset(
                base_dataset_path=base_dataset,
                model_dataset_path=model_data_path,
                out_path=mixed_data_path,
                total_episodes=args.mix_total_episodes,
                mix_ratio=args.mix_ratio,
                seed=args.seed + it,
            )

            run_train_phase(
                dataset_path=mixed_data_path,
                log_path=mix_log,
                epochs=args.mix_epochs,
                init_ckpt=current_ckpt,
                target_return=args.target_return,
                use_car_module=train_use_car,
            )
            _copy_latest_best(mix_ckpt)
            current_ckpt = mix_ckpt

        final_ckpt = work_dir / "model_final_best.pth"
        shutil.copy2(current_ckpt, final_ckpt)
        print("\n=== Done ===")
        print(f"Final checkpoint: {final_ckpt}")
        print(f"Logs/datasets under: {work_dir}")
    finally:
        if not args.keep_root_ckpt:
            _restore_root_checkpoints(root_ckpt_backup_dir)


if __name__ == "__main__":
    main()
