import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import math
import os
import pandas as pd
from DT import DecisionTransformer
from env import ChargingEnv
from generate_DT_dataset import get_state_vector
from car_module import apply_constraint_aware_reranking
from config import *

CONTEXT_LEN = 50
BATCH_SIZE = 64
EPOCHS = 150
MAX_LR = 2e-4  # peak learning rate
WARMUP_STEPS = 300  # warmup steps
WEIGHT_DECAY = 1e-4
ACTIVE_LOSS_WEIGHT = 2.0
ACTIVE_MOVE_THRESH_DEG = 1e-6
AUX_REWARD_LOSS_WEIGHT = 0.05
EVAL_TARGET_RETURN = 200300.0
EVAL_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
EVAL_EVERY_EPOCHS = 1
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_EPOCHS = 20
EARLY_STOP_WINDOW = 5
EARLY_STOP_SUCCESS_DELTA = 1e-4
EARLY_STOP_VAL_LOSS_DELTA = 1e-4
BUSINESS_SUCCESS_TOL = 1e-6
BUSINESS_WAIT_TOL = 1e-6
BUSINESS_MIN_SUCCESS_RATE = float(CONFIG.get("business_min_success_rate", 93.0))

TRAIN_LOG_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_aux_reward_loss",
    "val_aux_reward_loss",
    "dist_error_meters",
    "learning_rate",
    "active_dist_error_meters",
    "success_rate",
    "success_rate_std",
    "avg_wait_steps",
    "avg_wait_steps_std",
    "eval_num_seeds",
]


class ExpertDataset(Dataset):
    def __init__(self, pkl_path, context_len):
        with open(pkl_path, 'rb') as f:
            self.trajectories = pickle.load(f)
        self.context_len = context_len
        self.state_dim = self.trajectories[0]['observations'].shape[1]
        self.action_dim = np.prod(self.trajectories[0]['actions'].shape[1:])

        all_states = np.concatenate([traj['observations'] for traj in self.trajectories])
        all_actions = np.concatenate([traj['actions'].reshape(-1, self.action_dim) for traj in self.trajectories])
        all_rtgs = np.concatenate([traj['returns_to_go'] for traj in self.trajectories])
        all_rewards = np.concatenate([
            np.asarray(traj.get('rewards', np.zeros(len(traj['observations']), dtype=np.float32))).reshape(-1)
            for traj in self.trajectories
        ])

        self.state_mean, self.state_std = all_states.mean(0), all_states.std(0) + 1e-6
        self.action_mean, self.action_std = all_actions.mean(0), all_actions.std(0) + 1e-6
        self.rtg_scale = np.max(np.abs(all_rtgs)) + 1e-6
        self.reward_mean, self.reward_std = all_rewards.mean(), all_rewards.std() + 1e-6

        with open("scaler_params.pkl", "wb") as f:
            pickle.dump({
                'state_mean': self.state_mean,
                'state_std': self.state_std,
                'action_mean': self.action_mean,
                'action_std': self.action_std,
                'rtg_scale': self.rtg_scale,
                'reward_mean': self.reward_mean,
                'reward_std': self.reward_std
            }, f)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        seq_len = len(traj['observations'])
        start_idx = np.random.randint(0, seq_len - self.context_len) if seq_len > self.context_len else 0
        end_idx = start_idx + self.context_len

        s = traj['observations'][start_idx:end_idx]
        a = traj['actions'][start_idx:end_idx].reshape(-1, self.action_dim)
        rtg = traj['returns_to_go'][start_idx:end_idx].reshape(-1, 1)
        r = np.asarray(traj.get('rewards', np.zeros(seq_len, dtype=np.float32)), dtype=np.float32)[start_idx:end_idx].reshape(-1, 1)
        timesteps = np.arange(start_idx, start_idx + len(s))

        s = (s - self.state_mean) / self.state_std
        a = (a - self.action_mean) / self.action_std
        rtg = rtg / self.rtg_scale
        r = (r - self.reward_mean) / self.reward_std

        pad_len = self.context_len - len(s)
        if pad_len > 0:
            s = np.pad(s, ((pad_len, 0), (0, 0)), mode='constant')
            a = np.pad(a, ((pad_len, 0), (0, 0)), mode='constant')
            rtg = np.pad(rtg, ((pad_len, 0), (0, 0)), mode='constant')
            r = np.pad(r, ((pad_len, 0), (0, 0)), mode='constant')
            timesteps = np.pad(timesteps, (pad_len, 0), mode='constant')
            mask = np.concatenate([np.zeros(pad_len), np.ones(self.context_len - pad_len)])
        else:
            mask = np.ones(self.context_len)

        return (torch.FloatTensor(s), torch.FloatTensor(a), torch.FloatTensor(rtg), torch.FloatTensor(r),
                torch.LongTensor(timesteps), torch.BoolTensor(mask))


def _coord_error_meters(pred_coords, target_coords):
    """
    pred_coords/target_coords: [..., 2] in (lat, lon) degrees
    Return: [...], distance in meters
    """
    dlat_m = (pred_coords[..., 0] - target_coords[..., 0]) * 111000.0
    mean_lat_rad = torch.deg2rad((pred_coords[..., 0] + target_coords[..., 0]) * 0.5)
    dlon_m = (pred_coords[..., 1] - target_coords[..., 1]) * 111000.0 * torch.cos(mean_lat_rad)
    return torch.sqrt(dlat_m * dlat_m + dlon_m * dlon_m + 1e-12)


def _reshape_actions(action_tensor):
    mcs_num = action_tensor.shape[-1] // 2
    return action_tensor.view(action_tensor.shape[0], action_tensor.shape[1], mcs_num, 2)


def compute_active_dist_error(pred_action, target_action, a_mean, a_std, seq_mask=None, move_thresh_deg=1e-6):
    """Compute per-MCS distance error on active (moving target) timesteps only."""
    pred_real = pred_action * a_std + a_mean
    target_real = target_action * a_std + a_mean

    p = _reshape_actions(pred_real)
    t = _reshape_actions(target_real)

    # Keep first step aligned to itself to avoid roll wrap-around artifacts.
    prev_t = torch.cat([t[:, :1], t[:, :-1]], dim=1)
    movement = torch.norm(t - prev_t, dim=-1)
    active_mask = movement > move_thresh_deg

    if seq_mask is not None:
        active_mask = active_mask & seq_mask.unsqueeze(-1).bool()

    dist_err_m = _coord_error_meters(p, t)
    active_errors = dist_err_m[active_mask]
    if active_errors.numel() > 0:
        return active_errors.mean().item()
    return None


def compute_mean_dist_error(pred_action, target_action, a_mean, a_std, seq_mask):
    """Compute per-MCS mean distance error (meters) over valid sequence tokens."""
    pred_real = pred_action * a_std + a_mean
    target_real = target_action * a_std + a_mean
    p = _reshape_actions(pred_real)
    t = _reshape_actions(target_real)
    dist_err_m = _coord_error_meters(p, t)
    valid = seq_mask.unsqueeze(-1).bool().expand_as(dist_err_m)
    valid_err = dist_err_m[valid]
    if valid_err.numel() > 0:
        return valid_err.mean().item()
    return 0.0


def compute_weighted_action_loss(loss_elem, target_action, seq_mask, a_mean, a_std,
                                 active_weight=ACTIVE_LOSS_WEIGHT, move_thresh_deg=ACTIVE_MOVE_THRESH_DEG):
    """
    Increase supervision weight on moving MCS targets to reduce active distance error.
    loss_elem: [B, K, action_dim], element-wise L1/L2 loss
    """
    loss4 = _reshape_actions(loss_elem)  # [B, K, M, 2]
    target_real = target_action * a_std + a_mean
    t = _reshape_actions(target_real)
    prev_t = torch.cat([t[:, :1], t[:, :-1]], dim=1)
    movement = torch.norm(t - prev_t, dim=-1)
    active = (movement > move_thresh_deg).unsqueeze(-1)

    weights = torch.ones_like(loss4)
    weights = weights + active_weight * active.float()

    valid = seq_mask.unsqueeze(-1).unsqueeze(-1).bool().expand_as(loss4)
    weighted_loss = (loss4 * weights)[valid]
    weighted_norm = weights[valid].sum().clamp_min(1.0)
    return weighted_loss.sum() / weighted_norm


def compute_masked_regression_loss(loss_elem, seq_mask):
    """Mean regression loss on valid (non-padding) timesteps."""
    valid = seq_mask.unsqueeze(-1).bool().expand_as(loss_elem)
    valid_loss = loss_elem[valid]
    if valid_loss.numel() == 0:
        return torch.tensor(0.0, device=loss_elem.device)
    return valid_loss.mean()


def configure_optimizers(model, learning_rate, weight_decay):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim.AdamW(optim_groups, lr=learning_rate)


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_rollout_metrics(model, env, s_mean, s_std, a_mean, a_std, rtg_scale,
                             device, context_len=CONTEXT_LEN, target_return=EVAL_TARGET_RETURN):
    """
    Run one deterministic rollout and return success/waiting metrics.
    """
    env.reset()
    states_hist, actions_hist, rtgs_hist, timesteps_hist = [], [], [], []
    current_rtg = target_return
    final_info = {}

    wait_start_step = {}
    all_wait_durations = []
    last_step = -1

    model.eval()
    with torch.no_grad():
        for t in range(env.cfg["max_steps"]):
            last_step = t
            raw_state = get_state_vector(env)
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
            attn_mask = np.concatenate([np.zeros(pad_len), np.ones(len(s_input))]) if pad_len > 0 else np.ones(context_len)
            if pad_len > 0:
                s_input = np.pad(s_input, ((pad_len, 0), (0, 0)), mode='constant')
                a_input = np.pad(a_input, ((pad_len, 0), (0, 0)), mode='constant')
                rtg_input = np.pad(rtg_input, ((pad_len, 0), (0, 0)), mode='constant')
                t_input = np.pad(t_input, (pad_len, 0), mode='constant')

            s_tensor = torch.FloatTensor(s_input).unsqueeze(0).to(device)
            a_tensor = torch.FloatTensor(a_input).unsqueeze(0).to(device)
            rtg_tensor = torch.FloatTensor(rtg_input).unsqueeze(0).to(device)
            t_tensor = torch.LongTensor(t_input).unsqueeze(0).to(device)
            m_tensor = torch.BoolTensor(attn_mask).unsqueeze(0).to(device)

            action_preds = model(s_tensor, a_tensor, rtg_tensor, t_tensor, attention_mask=m_tensor)
            pred_action_norm = action_preds[0, -1].cpu().numpy()
            actions_hist[-1] = pred_action_norm
            real_action_matrix = (pred_action_norm * a_std + a_mean).reshape(env.cfg["mcs_num"], 2)
            real_action_matrix = apply_constraint_aware_reranking(env, real_action_matrix)

            prev_states = {ev.id: ev.state for ev in env.evs.values()}

            _, reward, done, info = env.step(real_action_matrix)
            final_info = info
            current_rtg -= reward

            for ev in env.evs.values():
                prev_state = prev_states.get(ev.id)
                curr_state = ev.state
                curr_source = ev.charging_source

                if curr_state == "WAITING":
                    if prev_state != "WAITING":
                        wait_start_step[ev.id] = t
                    continue

                # MOVING -> WAITING -> CHARGING can occur within one env step.
                # Count this as one full waiting episode with 0-step delay.
                immediate_wait_then_service = (
                    prev_state == "MOVING" and
                    curr_state == "CHARGING" and
                    curr_source in {"MCS", "FCS"}
                )
                if immediate_wait_then_service:
                    all_wait_durations.append(0)
                    wait_start_step.pop(ev.id, None)
                    continue

                if ev.id not in wait_start_step:
                    continue

                start_step = wait_start_step[ev.id]
                got_service_after_waiting = (
                    curr_state == "MOVING_TO_FCS" or
                    (curr_state == "CHARGING" and curr_source in {"MCS", "FCS"})
                )
                if got_service_after_waiting:
                    all_wait_durations.append(max(0, t - start_step))
                    wait_start_step.pop(ev.id, None)
                    continue

                # Full-wait metric: when a waiting EV exits WAITING without service
                # (timeout/offline/episode dynamics), still close this waiting episode.
                if prev_state == "WAITING":
                    all_wait_durations.append(max(0, t - start_step))
                    wait_start_step.pop(ev.id, None)

            if done:
                break

    # Include unfinished waiting episodes at rollout end in full-wait metric.
    if wait_start_step and last_step >= 0:
        episode_end_step = last_step + 1
        for start_step in wait_start_step.values():
            all_wait_durations.append(max(0, episode_end_step - start_step))

    avg_wait_steps_all = float(np.mean(all_wait_durations)) if all_wait_durations else 0.0
    return {
        "success_rate": float(final_info.get("success_rate", env._calculate_success_rate())),
        # Full-wait metric: all WAITING episodes, ended by service/offline/rollout-end.
        "avg_wait_steps": avg_wait_steps_all
    }


def evaluate_multi_seed_metrics(model, cfg, seed_list, s_mean, s_std, a_mean, a_std, rtg_scale,
                                device, context_len=CONTEXT_LEN, target_return=EVAL_TARGET_RETURN):
    """
    Evaluate rollout metrics on multiple seeds and aggregate mean/std.
    """
    success_list = []
    wait_steps_list = []
    eval_cfg = dict(cfg)
    eval_cfg["verbose_dataset_load"] = False
    for seed in seed_list:
        env = ChargingEnv(eval_cfg)
        env.seed(seed)
        m = evaluate_rollout_metrics(
            model=model,
            env=env,
            s_mean=s_mean,
            s_std=s_std,
            a_mean=a_mean,
            a_std=a_std,
            rtg_scale=rtg_scale,
            device=device,
            context_len=context_len,
            target_return=target_return
        )
        success_list.append(m["success_rate"])
        wait_steps_list.append(m["avg_wait_steps"])

    return {
        "success_rate": float(np.mean(success_list)),
        "success_rate_std": float(np.std(success_list)),
        "avg_wait_steps": float(np.mean(wait_steps_list)),
        "avg_wait_steps_std": float(np.std(wait_steps_list)),
        "num_seeds": int(len(seed_list))
    }


def _init_history(log_path, append_log):
    history = {k: [] for k in TRAIN_LOG_COLUMNS}
    epoch_offset = 0

    if (not append_log) or (not os.path.exists(log_path)):
        return history, epoch_offset

    try:
        old_df = pd.read_csv(log_path)
    except Exception:
        return history, epoch_offset

    if old_df.empty:
        return history, epoch_offset

    if "epoch" in old_df.columns:
        epoch_num = pd.to_numeric(old_df["epoch"], errors="coerce").dropna()
        epoch_offset = int(epoch_num.max()) if len(epoch_num) > 0 else len(old_df)
    else:
        epoch_offset = len(old_df)

    row_num = len(old_df)
    for col in TRAIN_LOG_COLUMNS:
        if col in old_df.columns:
            history[col] = old_df[col].tolist()
        else:
            history[col] = [np.nan] * row_num

    return history, epoch_offset


def train(
    dataset_path="expert_dataset.pkl",
    init_ckpt=None,
    log_path="train_log_v2.csv",
    append_log=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    history, epoch_offset = _init_history(log_path, append_log)


    full_dataset = ExpertDataset(dataset_path, CONTEXT_LEN)

    a_mean = torch.from_numpy(full_dataset.action_mean).to(device)
    a_std = torch.from_numpy(full_dataset.action_std).to(device)
    a_mean_np = full_dataset.action_mean
    a_std_np = full_dataset.action_std
    s_mean_np = full_dataset.state_mean
    s_std_np = full_dataset.state_std
    rtg_scale = full_dataset.rtg_scale

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = DecisionTransformer(
        state_dim=full_dataset.state_dim, action_dim=full_dataset.action_dim, max_length=200
    ).to(device)
    if init_ckpt is not None and os.path.exists(init_ckpt):
        ckpt = torch.load(init_ckpt, map_location=device, weights_only=True)
        load_res = model.load_state_dict(ckpt, strict=False)
        if load_res.missing_keys or load_res.unexpected_keys:
            print("Warning: init checkpoint partially loaded due to architecture mismatch.")

    optimizer = configure_optimizers(model, MAX_LR, WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    # loss_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.L1Loss(reduction='none')
    best_val_loss = float('inf')
    best_success_rate = -float('inf')
    best_avg_wait_steps = float('inf')
    best_business_epoch = -1
    best_business_global_epoch = -1
    best_success_ckpt_rate = -float('inf')
    best_success_ckpt_wait = float('inf')
    best_success_ckpt_epoch = -1
    best_success_ckpt_global_epoch = -1
    early_stop_counter = 0
    success_window = []
    val_loss_window = []
    best_window_success = -float('inf')
    best_window_val_loss = float('inf')
    effective_early_stop_min_epochs = min(
        EARLY_STOP_MIN_EPOCHS, max(EARLY_STOP_WINDOW, EPOCHS // 3)
    )
    last_eval_metrics = {
        "success_rate": np.nan,
        "success_rate_std": np.nan,
        "avg_wait_steps": np.nan,
        "avg_wait_steps_std": np.nan,
        "num_seeds": 0
    }

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        train_aux_reward_loss = 0.0

        for states, actions, rtgs, rewards, timesteps, masks in train_loader:
            states, actions, rtgs, rewards = states.to(device), actions.to(device), rtgs.to(device), rewards.to(device)
            timesteps, masks = timesteps.to(device), masks.to(device)

            action_preds, reward_preds = model(
                states, actions, rtgs, timesteps, attention_mask=masks, return_aux=True
            )

            action_loss_elem = loss_fn(action_preds, actions)
            action_loss = compute_weighted_action_loss(action_loss_elem, actions, masks, a_mean, a_std)
            reward_loss_elem = loss_fn(reward_preds, rewards)
            reward_loss = compute_masked_regression_loss(reward_loss_elem, masks)
            loss = action_loss + AUX_REWARD_LOSS_WEIGHT * reward_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_aux_reward_loss += reward_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_aux_reward_loss = train_aux_reward_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # 4. Validation
        active_val_errors = []
        model.eval()
        val_dist_errors = []
        val_loss = 0.0
        val_aux_reward_loss = 0.0
        with torch.no_grad():
            for states, actions, rtgs, rewards, timesteps, masks in val_loader:
                states, actions, rtgs, rewards = states.to(device), actions.to(device), rtgs.to(device), rewards.to(device)
                timesteps, masks = timesteps.to(device), masks.to(device)

                action_preds, reward_preds = model(
                    states, actions, rtgs, timesteps, attention_mask=masks, return_aux=True
                )
                error_m = compute_active_dist_error(action_preds, actions, a_mean, a_std, seq_mask=masks)
                if error_m is not None:
                    active_val_errors.append(error_m)

                action_loss_elem = loss_fn(action_preds, actions)
                action_loss = compute_weighted_action_loss(action_loss_elem, actions, masks, a_mean, a_std)
                reward_loss_elem = loss_fn(reward_preds, rewards)
                reward_loss = compute_masked_regression_loss(reward_loss_elem, masks)
                loss = action_loss + AUX_REWARD_LOSS_WEIGHT * reward_loss
                val_loss += loss.item()
                val_aux_reward_loss += reward_loss.item()

                val_dist_errors.append(
                    compute_mean_dist_error(action_preds, actions, a_mean, a_std, masks)
                )

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_aux_reward_loss = val_aux_reward_loss / max(1, len(val_loader))

        avg_dist_m = np.mean(val_dist_errors)
        global_epoch = epoch_offset + epoch + 1
        history['epoch'].append(global_epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_aux_reward_loss'].append(avg_train_aux_reward_loss)
        history['val_aux_reward_loss'].append(avg_val_aux_reward_loss)
        history['dist_error_meters'].append(avg_dist_m)
        history['learning_rate'].append(current_lr)

        # Record epoch-level metrics
        avg_active_dist = np.mean(active_val_errors)
        history['active_dist_error_meters'].append(avg_active_dist)

        do_eval = ((epoch + 1) % EVAL_EVERY_EPOCHS == 0)
        if do_eval:
            last_eval_metrics = evaluate_multi_seed_metrics(
                model=model,
                cfg=CONFIG,
                seed_list=EVAL_SEEDS,
                s_mean=s_mean_np,
                s_std=s_std_np,
                a_mean=a_mean_np,
                a_std=a_std_np,
                rtg_scale=rtg_scale,
                device=device
            )

        history['success_rate'].append(last_eval_metrics['success_rate'])
        history['success_rate_std'].append(last_eval_metrics['success_rate_std'])
        history['avg_wait_steps'].append(last_eval_metrics['avg_wait_steps'])
        history['avg_wait_steps_std'].append(last_eval_metrics['avg_wait_steps_std'])
        history['eval_num_seeds'].append(last_eval_metrics['num_seeds'])

        # Refresh CSV each epoch
        pd.DataFrame(history).to_csv(log_path, index=False)

        # Logging
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} (global {global_epoch:03d}) | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"AuxR: {avg_train_aux_reward_loss:.4f}/{avg_val_aux_reward_loss:.4f} | "
            f"Success: {last_eval_metrics['success_rate']:.2f}\u00b1{last_eval_metrics['success_rate_std']:.2f}% | "
            f"AvgWait: {last_eval_metrics['avg_wait_steps']:.1f}\u00b1{last_eval_metrics['avg_wait_steps_std']:.1f} steps "
            f"(seeds={last_eval_metrics['num_seeds']})"
        )

        # Keep val-loss best checkpoint for debugging/reference
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "dt_mcs_best_val.pth")

        # Business checkpoint:
        # 1) before reaching min success floor, optimize success first;
        # 2) once success floor reached, prioritize lower wait, then success.
        curr_success = last_eval_metrics['success_rate']
        curr_wait = last_eval_metrics['avg_wait_steps']

        # Success-first checkpoint: maximize success rate, then break ties with lower wait.
        success_ckpt_improved = (
            (curr_success > best_success_ckpt_rate + BUSINESS_SUCCESS_TOL) or
            (
                abs(curr_success - best_success_ckpt_rate) <= BUSINESS_SUCCESS_TOL and
                curr_wait < best_success_ckpt_wait - BUSINESS_WAIT_TOL
            )
        )
        if success_ckpt_improved:
            best_success_ckpt_rate = curr_success
            best_success_ckpt_wait = curr_wait
            best_success_ckpt_epoch = epoch + 1
            best_success_ckpt_global_epoch = global_epoch
            torch.save(model.state_dict(), "dt_mcs_best_success.pth")

        best_above_floor = (best_success_rate >= BUSINESS_MIN_SUCCESS_RATE)
        curr_above_floor = (curr_success >= BUSINESS_MIN_SUCCESS_RATE)

        if not best_above_floor and not curr_above_floor:
            business_improved = (
                (curr_success > best_success_rate + BUSINESS_SUCCESS_TOL) or
                (
                    abs(curr_success - best_success_rate) <= BUSINESS_SUCCESS_TOL and
                    curr_wait < best_avg_wait_steps - BUSINESS_WAIT_TOL
                )
            )
        elif curr_above_floor and not best_above_floor:
            business_improved = True
        elif curr_above_floor and best_above_floor:
            business_improved = (
                (curr_wait < best_avg_wait_steps - BUSINESS_WAIT_TOL) or
                (
                    abs(curr_wait - best_avg_wait_steps) <= BUSINESS_WAIT_TOL and
                    curr_success > best_success_rate + BUSINESS_SUCCESS_TOL
                )
            )
        else:
            business_improved = False

        if business_improved:
            best_success_rate = last_eval_metrics['success_rate']
            best_avg_wait_steps = last_eval_metrics['avg_wait_steps']
            best_business_epoch = epoch + 1
            best_business_global_epoch = global_epoch
            # Main checkpoint used by evaluate.py
            torch.save(model.state_dict(), "dt_mcs_best.pth")
            torch.save(model.state_dict(), "dt_mcs_best_business.pth")

        # Early stop by dual-window trend:
        # stop only when both success window and val-loss window stop improving.
        if do_eval:
            success_window.append(curr_success)
            val_loss_window.append(avg_val_loss)
            if len(success_window) > EARLY_STOP_WINDOW:
                success_window.pop(0)
            if len(val_loss_window) > EARLY_STOP_WINDOW:
                val_loss_window.pop(0)
            if len(success_window) == EARLY_STOP_WINDOW:
                window_success = float(np.mean(success_window))
                window_val_loss = float(np.mean(val_loss_window))
                success_improved = window_success > best_window_success + EARLY_STOP_SUCCESS_DELTA
                val_loss_improved = window_val_loss < best_window_val_loss - EARLY_STOP_VAL_LOSS_DELTA

                if success_improved:
                    best_window_success = window_success
                if val_loss_improved:
                    best_window_val_loss = window_val_loss

                if success_improved or val_loss_improved:
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if (
            do_eval and
            (epoch + 1) >= effective_early_stop_min_epochs and
            len(success_window) == EARLY_STOP_WINDOW and
            early_stop_counter >= EARLY_STOP_PATIENCE
        ):
            print(
                f"Early stop at epoch {epoch + 1} (global {global_epoch}): "
                f"window({EARLY_STOP_WINDOW}) success and val-loss both no improvement for "
                f"{EARLY_STOP_PATIENCE} eval rounds."
            )
            break

    print(
        f"Training done. Best val loss: {best_val_loss:.4f} | "
        f"Best business epoch: {best_business_epoch} (global {best_business_global_epoch}) | "
        f"Best success-only epoch: {best_success_ckpt_epoch} (global {best_success_ckpt_global_epoch}) | "
        f"Business success floor: {BUSINESS_MIN_SUCCESS_RATE:.1f}% | "
        f"Best success: {best_success_rate:.2f}% | "
        f"Best success-only: {best_success_ckpt_rate:.2f}% | "
        f"Best window({EARLY_STOP_WINDOW}) success mean: "
        f"{(best_window_success if best_window_success > -float('inf') else float('nan')):.2f}% | "
        f"Best window({EARLY_STOP_WINDOW}) val-loss mean: "
        f"{(best_window_val_loss if best_window_val_loss < float('inf') else float('nan')):.4f} | "
        f"ES min epochs: {effective_early_stop_min_epochs} | "
        f"Best avg wait(all waiting): {best_avg_wait_steps:.2f} steps "
        f"({best_avg_wait_steps * CONFIG.get('minutes_per_step', 24.0 * 60.0 / max(1, CONFIG.get('max_steps', 200))):.2f} min)"
    )


if __name__ == "__main__":
    train()

