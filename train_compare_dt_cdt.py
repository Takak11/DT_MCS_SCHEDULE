import argparse
import math
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from CDT import ConstrainedDecisionTransformer
from DT import DecisionTransformer
from car_module import apply_constraint_aware_reranking
from config import CONFIG
from env import ChargingEnv
from generate_DT_dataset import assignment_memory, expert_get_action_with_commitment, get_state_vector


CONTEXT_LEN = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 120
MAX_LR = 2e-4
WARMUP_STEPS = 300
WEIGHT_DECAY = 1e-4
ACTIVE_LOSS_WEIGHT = 2.0
ACTIVE_MOVE_THRESH_DEG = 1e-6
AUX_REWARD_LOSS_WEIGHT = 0.05
AUX_COST_LOSS_WEIGHT = 0.10
EVAL_TARGET_RETURN = 182500.0
DEFAULT_EVAL_SEEDS = [42,43,44,45,46,47,48,49,50,51]
EVAL_EVERY_EPOCHS = 1

LOG_COLUMNS = [
    "epoch",
    "variant",
    "train_loss",
    "val_loss",
    "train_aux_reward_loss",
    "val_aux_reward_loss",
    "train_aux_cost_loss",
    "val_aux_cost_loss",
    "dist_error_meters",
    "active_dist_error_meters",
    "learning_rate",
    "success_rate",
    "success_rate_std",
    "avg_wait_steps",
    "avg_wait_steps_std",
    "eval_num_seeds",
]


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_eval_seeds(seed_text):
    if seed_text is None or seed_text.strip() == "":
        return list(DEFAULT_EVAL_SEEDS)
    return [int(x.strip()) for x in seed_text.split(",") if x.strip() != ""]


def _coord_error_meters(pred_coords, target_coords):
    dlat_m = (pred_coords[..., 0] - target_coords[..., 0]) * 111000.0
    mean_lat_rad = torch.deg2rad((pred_coords[..., 0] + target_coords[..., 0]) * 0.5)
    dlon_m = (pred_coords[..., 1] - target_coords[..., 1]) * 111000.0 * torch.cos(mean_lat_rad)
    return torch.sqrt(dlat_m * dlat_m + dlon_m * dlon_m + 1e-12)


def _reshape_actions(action_tensor):
    mcs_num = action_tensor.shape[-1] // 2
    return action_tensor.view(action_tensor.shape[0], action_tensor.shape[1], mcs_num, 2)


def compute_active_dist_error(pred_action, target_action, a_mean, a_std, seq_mask=None):
    pred_real = pred_action * a_std + a_mean
    target_real = target_action * a_std + a_mean
    p = _reshape_actions(pred_real)
    t = _reshape_actions(target_real)
    prev_t = torch.cat([t[:, :1], t[:, :-1]], dim=1)
    movement = torch.norm(t - prev_t, dim=-1)
    active_mask = movement > ACTIVE_MOVE_THRESH_DEG
    if seq_mask is not None:
        active_mask = active_mask & seq_mask.unsqueeze(-1).bool()
    dist_err_m = _coord_error_meters(p, t)
    active_errors = dist_err_m[active_mask]
    if active_errors.numel() > 0:
        return active_errors.mean().item()
    return None


def compute_mean_dist_error(pred_action, target_action, a_mean, a_std, seq_mask):
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


def compute_weighted_action_loss(loss_elem, target_action, seq_mask, a_mean, a_std):
    loss4 = _reshape_actions(loss_elem)
    target_real = target_action * a_std + a_mean
    t = _reshape_actions(target_real)
    prev_t = torch.cat([t[:, :1], t[:, :-1]], dim=1)
    movement = torch.norm(t - prev_t, dim=-1)
    active = (movement > ACTIVE_MOVE_THRESH_DEG).unsqueeze(-1)

    weights = torch.ones_like(loss4)
    weights = weights + ACTIVE_LOSS_WEIGHT * active.float()

    valid = seq_mask.unsqueeze(-1).unsqueeze(-1).bool().expand_as(loss4)
    weighted_loss = (loss4 * weights)[valid]
    weighted_norm = weights[valid].sum().clamp_min(1.0)
    return weighted_loss.sum() / weighted_norm


def compute_masked_regression_loss(loss_elem, seq_mask):
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
        for pn, _ in m.named_parameters():
            fpn = ("%s.%s" % (mn, pn)) if mn else pn
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
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


class ExpertConstraintDataset(Dataset):
    def __init__(self, pkl_path, context_len, scaler_save_path):
        with open(pkl_path, "rb") as f:
            self.trajectories = pickle.load(f)

        self.context_len = int(context_len)
        self.state_dim = int(self.trajectories[0]["observations"].shape[1])
        self.action_dim = int(np.prod(self.trajectories[0]["actions"].shape[1:]))
        self.mcs_num = int(CONFIG["mcs_num"])
        ev_feature_dim = self.state_dim - self.mcs_num * 2
        self.max_waiting_slots = max(1, ev_feature_dim // 3)
        self.constraint_dim = 1

        all_states = []
        all_actions = []
        all_rtgs = []
        all_rewards = []
        all_costs = []
        all_ctgs = []
        self.init_ctg_values = []

        for traj in self.trajectories:
            obs = np.asarray(traj["observations"], dtype=np.float32)
            actions = np.asarray(traj["actions"], dtype=np.float32).reshape(-1, self.action_dim)
            rtg = np.asarray(traj["returns_to_go"], dtype=np.float32).reshape(-1)
            rewards = np.asarray(
                traj.get("rewards", np.zeros(len(obs), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)

            costs = self._build_constraint_cost(obs)
            ctg = self._to_go(costs)
            traj["_costs"] = costs.astype(np.float32)
            traj["_ctg"] = ctg.astype(np.float32)

            all_states.append(obs)
            all_actions.append(actions)
            all_rtgs.append(rtg)
            all_rewards.append(rewards)
            all_costs.append(costs)
            all_ctgs.append(ctg)
            self.init_ctg_values.append(float(ctg[0] if len(ctg) > 0 else 0.0))

        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_rtgs = np.concatenate(all_rtgs, axis=0)
        all_rewards = np.concatenate(all_rewards, axis=0)
        all_costs = np.concatenate(all_costs, axis=0)
        all_ctgs = np.concatenate(all_ctgs, axis=0)

        self.state_mean = all_states.mean(0)
        self.state_std = all_states.std(0) + 1e-6
        self.action_mean = all_actions.mean(0)
        self.action_std = all_actions.std(0) + 1e-6
        self.rtg_scale = float(np.max(np.abs(all_rtgs)) + 1e-6)
        self.reward_mean = float(all_rewards.mean())
        self.reward_std = float(all_rewards.std() + 1e-6)
        self.cost_mean = float(all_costs.mean())
        self.cost_std = float(all_costs.std() + 1e-6)
        self.ctg_scale = float(np.max(np.abs(all_ctgs)) + 1e-6)
        self.eval_target_constraint = float(np.percentile(np.array(self.init_ctg_values), 25))

        with open(scaler_save_path, "wb") as f:
            pickle.dump(
                {
                    "state_mean": self.state_mean,
                    "state_std": self.state_std,
                    "action_mean": self.action_mean,
                    "action_std": self.action_std,
                    "rtg_scale": self.rtg_scale,
                    "reward_mean": self.reward_mean,
                    "reward_std": self.reward_std,
                    "cost_mean": self.cost_mean,
                    "cost_std": self.cost_std,
                    "ctg_scale": self.ctg_scale,
                    "max_waiting_slots": self.max_waiting_slots,
                    "eval_target_constraint": self.eval_target_constraint,
                },
                f,
            )

    @staticmethod
    def _to_go(x):
        out = np.zeros_like(x, dtype=np.float32)
        running = 0.0
        for i in range(len(x) - 1, -1, -1):
            running += float(x[i])
            out[i] = running
        return out

    def _build_constraint_cost(self, obs_seq):
        wait_block = obs_seq[:, self.mcs_num * 2:]
        if wait_block.shape[1] <= 0 or wait_block.shape[1] % 3 != 0:
            return np.zeros(obs_seq.shape[0], dtype=np.float32)

        slots = wait_block.reshape(obs_seq.shape[0], -1, 3)
        coords = slots[:, :, :2]
        soc = slots[:, :, 2]
        valid = (np.abs(coords[:, :, 0]) > 1e-8) | (np.abs(coords[:, :, 1]) > 1e-8)
        waiting_count = valid.sum(axis=1).astype(np.float32)
        low_soc = ((soc < float(CONFIG.get("ev_request_threshold", 0.2))) & valid).sum(axis=1).astype(np.float32)
        denom = float(max(1, slots.shape[1]))
        queue_ratio = waiting_count / denom
        low_soc_ratio = low_soc / denom
        return queue_ratio + 0.5 * low_soc_ratio

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        seq_len = len(traj["observations"])
        start_idx = np.random.randint(0, seq_len - self.context_len) if seq_len > self.context_len else 0
        end_idx = start_idx + self.context_len

        s = np.asarray(traj["observations"][start_idx:end_idx], dtype=np.float32)
        a = np.asarray(traj["actions"][start_idx:end_idx], dtype=np.float32).reshape(-1, self.action_dim)
        rtg = np.asarray(traj["returns_to_go"][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)
        r = np.asarray(
            traj.get("rewards", np.zeros(seq_len, dtype=np.float32))[start_idx:end_idx],
            dtype=np.float32,
        ).reshape(-1, 1)
        c = np.asarray(traj["_costs"][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)
        ctg = np.asarray(traj["_ctg"][start_idx:end_idx], dtype=np.float32).reshape(-1, 1)
        timesteps = np.arange(start_idx, start_idx + len(s))

        s = (s - self.state_mean) / self.state_std
        a = (a - self.action_mean) / self.action_std
        rtg = rtg / self.rtg_scale
        r = (r - self.reward_mean) / self.reward_std
        c = (c - self.cost_mean) / self.cost_std
        ctg = ctg / self.ctg_scale

        pad_len = self.context_len - len(s)
        if pad_len > 0:
            s = np.pad(s, ((pad_len, 0), (0, 0)), mode="constant")
            a = np.pad(a, ((pad_len, 0), (0, 0)), mode="constant")
            rtg = np.pad(rtg, ((pad_len, 0), (0, 0)), mode="constant")
            r = np.pad(r, ((pad_len, 0), (0, 0)), mode="constant")
            c = np.pad(c, ((pad_len, 0), (0, 0)), mode="constant")
            ctg = np.pad(ctg, ((pad_len, 0), (0, 0)), mode="constant")
            timesteps = np.pad(timesteps, (pad_len, 0), mode="constant")
            mask = np.concatenate([np.zeros(pad_len), np.ones(self.context_len - pad_len)])
        else:
            mask = np.ones(self.context_len)

        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(rtg),
            torch.FloatTensor(r),
            torch.FloatTensor(c),
            torch.FloatTensor(ctg),
            torch.LongTensor(timesteps),
            torch.BoolTensor(mask),
        )


def _online_step_constraint_cost(info, max_waiting_slots, prev_dead_count):
    waiting_count = float(info.get("waiting_count", 0))
    dead_count = int(info.get("dead_count", 0))
    dead_increase = max(0, dead_count - prev_dead_count)
    queue_cost = waiting_count / float(max(1, max_waiting_slots))
    dead_cost = 2.0 * float(dead_increase)
    return queue_cost + dead_cost, dead_count


def evaluate_rollout_metrics(
    model,
    model_kind,
    env,
    s_mean,
    s_std,
    a_mean,
    a_std,
    rtg_scale,
    ctg_scale,
    target_return,
    target_constraint,
    max_waiting_slots,
    device,
    use_car=False,
):
    env.reset()
    states_hist, actions_hist, rtgs_hist, timesteps_hist = [], [], [], []
    ctgs_hist = []
    current_rtg = float(target_return)
    current_ctg = float(target_constraint)
    final_info = {}
    prev_dead_count = 0

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
            rtgs_hist.append([current_rtg / max(1e-6, rtg_scale)])
            timesteps_hist.append(t)
            if model_kind == "cdt":
                ctgs_hist.append([current_ctg / max(1e-6, ctg_scale)])

            if len(actions_hist) < len(states_hist):
                actions_hist.append(np.zeros(len(a_mean), dtype=np.float32))

            s_input = np.array(states_hist[-CONTEXT_LEN:])
            a_input = np.array(actions_hist[-CONTEXT_LEN:])
            rtg_input = np.array(rtgs_hist[-CONTEXT_LEN:])
            t_input = np.array(timesteps_hist[-CONTEXT_LEN:])
            ctg_input = np.array(ctgs_hist[-CONTEXT_LEN:]) if model_kind == "cdt" else None

            pad_len = CONTEXT_LEN - len(s_input)
            attn_mask = (
                np.concatenate([np.zeros(pad_len), np.ones(len(s_input))])
                if pad_len > 0
                else np.ones(CONTEXT_LEN)
            )
            if pad_len > 0:
                s_input = np.pad(s_input, ((pad_len, 0), (0, 0)), mode="constant")
                a_input = np.pad(a_input, ((pad_len, 0), (0, 0)), mode="constant")
                rtg_input = np.pad(rtg_input, ((pad_len, 0), (0, 0)), mode="constant")
                t_input = np.pad(t_input, (pad_len, 0), mode="constant")
                if model_kind == "cdt":
                    ctg_input = np.pad(ctg_input, ((pad_len, 0), (0, 0)), mode="constant")

            s_tensor = torch.FloatTensor(s_input).unsqueeze(0).to(device)
            a_tensor = torch.FloatTensor(a_input).unsqueeze(0).to(device)
            rtg_tensor = torch.FloatTensor(rtg_input).unsqueeze(0).to(device)
            t_tensor = torch.LongTensor(t_input).unsqueeze(0).to(device)
            m_tensor = torch.BoolTensor(attn_mask).unsqueeze(0).to(device)

            if model_kind == "cdt":
                ctg_tensor = torch.FloatTensor(ctg_input).unsqueeze(0).to(device)
                action_preds = model(
                    s_tensor, a_tensor, rtg_tensor, ctg_tensor, t_tensor, attention_mask=m_tensor
                )
            else:
                action_preds = model(s_tensor, a_tensor, rtg_tensor, t_tensor, attention_mask=m_tensor)

            pred_action_norm = action_preds[0, -1].cpu().numpy()
            actions_hist[-1] = pred_action_norm
            real_action_matrix = (pred_action_norm * a_std + a_mean).reshape(env.cfg["mcs_num"], 2)
            if use_car:
                real_action_matrix = apply_constraint_aware_reranking(env, real_action_matrix)

            prev_states = {ev.id: ev.state for ev in env.evs.values()}
            _, reward, done, info = env.step(real_action_matrix)
            final_info = info
            current_rtg -= reward

            if model_kind == "cdt":
                step_cost, prev_dead_count = _online_step_constraint_cost(
                    info, max_waiting_slots=max_waiting_slots, prev_dead_count=prev_dead_count
                )
                current_ctg = max(0.0, current_ctg - step_cost)

            for ev in env.evs.values():
                prev_state = prev_states.get(ev.id)
                curr_state = ev.state
                curr_source = ev.charging_source

                if curr_state == "WAITING":
                    if prev_state != "WAITING":
                        wait_start_step[ev.id] = t
                    continue

                immediate_wait_then_service = (
                    prev_state == "MOVING"
                    and curr_state == "CHARGING"
                    and curr_source in {"MCS", "FCS"}
                )
                if immediate_wait_then_service:
                    all_wait_durations.append(0)
                    wait_start_step.pop(ev.id, None)
                    continue

                if ev.id not in wait_start_step:
                    continue

                start_step = wait_start_step[ev.id]
                got_service_after_waiting = (
                    curr_state == "MOVING_TO_FCS"
                    or (curr_state == "CHARGING" and curr_source in {"MCS", "FCS"})
                )
                if got_service_after_waiting:
                    all_wait_durations.append(max(0, t - start_step))
                    wait_start_step.pop(ev.id, None)
                    continue

                if prev_state == "WAITING":
                    all_wait_durations.append(max(0, t - start_step))
                    wait_start_step.pop(ev.id, None)

            if done:
                break

    if wait_start_step and last_step >= 0:
        episode_end_step = last_step + 1
        for start_step in wait_start_step.values():
            all_wait_durations.append(max(0, episode_end_step - start_step))

    avg_wait_steps_all = float(np.mean(all_wait_durations)) if all_wait_durations else 0.0
    return {
        "success_rate": float(final_info.get("success_rate", env._calculate_success_rate())),
        "avg_wait_steps": avg_wait_steps_all,
    }


def evaluate_multi_seed_metrics(
    model,
    model_kind,
    cfg,
    seed_list,
    s_mean,
    s_std,
    a_mean,
    a_std,
    rtg_scale,
    ctg_scale,
    target_return,
    target_constraint,
    max_waiting_slots,
    device,
    use_car=False,
):
    success_list = []
    wait_steps_list = []
    for seed in seed_list:
        eval_cfg = dict(cfg)
        eval_cfg["verbose_dataset_load"] = False
        eval_cfg["use_car_module"] = bool(use_car)
        env = ChargingEnv(eval_cfg)
        env.seed(seed)
        m = evaluate_rollout_metrics(
            model=model,
            model_kind=model_kind,
            env=env,
            s_mean=s_mean,
            s_std=s_std,
            a_mean=a_mean,
            a_std=a_std,
            rtg_scale=rtg_scale,
            ctg_scale=ctg_scale,
            target_return=target_return,
            target_constraint=target_constraint,
            max_waiting_slots=max_waiting_slots,
            device=device,
            use_car=use_car,
        )
        success_list.append(m["success_rate"])
        wait_steps_list.append(m["avg_wait_steps"])

    return {
        "success_rate": float(np.mean(success_list)),
        "success_rate_std": float(np.std(success_list)),
        "avg_wait_steps": float(np.mean(wait_steps_list)),
        "avg_wait_steps_std": float(np.std(wait_steps_list)),
        "num_seeds": int(len(seed_list)),
    }


def evaluate_expert_once(cfg, seed):
    eval_cfg = dict(cfg)
    eval_cfg["verbose_dataset_load"] = False
    env = ChargingEnv(eval_cfg)
    env.seed(seed)
    assignment_memory.clear()
    env.reset()

    wait_start_step = {}
    all_wait_durations = []
    final_info = {}
    last_step = -1

    for t in range(eval_cfg["max_steps"]):
        last_step = t
        action = expert_get_action_with_commitment(env, epsilon=0.0)
        prev_states = {ev.id: ev.state for ev in env.evs.values()}
        _, _, done, info = env.step(action)
        final_info = info

        for ev in env.evs.values():
            prev_state = prev_states.get(ev.id)
            curr_state = ev.state
            curr_source = ev.charging_source

            if curr_state == "WAITING":
                if prev_state != "WAITING":
                    wait_start_step[ev.id] = t
                continue

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

            if prev_state == "WAITING":
                all_wait_durations.append(max(0, t - start_step))
                wait_start_step.pop(ev.id, None)

        if done:
            break

    if wait_start_step and last_step >= 0:
        episode_end_step = last_step + 1
        for start_step in wait_start_step.values():
            all_wait_durations.append(max(0, episode_end_step - start_step))

    return {
        "success_rate": float(final_info.get("success_rate", env._calculate_success_rate())),
        "avg_wait_steps": float(np.mean(all_wait_durations)) if all_wait_durations else 0.0,
    }


def evaluate_expert_multi_seed(cfg, seed_list):
    success_list = []
    wait_steps_list = []
    if seed_list is None or len(seed_list) == 0:
        seed_list = [42]

    for seed in seed_list:
        m = evaluate_expert_once(cfg, seed)
        success_list.append(m["success_rate"])
        wait_steps_list.append(m["avg_wait_steps"])

    return {
        "success_rate": float(np.mean(success_list)),
        "success_rate_std": float(np.std(success_list)),
        "avg_wait_steps": float(np.mean(wait_steps_list)),
        "avg_wait_steps_std": float(np.std(wait_steps_list)),
        "num_seeds": int(len(seed_list)),
    }


def _build_model(model_kind, state_dim, action_dim, max_length):
    if model_kind == "cdt":
        return ConstrainedDecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_dim=1,
            max_length=max_length,
        )
    if model_kind == "dt":
        return DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_length=max_length,
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def train_one_variant(
    label,
    model_kind,
    use_car,
    dataset_path,
    log_dir,
    epochs,
    batch_size,
    eval_seeds,
    seed,
):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== [{label}] model={model_kind}, use_car={use_car} ===")
    print(f"Device: {device}")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_log_{label}.csv"
    ckpt_path = log_dir / f"best_{label}.pth"
    scaler_path = log_dir / f"scaler_{label}.pkl"

    dataset = ExpertConstraintDataset(
        pkl_path=dataset_path,
        context_len=CONTEXT_LEN,
        scaler_save_path=scaler_path,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_gen)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = _build_model(
        model_kind=model_kind,
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        max_length=CONFIG["max_steps"],
    ).to(device)

    optimizer = configure_optimizers(model, MAX_LR, WEIGHT_DECAY)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = get_lr_scheduler(optimizer, WARMUP_STEPS, total_steps)
    loss_fn = nn.L1Loss(reduction="none")

    a_mean = torch.from_numpy(dataset.action_mean).to(device)
    a_std = torch.from_numpy(dataset.action_std).to(device)

    history = {k: [] for k in LOG_COLUMNS}
    best_success = -float("inf")
    best_wait = float("inf")
    last_eval_metrics = {
        "success_rate": np.nan,
        "success_rate_std": np.nan,
        "avg_wait_steps": np.nan,
        "avg_wait_steps_std": np.nan,
        "num_seeds": 0,
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_aux_reward_loss = 0.0
        train_aux_cost_loss = 0.0

        for states, actions, rtgs, rewards, costs, ctgs, timesteps, masks in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            rewards = rewards.to(device)
            costs = costs.to(device)
            ctgs = ctgs.to(device)
            timesteps = timesteps.to(device)
            masks = masks.to(device)

            if model_kind == "cdt":
                action_preds, reward_preds, cost_preds = model(
                    states, actions, rtgs, ctgs, timesteps, attention_mask=masks, return_aux=True
                )
            else:
                action_preds, reward_preds = model(
                    states, actions, rtgs, timesteps, attention_mask=masks, return_aux=True
                )
                cost_preds = None

            action_loss_elem = loss_fn(action_preds, actions)
            action_loss = compute_weighted_action_loss(action_loss_elem, actions, masks, a_mean, a_std)
            reward_loss_elem = loss_fn(reward_preds, rewards)
            reward_loss = compute_masked_regression_loss(reward_loss_elem, masks)
            if cost_preds is not None:
                cost_loss_elem = loss_fn(cost_preds, costs)
                cost_loss = compute_masked_regression_loss(cost_loss_elem, masks)
            else:
                cost_loss = torch.tensor(0.0, device=device)

            loss = action_loss + AUX_REWARD_LOSS_WEIGHT * reward_loss + AUX_COST_LOSS_WEIGHT * cost_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_aux_reward_loss += reward_loss.item()
            train_aux_cost_loss += cost_loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_train_aux_reward_loss = train_aux_reward_loss / max(1, len(train_loader))
        avg_train_aux_cost_loss = train_aux_cost_loss / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        val_dist_errors = []
        active_val_errors = []
        val_loss = 0.0
        val_aux_reward_loss = 0.0
        val_aux_cost_loss = 0.0
        with torch.no_grad():
            for states, actions, rtgs, rewards, costs, ctgs, timesteps, masks in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                rewards = rewards.to(device)
                costs = costs.to(device)
                ctgs = ctgs.to(device)
                timesteps = timesteps.to(device)
                masks = masks.to(device)

                if model_kind == "cdt":
                    action_preds, reward_preds, cost_preds = model(
                        states, actions, rtgs, ctgs, timesteps, attention_mask=masks, return_aux=True
                    )
                else:
                    action_preds, reward_preds = model(
                        states, actions, rtgs, timesteps, attention_mask=masks, return_aux=True
                    )
                    cost_preds = None

                error_m = compute_active_dist_error(action_preds, actions, a_mean, a_std, seq_mask=masks)
                if error_m is not None:
                    active_val_errors.append(error_m)
                val_dist_errors.append(compute_mean_dist_error(action_preds, actions, a_mean, a_std, masks))

                action_loss_elem = loss_fn(action_preds, actions)
                action_loss = compute_weighted_action_loss(action_loss_elem, actions, masks, a_mean, a_std)
                reward_loss_elem = loss_fn(reward_preds, rewards)
                reward_loss = compute_masked_regression_loss(reward_loss_elem, masks)
                if cost_preds is not None:
                    cost_loss_elem = loss_fn(cost_preds, costs)
                    cost_loss = compute_masked_regression_loss(cost_loss_elem, masks)
                else:
                    cost_loss = torch.tensor(0.0, device=device)

                loss = action_loss + AUX_REWARD_LOSS_WEIGHT * reward_loss + AUX_COST_LOSS_WEIGHT * cost_loss
                val_loss += loss.item()
                val_aux_reward_loss += reward_loss.item()
                val_aux_cost_loss += cost_loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_aux_reward_loss = val_aux_reward_loss / max(1, len(val_loader))
        avg_val_aux_cost_loss = val_aux_cost_loss / max(1, len(val_loader))
        avg_dist_m = float(np.mean(val_dist_errors)) if val_dist_errors else 0.0
        avg_active_dist_m = float(np.mean(active_val_errors)) if active_val_errors else np.nan

        if (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
            last_eval_metrics = evaluate_multi_seed_metrics(
                model=model,
                model_kind=model_kind,
                cfg=CONFIG,
                seed_list=eval_seeds,
                s_mean=dataset.state_mean,
                s_std=dataset.state_std,
                a_mean=dataset.action_mean,
                a_std=dataset.action_std,
                rtg_scale=dataset.rtg_scale,
                ctg_scale=dataset.ctg_scale,
                target_return=EVAL_TARGET_RETURN,
                target_constraint=dataset.eval_target_constraint,
                max_waiting_slots=dataset.max_waiting_slots,
                device=device,
                use_car=use_car,
            )

        global_epoch = epoch + 1
        history["epoch"].append(global_epoch)
        history["variant"].append(label)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_aux_reward_loss"].append(avg_train_aux_reward_loss)
        history["val_aux_reward_loss"].append(avg_val_aux_reward_loss)
        history["train_aux_cost_loss"].append(avg_train_aux_cost_loss if model_kind == "cdt" else np.nan)
        history["val_aux_cost_loss"].append(avg_val_aux_cost_loss if model_kind == "cdt" else np.nan)
        history["dist_error_meters"].append(avg_dist_m)
        history["active_dist_error_meters"].append(avg_active_dist_m)
        history["learning_rate"].append(current_lr)
        history["success_rate"].append(last_eval_metrics["success_rate"])
        history["success_rate_std"].append(last_eval_metrics["success_rate_std"])
        history["avg_wait_steps"].append(last_eval_metrics["avg_wait_steps"])
        history["avg_wait_steps_std"].append(last_eval_metrics["avg_wait_steps_std"])
        history["eval_num_seeds"].append(last_eval_metrics["num_seeds"])
        pd.DataFrame(history).to_csv(log_path, index=False)

        curr_success = last_eval_metrics["success_rate"]
        curr_wait = last_eval_metrics["avg_wait_steps"]
        improved = (curr_success > best_success + 1e-6) or (
            abs(curr_success - best_success) <= 1e-6 and curr_wait < best_wait - 1e-6
        )
        if improved:
            best_success = curr_success
            best_wait = curr_wait
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"[{label}] Epoch {global_epoch:03d}/{epochs} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"Success {curr_success:.2f}\u00b1{last_eval_metrics['success_rate_std']:.2f}% | "
            f"Wait {curr_wait:.2f}\u00b1{last_eval_metrics['avg_wait_steps_std']:.2f} steps"
        )

    print(
        f"[{label}] done. Best success={best_success:.2f}%, "
        f"best wait={best_wait:.2f} steps | ckpt={ckpt_path}"
    )
    return log_path


def plot_compare(log_paths, out_path, expert_metrics=None):
    frames = []
    for label, path in log_paths.items():
        df = pd.read_csv(path).copy()
        df["setting"] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    style = {
        "DT": "#9467bd",
        "DT+CAR": "#1f77b4",
        "CDT": "#d62728",
        "CDT+CAR": "#2ca02c",
        "PMIX": "#bcbd22",
        "PMIX+CAR": "#8c564b",
    }

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for setting, g in df.groupby("setting"):
        color = style.get(setting, None)
        axes[0].plot(g["epoch"], g["success_rate"], label=setting, lw=2, color=color)
        axes[1].plot(g["epoch"], g["avg_wait_steps"], label=setting, lw=2, color=color)
        axes[2].plot(g["epoch"], g["val_loss"], label=setting, lw=2, color=color)

    if expert_metrics is not None and len(df) > 0:
        x_min = float(df["epoch"].min())
        x_max = float(df["epoch"].max())
        ex_color = "#ff7f0e"

        ex_sr = float(expert_metrics["success_rate"])
        ex_sr_std = float(expert_metrics["success_rate_std"])
        axes[0].plot(
            [x_min, x_max],
            [ex_sr, ex_sr],
            linestyle="--",
            color=ex_color,
            lw=2,
            label=f"Expert ({ex_sr:.2f}卤{ex_sr_std:.2f})",
        )
        if ex_sr_std > 0:
            axes[0].fill_between(
                [x_min, x_max],
                [ex_sr - ex_sr_std, ex_sr - ex_sr_std],
                [ex_sr + ex_sr_std, ex_sr + ex_sr_std],
                color=ex_color,
                alpha=0.12,
            )

        ex_w = float(expert_metrics["avg_wait_steps"])
        ex_w_std = float(expert_metrics["avg_wait_steps_std"])
        axes[1].plot(
            [x_min, x_max],
            [ex_w, ex_w],
            linestyle="--",
            color=ex_color,
            lw=2,
            label=f"Expert ({ex_w:.2f}卤{ex_w_std:.2f})",
        )
        if ex_w_std > 0:
            axes[1].fill_between(
                [x_min, x_max],
                [ex_w - ex_w_std, ex_w - ex_w_std],
                [ex_w + ex_w_std, ex_w + ex_w_std],
                color=ex_color,
                alpha=0.12,
            )

    axes[0].set_title("Success Rate")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Percent (%)")
    axes[1].set_title("Average Wait Steps")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Steps")
    axes[2].set_title("Validation Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L1 Loss")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()

    plt.suptitle("DT vs DT+CAR vs CDT vs CDT+CAR", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to: {out_path}")


def summarize_variant(log_path):
    df = pd.read_csv(log_path)
    if df.empty:
        return {"best_success": np.nan, "wait_at_best_success": np.nan, "best_epoch": -1}
    best_idx = df["success_rate"].idxmax()
    row = df.loc[best_idx]
    return {
        "best_success": float(row["success_rate"]),
        "wait_at_best_success": float(row["avg_wait_steps"]),
        "best_epoch": int(row["epoch"]),
    }


def _extract_mix_iter(path_obj):
    m = re.search(r"train_log_mix_iter(\d+)\.csv$", str(path_obj))
    if m is None:
        return -1
    return int(m.group(1))


def build_pmix_combined_log(pmix_work_dir):
    pmix_work_dir = Path(pmix_work_dir)
    phase0 = pmix_work_dir / "train_log_phase0.csv"
    mix_logs = sorted(
        list(pmix_work_dir.glob("train_log_mix_iter*.csv")),
        key=_extract_mix_iter,
    )

    pieces = []
    global_offset = 0
    if phase0.exists():
        df0 = pd.read_csv(phase0).copy()
        if not df0.empty:
            df0["epoch"] = np.arange(global_offset + 1, global_offset + len(df0) + 1)
            pieces.append(df0)
            global_offset += len(df0)

    for p in mix_logs:
        dfi = pd.read_csv(p).copy()
        if dfi.empty:
            continue
        dfi["epoch"] = np.arange(global_offset + 1, global_offset + len(dfi) + 1)
        pieces.append(dfi)
        global_offset += len(dfi)

    if len(pieces) == 0:
        raise FileNotFoundError(
            f"No PMIX logs found in {pmix_work_dir}. "
            f"Expected train_log_phase0.csv and/or train_log_mix_iter*.csv."
        )

    out = pd.concat(pieces, ignore_index=True)
    out["variant"] = "pmix"
    out_path = pmix_work_dir / "train_log_pmix_combined.csv"
    out.to_csv(out_path, index=False)
    return out_path


def run_pmix_training(
    dataset_path,
    pmix_work_dir,
    target_return,
    seed,
    early_epochs,
    mix_iters,
    mix_epochs,
    mix_total_episodes,
    mix_ratio,
    rollout_use_car,
    disable_car_module,
):
    cmd = [
        sys.executable,
        "performative_mix_train.py",
        "--base-dataset", str(dataset_path),
        "--work-dir", str(pmix_work_dir),
        "--seed", str(seed),
        "--target-return", str(target_return),
        "--early-epochs", str(early_epochs),
        "--mix-iters", str(mix_iters),
        "--mix-epochs", str(mix_epochs),
        "--mix-total-episodes", str(mix_total_episodes),
        "--mix-ratio", str(mix_ratio),
    ]
    if rollout_use_car:
        cmd.append("--rollout-use-car")
    if disable_car_module:
        cmd.append("--disable-car-module")
    print("\nRun PMIX training command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare DT/CDT plus PMIX variants in one script."
    )
    parser.add_argument("--dataset-path", type=str, default="expert_dataset.pkl")
    parser.add_argument("--log-dir", type=str, default="result")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-expert", action="store_true")
    parser.add_argument("--include-pmix", action="store_true")
    parser.add_argument("--pmix-work-dir", type=str, default="")
    parser.add_argument("--include-pmix-pure", action="store_true")
    parser.add_argument("--pmix-pure-work-dir", type=str, default="")
    parser.add_argument("--pmix-early-epochs", type=int, default=15)
    parser.add_argument("--pmix-mix-iters", type=int, default=3)
    parser.add_argument("--pmix-mix-epochs", type=int, default=8)
    parser.add_argument("--pmix-mix-total-episodes", type=int, default=2000)
    parser.add_argument("--pmix-mix-ratio", type=float, default=0.35)
    parser.add_argument("--pmix-rollout-use-car", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = parse_eval_seeds(args.eval_seeds)

    pmix_car_work_dir = Path(args.pmix_work_dir) if args.pmix_work_dir.strip() != "" else (log_dir / "pmix_run")
    pmix_pure_work_dir = (
        Path(args.pmix_pure_work_dir)
        if args.pmix_pure_work_dir.strip() != ""
        else (log_dir / "pmix_pure_run")
    )

    # Unified settings: each item fully declares its required parameters.
    settings = [
        {
            "name": "DT",
            "kind": "model",
            "enabled": True,
            "label": "dt",
            "model_kind": "dt",
            "use_car": False,
        },
        {
            "name": "DT+CAR",
            "kind": "model",
            "enabled": True,
            "label": "dt_car",
            "model_kind": "dt",
            "use_car": True,
        },
        {
            "name": "CDT",
            "kind": "model",
            "enabled": True,
            "label": "cdt",
            "model_kind": "cdt",
            "use_car": False,
        },
        {
            "name": "CDT+CAR",
            "kind": "model",
            "enabled": True,
            "label": "cdt_car",
            "model_kind": "cdt",
            "use_car": True,
        },
        {
            "name": "PMIX",
            "kind": "pmix",
            "enabled": bool(args.include_pmix_pure),
            "work_dir": pmix_pure_work_dir,
            "run_cfg": {
                "rollout_use_car": False,
                "disable_car_module": True,
            },
        },
        {
            "name": "PMIX+CAR",
            "kind": "pmix",
            "enabled": bool(args.include_pmix),
            "work_dir": pmix_car_work_dir,
            "run_cfg": {
                "rollout_use_car": bool(args.pmix_rollout_use_car),
                "disable_car_module": False,
            },
        },
    ]

    log_paths = {}
    for cfg_item in settings:
        if not bool(cfg_item.get("enabled", True)):
            continue

        if cfg_item["kind"] == "model":
            if not args.skip_train:
                log_path = train_one_variant(
                    label=cfg_item["label"],
                    model_kind=cfg_item["model_kind"],
                    use_car=bool(cfg_item["use_car"]),
                    dataset_path=str(dataset_path),
                    log_dir=log_dir,
                    epochs=int(args.epochs),
                    batch_size=int(args.batch_size),
                    eval_seeds=eval_seeds,
                    seed=int(args.seed),
                )
            else:
                log_path = log_dir / f"train_log_{cfg_item['label']}.csv"
                if not log_path.exists():
                    raise FileNotFoundError(f"skip-train enabled, but log not found: {log_path}")
            log_paths[cfg_item["name"]] = log_path
            continue

        if cfg_item["kind"] == "pmix":
            work_dir = Path(cfg_item["work_dir"])
            run_cfg = dict(cfg_item["run_cfg"])
            if not args.skip_train:
                run_pmix_training(
                    dataset_path=dataset_path,
                    pmix_work_dir=work_dir,
                    target_return=EVAL_TARGET_RETURN,
                    seed=int(args.seed),
                    early_epochs=int(args.pmix_early_epochs),
                    mix_iters=int(args.pmix_mix_iters),
                    mix_epochs=int(args.pmix_mix_epochs),
                    mix_total_episodes=int(args.pmix_mix_total_episodes),
                    mix_ratio=float(args.pmix_mix_ratio),
                    rollout_use_car=bool(run_cfg["rollout_use_car"]),
                    disable_car_module=bool(run_cfg["disable_car_module"]),
                )
            pmix_log = build_pmix_combined_log(work_dir)
            log_paths[cfg_item["name"]] = pmix_log
            continue

    expert_metrics = None
    if not args.skip_expert:
        expert_metrics = evaluate_expert_multi_seed(CONFIG, eval_seeds)
        pd.DataFrame([expert_metrics]).to_csv(log_dir / "expert_baseline_metrics_dt_cdt.csv", index=False)
        print(
            "\nExpert baseline | "
            f"Success: {expert_metrics['success_rate']:.2f}+/-{expert_metrics['success_rate_std']:.2f}% | "
            f"AvgWait: {expert_metrics['avg_wait_steps']:.2f}+/-{expert_metrics['avg_wait_steps_std']:.2f} "
            f"(seeds={expert_metrics['num_seeds']})"
        )

    fig_path = log_dir / "dt_cdt_car_compare.png"
    plot_compare(log_paths, fig_path, expert_metrics=expert_metrics)

    summary_rows = []
    for pretty_name, _ in log_paths.items():
        s = summarize_variant(log_paths[pretty_name])
        summary_rows.append(
            {
                "setting": pretty_name,
                "best_success_rate": s["best_success"],
                "wait_at_best_success": s["wait_at_best_success"],
                "best_epoch": s["best_epoch"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = log_dir / "dt_cdt_car_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n=== Summary (best success point) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()

