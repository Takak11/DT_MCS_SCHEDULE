import torch
import numpy as np
import pickle

from env import ChargingEnv
from config import CONFIG
from DT import DecisionTransformer
from car_module import apply_constraint_aware_reranking
from generate_DT_dataset import get_state_vector, expert_get_action_with_commitment, assignment_memory

CONTEXT_LEN = 50
TARGET_RETURN = 150000.0


def evaluate_and_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ChargingEnv(CONFIG)

    with open("scaler_params.pkl", "rb") as f:
        scalers = pickle.load(f)
        s_mean, s_std = scalers["state_mean"], scalers["state_std"]
        a_mean, a_std = scalers["action_mean"], scalers["action_std"]
        rtg_scale = scalers["rtg_scale"]

    model = DecisionTransformer(
        state_dim=len(s_mean), action_dim=len(a_mean), max_length=CONFIG["max_steps"]
    ).to(device)
    ckpt = torch.load("dt_mcs_best.pth", map_location=device, weights_only=True)
    load_res = model.load_state_dict(ckpt, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print("Warning: checkpoint/model mismatch detected. It is recommended to retrain with current architecture.")
    model.eval()

    metrics = {
        "wait_start_step": {},
        "all_wait_durations": [],
        "mcs_moving_steps": 0,
    }

    env.reset()
    states_hist, actions_hist, rtgs_hist, timesteps_hist = [], [], [], []
    current_rtg = TARGET_RETURN
    total_reward = 0.0
    final_info = {}

    print("Start evaluation run...")

    last_step = -1
    for t in range(CONFIG["max_steps"]):
        last_step = t
        raw_state = get_state_vector(env)
        norm_state = (raw_state - s_mean) / s_std

        states_hist.append(norm_state)
        rtgs_hist.append([current_rtg / rtg_scale])
        timesteps_hist.append(t)

        if len(actions_hist) < len(states_hist):
            actions_hist.append(np.zeros(len(a_mean)))

        s_input = np.array(states_hist[-CONTEXT_LEN:])
        a_input = np.array(actions_hist[-CONTEXT_LEN:])
        rtg_input = np.array(rtgs_hist[-CONTEXT_LEN:])
        t_input = np.array(timesteps_hist[-CONTEXT_LEN:])

        pad_len = CONTEXT_LEN - len(s_input)
        attn_mask = np.concatenate([np.zeros(pad_len), np.ones(len(s_input))]) if pad_len > 0 else np.ones(CONTEXT_LEN)
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

        with torch.no_grad():
            action_preds = model(s_tensor, a_tensor, rtg_tensor, t_tensor, attention_mask=m_tensor)

        pred_action_norm = action_preds[0, -1].cpu().numpy()
        actions_hist[-1] = pred_action_norm
        real_action_matrix = (pred_action_norm * a_std + a_mean).reshape(CONFIG["mcs_num"], 2)
        real_action_matrix = apply_constraint_aware_reranking(env, real_action_matrix)

        prev_states = {ev.id: ev.state for ev in env.evs.values()}

        _, reward, done, info = env.step(real_action_matrix)
        final_info = info
        current_rtg -= reward
        total_reward += reward

        for ev in env.evs.values():
            prev_state = prev_states.get(ev.id)
            curr_state = ev.state
            curr_source = ev.charging_source

            if curr_state == "WAITING":
                if prev_state != "WAITING":
                    metrics["wait_start_step"][ev.id] = t
            else:
                immediate_wait_then_service = (
                    prev_state == "MOVING" and
                    curr_state == "CHARGING" and
                    curr_source in {"MCS", "FCS"}
                )
                if immediate_wait_then_service:
                    metrics["all_wait_durations"].append(0)
                    metrics["wait_start_step"].pop(ev.id, None)
                elif ev.id in metrics["wait_start_step"]:
                    start_step = metrics["wait_start_step"][ev.id]
                    got_service_after_waiting = (
                        curr_state == "MOVING_TO_FCS" or
                        (curr_state == "CHARGING" and curr_source in {"MCS", "FCS"})
                    )
                    if got_service_after_waiting:
                        metrics["all_wait_durations"].append(max(0, t - start_step))
                        metrics["wait_start_step"].pop(ev.id, None)
                    elif prev_state == "WAITING":
                        metrics["all_wait_durations"].append(max(0, t - start_step))
                        metrics["wait_start_step"].pop(ev.id, None)

        for mcs in env.mcs.values():
            if mcs.state == "MOVING":
                metrics["mcs_moving_steps"] += 1

        if done:
            break

    if metrics["wait_start_step"] and last_step >= 0:
        episode_end_step = last_step + 1
        for start_step in metrics["wait_start_step"].values():
            metrics["all_wait_durations"].append(max(0, episode_end_step - start_step))

    total_requests = env.stats.get("total_requests", 0)
    served_total = final_info.get(
        "served_total", env.stats.get("served_mcs", 0) + env.stats.get("served_fcs", 0)
    )
    dead_count = int(final_info.get("dead_count", env.stats.get("dead_evs", 0)))
    unresolved = max(0, total_requests - served_total - dead_count)

    service_rate = (served_total / total_requests * 100.0) if total_requests > 0 else 100.0
    dead_rate = (dead_count / total_requests * 100.0) if total_requests > 0 else 0.0
    step_minutes = CONFIG.get("minutes_per_step", 24.0 * 60.0 / max(1, CONFIG["max_steps"]))
    mcs_price_per_kwh = float(CONFIG.get("mcs_price_per_kwh", 1.6))
    avg_wait_all_steps = (
        float(np.mean(metrics["all_wait_durations"])) if metrics["all_wait_durations"] else 0.0
    )
    mcs_total_energy_kwh = float(final_info.get("mcs_total_energy_kwh", env.stats.get("mcs_total_energy_kwh", 0.0)))
    mcs_total_revenue = float(final_info.get("mcs_total_revenue", env.stats.get("mcs_total_revenue", 0.0)))
    mcs_avg_revenue = float(
        final_info.get(
            "mcs_avg_revenue_per_vehicle",
            mcs_total_revenue / float(max(1, int(CONFIG.get("mcs_num", 1))))
        )
    )

    print("\n" + "=" * 40)
    print("Decision Transformer Evaluation")
    print("=" * 40)
    print(f"Total Reward (RL Metric): {total_reward:.1f}")
    print("-" * 40)
    print(f"Total Requests: {total_requests}")
    print(f"Served Total: {served_total}")
    print(f"Dead EVs: {dead_count}")
    print(f"Unresolved Requests: {unresolved}")
    print("-" * 40)
    print(f"Service Rate: {service_rate:.2f}%")
    print(f"Dead Rate: {dead_rate:.2f}%")
    print(f"MCS Served: {env.stats.get('served_mcs', 0)}")
    print(f"FCS Served: {env.stats.get('served_fcs', 0)}")
    print(f"Average Wait (all waiting): {avg_wait_all_steps:.1f} steps ({avg_wait_all_steps * step_minutes:.1f} min)")
    print(f"MCS Total Energy: {mcs_total_energy_kwh:.1f} kWh")
    print(f"MCS Price: {mcs_price_per_kwh:.2f} unit/kWh")
    print(f"MCS Total Revenue: {mcs_total_revenue:.1f} unit")
    print(f"MCS Avg Revenue per Vehicle: {mcs_avg_revenue:.2f} unit/MCS")
    print(f"MCS Moving Steps: {metrics['mcs_moving_steps']}")
    print("=" * 40)


def run_expert_episode(env):
    assignment_memory.clear()
    env.reset()
    done = False
    final_info = {}

    while not done:
        action = expert_get_action_with_commitment(env, epsilon=0.0)
        _, _, done, info = env.step(action)
        final_info = info

    return final_info.get("success_rate", 100.0)


def compare_performance(seed=42):
    env = ChargingEnv(CONFIG)
    env.seed(seed)
    expert_success = run_expert_episode(env)
    print(f"Expert Success Rate: {expert_success:.2f}%")


if __name__ == "__main__":
    evaluate_and_benchmark()
    compare_performance()
