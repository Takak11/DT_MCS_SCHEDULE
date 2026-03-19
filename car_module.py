import numpy as np

from utils import haversine_distance

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def _sanitize_action_matrix(env, action_matrix):
    """
    Clamp action coordinates to valid map bounds and repair non-finite values.
    """
    mcs_list = list(env.mcs.values())
    arr = np.asarray(action_matrix, dtype=np.float32).reshape(len(mcs_list), 2).copy()
    lat_min, lat_max = env.cfg["SOUTH"], env.cfg["NORTH"]
    lon_min, lon_max = env.cfg["WEST"], env.cfg["EAST"]

    for i, mcs in enumerate(mcs_list):
        lat, lon = float(arr[i, 0]), float(arr[i, 1])
        if (not np.isfinite(lat)) or (not np.isfinite(lon)):
            lat, lon = float(mcs.pos[0]), float(mcs.pos[1])
        arr[i, 0] = float(np.clip(lat, lat_min, lat_max))
        arr[i, 1] = float(np.clip(lon, lon_min, lon_max))
    return arr


def _build_cost_matrix(env, mcs_list, waiting_evs, pred_targets):
    cfg = env.cfg
    w_pred = float(cfg.get("car_weight_pred", 0.4))
    w_move = float(cfg.get("car_weight_move", 1.0))
    wait_target = int(cfg.get("wait_target_steps", 2))
    urgency_km = float(cfg.get("car_wait_urgency_km_per_step", cfg.get("expert_wait_urgency_km_per_step", 0.8)))
    overdue_priority_km = float(cfg.get("car_overdue_priority_km", cfg.get("expert_overdue_priority_km", 6.0)))

    cost = np.zeros((len(mcs_list), len(waiting_evs)), dtype=np.float32)
    for i, mcs in enumerate(mcs_list):
        for j, ev in enumerate(waiting_evs):
            pred_dist_km = haversine_distance(pred_targets[i], ev.pos)
            move_dist_km = haversine_distance(mcs.pos, ev.pos)
            wait_steps = int(getattr(env, "waiting_streak", {}).get(ev.id, 0))
            overdue_steps = max(0, wait_steps - wait_target)
            urgency_bonus = urgency_km * wait_steps + overdue_priority_km * overdue_steps
            cost[i, j] = w_pred * pred_dist_km + w_move * move_dist_km - urgency_bonus
    return cost


def _greedy_assign(env, mcs_list, waiting_evs, pred_targets):
    """
    Fallback when scipy is unavailable: greedy by minimal adjusted cost.
    """
    cost = _build_cost_matrix(env, mcs_list, waiting_evs, pred_targets)
    chosen_cols = set()
    pairs = []
    for i in range(cost.shape[0]):
        best_j = None
        best_val = float("inf")
        for j in range(cost.shape[1]):
            if j in chosen_cols:
                continue
            if cost[i, j] < best_val:
                best_val = float(cost[i, j])
                best_j = j
        if best_j is not None:
            chosen_cols.add(best_j)
            pairs.append((i, best_j))
    return pairs


def apply_constraint_aware_reranking(env, predicted_action_matrix):
    """
    CAR module:
    1) sanitize DT action;
    2) rerank/assign MCS targets to current WAITING EVs with assignment cost;
    3) keep original DT targets for unassigned MCS.
    """
    sanitized = _sanitize_action_matrix(env, predicted_action_matrix)
    if not bool(env.cfg.get("use_car_module", True)):
        return sanitized

    waiting_evs = [ev for ev in env.evs.values() if ev.state == "WAITING"]
    if len(waiting_evs) == 0:
        return sanitized

    max_cands = int(env.cfg.get("car_max_candidate_waiting", 200))
    if max_cands > 0 and len(waiting_evs) > max_cands:
        waiting_evs = sorted(
            waiting_evs,
            key=lambda ev: (
                -int(getattr(env, "waiting_streak", {}).get(ev.id, 0)),
                float(ev.soc),
            ),
        )[:max_cands]

    mcs_list = list(env.mcs.values())
    if len(mcs_list) == 0:
        return sanitized

    if linear_sum_assignment is None:
        pairs = _greedy_assign(env, mcs_list, waiting_evs, sanitized)
    else:
        cost = _build_cost_matrix(env, mcs_list, waiting_evs, sanitized)
        r_idx, c_idx = linear_sum_assignment(cost)
        pairs = list(zip(r_idx.tolist(), c_idx.tolist()))

    out = sanitized.copy()
    for i, j in pairs:
        if 0 <= i < len(mcs_list) and 0 <= j < len(waiting_evs):
            out[i, 0] = float(waiting_evs[j].pos[0])
            out[i, 1] = float(waiting_evs[j].pos[1])
    return out
