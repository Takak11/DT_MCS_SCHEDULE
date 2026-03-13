import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment  # 引入匈牙利匹配算法
from env import ChargingEnv, haversine_distance
from config import CONFIG
# 全局字典，用于记忆每辆 MCS 锁定的 EV ID
# 格式: {mcs_id: ev_id}
assignment_memory = {}


# ==========================================
# 1. 完整状态向量化 (含电量 SOC 补丁)
# ==========================================
def get_state_vector(env, max_waiting_evs=50):
    """
    返回固定长度的一维特征向量。
    MCS特征: mcs_num *  2 (纬度, 经度)
    EV特征: max_waiting_evs * 3 (纬度, 经度, SOC剩余电量)
    """
    state_features = []

    # 1. 提取 MCS 坐标
    for i in range(CONFIG["mcs_num"]):
        mcs = env.mcs[f"MCS_{i}"]
        state_features.extend([mcs.pos[0], mcs.pos[1]])

    # 2. 提取 WAITING 状态的 EV 坐标与电量
    waiting_evs = [ev for ev in env.evs.values() if ev.state == "WAITING"]

    for i in range(max_waiting_evs):
        if i < len(waiting_evs):
            ev = waiting_evs[i]
            state_features.extend([ev.pos[0], ev.pos[1], ev.soc])
        else:
            state_features.extend([0.0, 0.0, 0.0])

    return np.array(state_features, dtype=np.float32)

def expert_get_action_with_commitment(env, epsilon=0.0):
    global assignment_memory

    # 1. 修复后的记忆释放逻辑 (适配 EV 恢复 MOVING 的特性)
    for m_id, e_id in list(assignment_memory.items()):
        # 如果这辆车被环境销毁了，释放 MCS
        if e_id not in env.evs:
            del assignment_memory[m_id]
            continue

        ev_state = env.evs[e_id].state

        # 任务结束逻辑：只要不是在等待或充电，释放MCS
        if ev_state not in ["WAITING", "CHARGING"]:
            del assignment_memory[m_id]

    # 2. 划分出 "需要派单的空闲 MCS" 和 "等待救援的 EV"
    free_mcs = [m for m in env.mcs.values() if m.id not in assignment_memory]

    # 获取还没被任何 MCS 锁定的纯粹的 WAITING EV
    locked_ev_ids = set(assignment_memory.values())
    unassigned_evs = [e for e in env.evs.values() if e.state == "WAITING" and e.id not in locked_ev_ids]

    # 3. 只对空闲资源进行二分图匹配
    if free_mcs and unassigned_evs:
        if np.random.rand() < epsilon:
            shuffled_mcs = np.random.permutation(len(free_mcs))
            shuffled_evs = np.random.permutation(len(unassigned_evs))
            pair_num = min(len(free_mcs), len(unassigned_evs))
            for k in range(pair_num):
                m_id = free_mcs[shuffled_mcs[k]].id
                e_id = unassigned_evs[shuffled_evs[k]].id
                assignment_memory[m_id] = e_id
        else:
            wait_target = int(env.cfg.get("wait_target_steps", 2))
            urgency_km = float(env.cfg.get("expert_wait_urgency_km_per_step", 0.0))
            overdue_priority_km = float(env.cfg.get("expert_overdue_priority_km", 0.0))
            cost_matrix = np.zeros((len(free_mcs), len(unassigned_evs)))
            for i, mcs in enumerate(free_mcs):
                for j, ev in enumerate(unassigned_evs):
                    # Use kilometer distance + waiting urgency to push overdue EVs first.
                    dist_km = haversine_distance(mcs.pos, ev.pos)
                    wait_steps = int(getattr(env, "waiting_streak", {}).get(ev.id, 0))
                    overdue_steps = max(0, wait_steps - wait_target)
                    urgency_bonus = urgency_km * wait_steps + overdue_priority_km * overdue_steps
                    cost_matrix[i, j] = dist_km - urgency_bonus

            m_indices, e_indices = linear_sum_assignment(cost_matrix)
            for m_idx, e_idx in zip(m_indices, e_indices):
                m_id = free_mcs[m_idx].id
                e_id = unassigned_evs[e_idx].id
                assignment_memory[m_id] = e_id

    # 4. 生成最终的 Action 矩阵
    # 假设需要生成固定槽位长度的 actions (如果是动态的，写 len(env.mcs) 也可以)
    actions = np.zeros((len(env.mcs), 2))

    # 使用 enumerate 获取纯数字索引 i (0, 1, 2...)
    for i, mcs in enumerate(env.mcs.values()):
        if mcs.id in assignment_memory:
            # 如果有锁定目标，动作就是朝着目标开
            target_ev_id = assignment_memory[mcs.id]
            # 用纯数字 i 进行赋值
            actions[i] = env.evs[target_ev_id].pos
        else:
            # 如果没活干，原地待命
            actions[i] = mcs.pos

    return actions


# ==========================================
# 3. 收集并保存离线数据集
# ==========================================
def generate_offline_dataset(episodes=1000, save_path="expert_dataset.pkl", base_seed=42, epsilon=None):
    if epsilon is None:
        epsilon = float(CONFIG.get("expert_epsilon", 0.0))
    print(f"Start generating dataset: episodes={episodes}, base_seed={base_seed}, epsilon={epsilon}")

    dataset = []
    env = ChargingEnv(CONFIG)

    for ep in range(episodes):
        assignment_memory.clear()
        env.seed(base_seed + ep)
        env.reset()
        obs_seq, action_seq, reward_seq = [], [], []
        done = False

        while not done:
            current_obs = get_state_vector(env)
            obs_seq.append(current_obs)

            # action = expert_get_action(env, epsilon=0.01)
            action = expert_get_action_with_commitment(env, epsilon=epsilon)
            action_seq.append(action)

            _, reward, done, _ = env.step(action)
            reward_seq.append(reward)

        # 计算 Return-To-Go
        returns_to_go = np.zeros_like(reward_seq, dtype=np.float32)
        curr_rtg = 0
        for t in reversed(range(len(reward_seq))):
            curr_rtg += reward_seq[t]
            returns_to_go[t] = curr_rtg

        dataset.append({
            'observations': np.array(obs_seq, dtype=np.float32),
            'actions': np.array(action_seq, dtype=np.float32),
            'rewards': np.array(reward_seq, dtype=np.float32),
            'returns_to_go': returns_to_go
        })

        # 每跑 50 个 Episode 打印一次进度
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{episodes} | 最近一条轨迹总得分: {returns_to_go[0]:.1f}")

    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"数据集已保存至: {save_path}，轨迹条数{len(dataset)}")


if __name__ == "__main__":
    generate_offline_dataset(episodes=1000)
