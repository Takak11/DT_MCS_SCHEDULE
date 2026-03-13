import numpy as np
from utils import *


class EV:
    def __init__(self, ev_id, trajectory, config):
        self.id = ev_id
        self.trajectory = trajectory  # [(lat, lon), (lat, lon), ...]
        self.traj_idx = 0
        self.pos = self.trajectory[0]
        self.cfg = config
        # Use per-EV RNG derived from global RNG state so episodes vary with env seed.
        self.seed(np.random.randint(0, 2**31 - 1))

        self.soc = np.clip(
            self.np_random.normal(loc=self.cfg["ev_init_soc_mean"], scale=self.cfg["ev_init_soc_std"]),
            0.10, 1.0
        )
        self.state = "MOVING"  # MOVING, WAITING, MOVING_TO_FCS, CHARGING, DEAD, DONE
        self.target_fcs_id = None
        self.assigned_mcs_id = None
        self.charging_source = None  # None, FCS, MCS
        self.fcs_charge_steps = 0

    def seed(self, seed=None):
        """Set EV-local random state without mutating global RNG."""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def step_physics(self):
        """娌胯建杩圭Щ鍔ㄥ苟璁＄畻鐪熷疄鎺夌數"""
        if self.state == "MOVING":
            if self.traj_idx < len(self.trajectory) - 1:
                next_pos = self.trajectory[self.traj_idx + 1]

                dist_km = haversine_distance(self.pos, next_pos)
                consumed_soc = (dist_km * self.cfg["ev_consumption_rate"]) / self.cfg["ev_battery_capacity_kwh"]

                self.pos = next_pos
                self.soc -= consumed_soc
                self.traj_idx += 1

                if self.soc <= 0:
                    self.soc = 0
                    self.state = "DEAD"
            else:
                self.state = "DONE"


class FCS:
    def __init__(self, fcs_id, pos, capacity):
        self.id = fcs_id
        self.pos = pos
        self.capacity = capacity
        self.serving_list = []

    @property
    def is_full(self):
        return len(self.serving_list) >= self.capacity


class MCS:
    def __init__(self, mcs_id, start_pos, config):
        self.id = mcs_id
        self.pos = start_pos
        self.target_pos = start_pos
        self.state = "IDLE"  # IDLE, MOVING, CHARGING
        self.serving_ev_id = None
        self.cfg = config

    def step_physics(self):
        if self.state in ["IDLE", "MOVING"]:
            dist = haversine_distance(self.pos, self.target_pos)

            if dist <= 0.01:
                self.pos = self.target_pos
                self.state = "IDLE"
            else:
                self.state = "MOVING"
                move_dist = min(self.cfg["mcs_speed_km_per_step"], dist)
                ratio = move_dist / dist
                self.pos = (
                    self.pos[0] + (self.target_pos[0] - self.pos[0]) * ratio,
                    self.pos[1] + (self.target_pos[1] - self.pos[1]) * ratio
                )

