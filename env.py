import pandas as pd
from entities import *

import numpy as np
import random
from config import CONFIG


def get_uniform_grid_positions(num_agents, lat_min, lat_max, lon_min, lon_max):

    cols = int(np.ceil(np.sqrt(num_agents)))
    rows = int(np.ceil(num_agents / cols))

    lat_step = (lat_max - lat_min) / rows
    lon_step = (lon_max - lon_min) / cols

    positions = []
    count = 0

    for r in range(rows):
        for c in range(cols):
            if count >= num_agents:
                break

            lat = lat_min + (r + 0.5) * lat_step
            lon = lon_min + (c + 0.5) * lon_step
            positions.append([lat, lon])
            count += 1

    return positions


class ChargingEnv:
    def __init__(self, config):
        self.processed_request_ids = set()
        self.seed(42)
        self.cfg = config
        self.time_step = 0
        self.evs = {}
        self.mcs = {}
        self.fcs = {}
        self.stats = {
            "total_requests": 0,
            "served_mcs": 0,
            "served_fcs": 0,
            "dead_evs": 0,
            "current_waiting": 0,
            "overdue_wait_steps": 0,
            "fcs_reassignments": 0,
        }
        self.waiting_streak = {}
        self._load_dataset()

    def seed(self, seed=None):
        """Set random seed for environment."""
        self.np_random = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _load_dataset(self):
        """Load dataset and truncate to configured EV count."""
        df = pd.read_csv(self.cfg["dataset_path"])

        target_count = self.cfg["ev_count"]
        df_limited = df.head(target_count)

        self.trajectories = {}
        for _, row in df_limited.iterrows():
            ev_id = f"EV_{row['id']}"
            track_str = str(row['track'])
            if track_str.strip() == "": continue

            points = []
            for pt in track_str.split(','):
                lat, lon = pt.strip().split(' ')
                points.append((float(lat), float(lon)))
            self.trajectories[ev_id] = points

        if self.cfg.get("verbose_dataset_load", True):
            print(f"Loaded {len(self.trajectories)} EV trajectories from dataset.")

    def reset(self):
        self.time_step = 0
        self.evs.clear()
        self.mcs.clear()
        self.fcs.clear()
        self.stats = {
            "total_requests": 0,
            "served_mcs": 0,
            "served_fcs": 0,
            "dead_evs": 0,
            "current_waiting": 0,
            "overdue_wait_steps": 0,
            "fcs_reassignments": 0,
        }
        self.waiting_streak = {}
        for ev_id, track in self.trajectories.items():
            self.evs[ev_id] = EV(ev_id, track, self.cfg)

        uniform_positions = get_uniform_grid_positions(
            num_agents=self.cfg["mcs_num"],
            lat_min=self.cfg["SOUTH"],
            lat_max=self.cfg["NORTH"],
            lon_min=self.cfg["WEST"],
            lon_max=self.cfg["EAST"]
        )
        self.processed_request_ids = set()
        self.mcs = {}
        for i in range(self.cfg["mcs_num"]):
            init_lat, init_lon = uniform_positions[i]
            pos = (init_lat, init_lon)
            self.mcs[f"MCS_{i}"] = MCS(f"MCS_{i}", pos, self.cfg)

        for i, pos in enumerate(self.cfg["fcs_locations"]):
            self.fcs[f"FCS_{i}"] = FCS(f"FCS_{i}", pos, self.cfg["fcs_capacity"])

        return self.get_state()

    def get_state(self):
        return {
            "step": self.time_step,
            "requests": [ev.id for ev in self.evs.values() if ev.state == "WAITING"]
        }

    def _release_fcs_slot(self, ev):
        """Release reserved/charging slot in FCS when EV exits FCS flow."""
        if not ev.target_fcs_id:
            return
        fcs = self.fcs.get(ev.target_fcs_id)
        if fcs is not None and ev.id in fcs.serving_list:
            fcs.serving_list.remove(ev.id)
        ev.target_fcs_id = None

    def _release_mcs_service(self, ev):
        """Release MCS occupation when EV exits MCS charging flow."""
        if not ev.assigned_mcs_id:
            return
        mcs = self.mcs.get(ev.assigned_mcs_id)
        if mcs is not None and mcs.serving_ev_id == ev.id:
            mcs.serving_ev_id = None
            mcs.state = "IDLE"
        ev.assigned_mcs_id = None

    def _assign_ev_to_fcs(self, ev):
        """Reserve one nearby FCS slot and switch EV into MOVING_TO_FCS."""
        best_fcs = None
        best_dist = float('inf')
        for fcs in self.fcs.values():
            if fcs.is_full:
                continue
            dist = haversine_distance(ev.pos, fcs.pos)
            if dist <= self.cfg["ev_fcs_search_radius"] and dist < best_dist:
                best_fcs = fcs
                best_dist = dist

        if best_fcs is None:
            return False

        best_fcs.serving_list.append(ev.id)
        ev.target_fcs_id = best_fcs.id
        ev.charging_source = None
        ev.assigned_mcs_id = None
        ev.state = "MOVING_TO_FCS"
        ev.fcs_charge_steps = 0
        return True

    def _reassign_waiting_evs_to_fcs(self, waiting_evs):
        """Try to dispatch still-waiting EVs to FCS each step as a fallback."""
        if not self.cfg.get("enable_waiting_fcs_reassign", False):
            return

        for ev in list(waiting_evs):
            if ev.state != "WAITING":
                continue
            if self._assign_ev_to_fcs(ev):
                waiting_evs.remove(ev)
                self.stats["fcs_reassignments"] += 1

    def _refresh_waiting_streak(self):
        """Maintain per-EV consecutive WAITING steps for overtime penalties."""
        waiting_ids = {ev.id for ev in self.evs.values() if ev.state == "WAITING"}
        for ev_id in waiting_ids:
            self.waiting_streak[ev_id] = self.waiting_streak.get(ev_id, 0) + 1
        for ev_id in list(self.waiting_streak.keys()):
            if ev_id not in waiting_ids:
                del self.waiting_streak[ev_id]

    def _step_ev_to_fcs(self, ev):
        """Move EV toward assigned FCS with SOC consumption each step."""
        fcs = self.fcs.get(ev.target_fcs_id)
        if fcs is None:
            ev.target_fcs_id = None
            ev.state = "WAITING"
            return

        dist = haversine_distance(ev.pos, fcs.pos)
        if dist <= 0.01:
            ev.pos = fcs.pos
            ev.state = "CHARGING"
            ev.charging_source = "FCS"
            return

        move_dist = min(self.cfg["ev_fcs_drive_speed_km_per_step"], dist)
        ratio = move_dist / dist
        ev.pos = (
            ev.pos[0] + (fcs.pos[0] - ev.pos[0]) * ratio,
            ev.pos[1] + (fcs.pos[1] - ev.pos[1]) * ratio
        )

        consumed_soc = (move_dist * self.cfg["ev_consumption_rate"]) / self.cfg["ev_battery_capacity_kwh"]
        ev.soc -= consumed_soc
        if ev.soc <= 0:
            ev.soc = 0
            ev.state = "DEAD"
            self._release_fcs_slot(ev)
            return

        if haversine_distance(ev.pos, fcs.pos) <= 0.01:
            ev.pos = fcs.pos
            ev.state = "CHARGING"
            ev.charging_source = "FCS"

    def _step_ev_charging(self, ev):
        """Charge EV gradually at FCS or MCS and return reward gain."""
        if ev.charging_source == "FCS":
            if ev.target_fcs_id not in self.fcs:
                ev.target_fcs_id = None
                ev.charging_source = None
                ev.state = "WAITING"
                return 0.0

            ev.fcs_charge_steps += 1
            target_soc = self.cfg["ev_target_soc"]
            delta_soc = self.cfg["ev_charge_soc_per_step"]
            ev.soc = min(target_soc, ev.soc + delta_soc)

            if ev.soc >= target_soc:
                ev.state = "DONE"
                ev.charging_source = None
                self.stats["served_fcs"] += 1
                self._release_fcs_slot(ev)
            return 0.0

        if ev.charging_source == "MCS":
            mcs = self.mcs.get(ev.assigned_mcs_id)
            if mcs is None or mcs.serving_ev_id != ev.id:
                ev.assigned_mcs_id = None
                ev.charging_source = None
                ev.state = "WAITING"
                return 0.0

            target_soc = self.cfg["ev_target_soc"]
            delta_soc = self.cfg["ev_charge_soc_per_step"]
            ev.soc = min(target_soc, ev.soc + delta_soc)

            if ev.soc >= target_soc:
                ev.state = "DONE"
                ev.charging_source = None
                self.stats["served_mcs"] += 1
                self._release_mcs_service(ev)
                return float(self.cfg["reward_serve_success"])
            return 0.0

        ev.state = "WAITING"
        return 0.0

    def step(self, action_matrix):
        reward = 0
        self.time_step += 1

        if action_matrix is not None:
            for i, target_coord in enumerate(action_matrix):
                mcs_id = f"MCS_{i}"
                if mcs_id in self.mcs:
                    self.mcs[mcs_id].target_pos = (target_coord[0], target_coord[1])

        for ev in self.evs.values():
            if ev.state == "MOVING":
                ev.step_physics()
                if 0 < ev.soc < self.cfg["ev_request_threshold"]:
                    self.stats["total_requests"] += 1
                    if not self._assign_ev_to_fcs(ev):
                        ev.state = "WAITING"
            elif ev.state == "MOVING_TO_FCS":
                self._step_ev_to_fcs(ev)
            elif ev.state == "CHARGING":
                reward += self._step_ev_charging(ev)

        for mcs in self.mcs.values():
            mcs.step_physics()

        waiting_evs = [ev for ev in self.evs.values() if ev.state == "WAITING"]

        for mcs in self.mcs.values():
            if mcs.state == "CHARGING":
                continue
            closest_ev = None
            min_dist = float('inf')

            for ev in waiting_evs:
                dist = haversine_distance(mcs.pos, ev.pos)
                if dist <= self.cfg["mcs_service_radius_km"] and dist < min_dist:
                    closest_ev = ev
                    min_dist = dist

            if closest_ev is not None:
                mcs.state = "CHARGING"
                mcs.serving_ev_id = closest_ev.id
                closest_ev.state = "CHARGING"
                closest_ev.charging_source = "MCS"
                closest_ev.assigned_mcs_id = mcs.id
                waiting_evs.remove(closest_ev)

        self._reassign_waiting_evs_to_fcs(waiting_evs)
        self.stats["current_waiting"] = sum(1 for ev in self.evs.values() if ev.state == "WAITING")
        self._refresh_waiting_streak()

        for ev in self.evs.values():
            if ev.state == "WAITING":
                reward += self.cfg["reward_wait_penalty"]
                wait_target = int(self.cfg.get("wait_target_steps", 2))
                wait_streak = self.waiting_streak.get(ev.id, 0)
                if wait_streak > wait_target:
                    reward += float(self.cfg.get("reward_wait_overdue_penalty", 0.0))
                    self.stats["overdue_wait_steps"] += 1
            elif ev.state == "DEAD":
                reward += self.cfg["reward_dead_penalty"]
                self.stats["dead_evs"] += 1
                self._release_fcs_slot(ev)
                self._release_mcs_service(ev)
                ev.charging_source = None

        info = {
            "served_total": self.stats["served_mcs"] + self.stats["served_fcs"],
            "success_rate": self._calculate_success_rate(),
            "mcs_contribution": self.stats["served_mcs"],
            "fcs_contribution": self.stats["served_fcs"],
            "waiting_count": self.stats["current_waiting"],
            "dead_count": self.stats["dead_evs"],
            "overdue_wait_steps": self.stats["overdue_wait_steps"],
            "fcs_reassignments": self.stats["fcs_reassignments"]
        }

        done = self.time_step >= self.cfg["max_steps"]
        return self.get_state(), reward, done, info

    def _calculate_success_rate(self):
        total = self.stats["total_requests"]
        if total == 0:
            return 100.0
        served = self.stats["served_mcs"] + self.stats["served_fcs"]
        return (served / total) * 100


