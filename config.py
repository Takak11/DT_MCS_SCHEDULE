CONFIG = {
    "ev_battery_capacity_kwh": 50.0,
    "ev_consumption_rate": 0.2,
    "ev_init_soc_mean": 0.80,
    "ev_init_soc_std": 0.28,
    "ev_request_threshold": 0.2,

    "mcs_speed_km_per_step": 4.0,
    "mcs_num": 20,

    "reward_wait_penalty": -1.0,
    "reward_wait_overdue_penalty": -4.0,
    "reward_dead_penalty": -50.0,
    "reward_serve_success": 300.0,
    "max_steps": 200,
    "minutes_per_step": 5.4,
    "wait_target_steps": 2,
    "wait_timeout_steps": 5,
    "enable_waiting_fcs_reassign": False,
    "use_car_module": True,
    "car_weight_pred": 0.4,
    "car_weight_move": 1.0,
    "car_wait_urgency_km_per_step": 0.8,
    "car_overdue_priority_km": 6.0,
    "car_max_candidate_waiting": 200,
    "expert_epsilon": 0.0,
    "expert_wait_urgency_km_per_step": 0.8,
    "expert_overdue_priority_km": 6.0,
    "business_min_success_rate": 93.0,

    "dataset_path": "dataset/20140818.csv"
}

CONFIG.update({
    "fcs_locations": [
        (30.70707274465113, 104.07255293913542),
        (30.617588774273347, 104.13052661526324),
        (30.674417, 104.003988)
    ],
    "fcs_capacity": 3,
    "ev_fcs_search_radius": 3,
    "ev_fcs_drive_speed_km_per_step": 4.0,
    "ev_charge_soc_per_step": 0.216,
    "ev_target_soc": 0.8
})

CONFIG.update({
    "mcs_service_radius_km": 2,
})

CONFIG.update({
    "ev_count": 1000,
})

CONFIG.update({
    "result_path": "result/",
})

CONFIG.update({
    "WEST": 103.9808,
    "SOUTH": 30.5963,
    "EAST": 104.1614,
    "NORTH": 30.7291,
})
