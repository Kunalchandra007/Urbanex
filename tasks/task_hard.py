"""
Task Hard — dynamic_rerouting  v2
More aggressive dynamics: 6 spawn steps, 2-stage traffic, earlier weather change,
tighter max_steps (18), 2 pre-placed incidents for a brutal opening.
"""

HARD_CONFIG = {
    "id": "hard",
    "name": "dynamic_rerouting",
    "max_steps": 18,                      # tighter (was 20)
    "dynamic_incidents": True,
    "traffic_level": "low",              # two-stage escalation
    "weather": "clear",                  # two-stage change
    "time_limit_min": 55.0,
    # Two incidents pre-placed at start
    "setup_incidents": [
        {"type": "flooding",    "severity": "medium", "affects_routes": ["eco"]},
        {"type": "construction","severity": "high",   "affects_routes": ["fastest"]},
    ],
    # Dynamic spawn: 6 events (was 4)
    "spawn_steps": [2, 4, 6, 9, 12, 15],
    # Two-stage traffic escalation
    "traffic_change_step": 5,
    "traffic_after": "medium",
    "traffic_change_step_2": 10,
    "traffic_after_2": "high",
    # Two-stage weather change
    "weather_change_step": 7,
    "weather_after": "rain",
    "weather_change_step_2": 13,
    "weather_after_2": "fog",
    "reward_weights": {"safety": 0.5, "time": 0.3, "fuel": 0.2},
    "description": "Navigate a brutally dynamic city: 6 incident spawns, 2-stage traffic & weather changes.",
    "success_condition": {
        "must_reach_destination": True,
        "min_avg_safety_score": 0.6,
    },
}
