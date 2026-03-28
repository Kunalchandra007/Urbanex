"""
Task Easy — reach_destination
Goal: Reach destination in under 35 minutes with no constraints.
No incidents, low traffic, clear weather.
"""

EASY_CONFIG = {
    "id": "easy",
    "name": "reach_destination",
    "max_steps": 10,
    "dynamic_incidents": False,
    "traffic_level": "low",
    "weather": "clear",
    "time_limit_min": 35.0,
    "setup_incidents": [],
    "spawn_steps": [],
    "hidden_risk_scale": 0.10,   # Very low hidden risk — easy to succeed
    "reward_weights": {"safety": 0.3, "time": 0.5, "fuel": 0.2},
    "description": "Reach destination with no constraints in a clear city.",
}
