"""
Task Hard — dynamic_rerouting.

Designed to force genuine rerouting decisions under changing conditions.
"""

HARD_CONFIG = {
    "id": "hard",
    "name": "dynamic_rerouting",
    "max_steps": 18,
    "dynamic_incidents": True,
    "traffic_level": "low",
    "weather": "clear",
    "time_limit_min": 55.0,
    "setup_incidents": [
        {"type": "flooding", "severity": "medium", "affects_routes": ["eco"]},
    ],
    "spawn_steps": [2, 4, 6, 9, 12, 15],
    "traffic_change_step": 5,
    "traffic_after": "medium",
    "traffic_change_step_2": 10,
    "traffic_after_2": "high",
    "force_incident_step": 3,
    "force_incident_type": "accident",
    "force_incident_severity": "high",
    "force_incident_route": "fastest",
    "fuel_critical_step": 12,
    "fuel_critical_penalty": 0.1,
    "fuel_critical_route": "eco",
    "weather_change_step": 8,
    "weather_after": "heavy_rain",
    "weather_safety_penalty": 0.3,
    "reward_weights": {"safety": 0.5, "time": 0.3, "fuel": 0.2},
    "description": (
        "Navigate a dynamic city with forced rerouting on the fastest route, "
        "fuel pressure late in the episode, and a heavy-rain safety collapse."
    ),
    "success_condition": {
        "must_reach_destination": True,
        "min_avg_safety_score": 0.6,
    },
}
