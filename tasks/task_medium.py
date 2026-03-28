"""
Task Medium — avoid_and_optimize  v2
Adds greedy trap: fastest route looks clean but a hidden incident fires at step 2.
"""

MEDIUM_CONFIG = {
    "id": "medium",
    "name": "avoid_and_optimize",
    "max_steps": 15,
    "dynamic_incidents": False,
    "traffic_level": "medium",
    "weather": "clear",
    "time_limit_min": 45.0,
    # Pre-placed incidents: 2 medium + 1 high on fastest route
    "setup_incidents": [
        {"type": "construction", "severity": "medium", "affects_routes": ["fastest"]},
        {"type": "pothole",      "severity": "medium", "affects_routes": ["fastest", "eco"]},
        {"type": "accident",     "severity": "high",   "affects_routes": ["fastest"]},
    ],
    "spawn_steps": [],
    # ── GREEDY TRAP ──────────────────────────────────────────────────────────
    # At start, fastest route *appears* clear (incidents age out by step 0).
    # But at step 2, a hidden accident triggers a time+safety penalty.
    # Agents that greedily pick fastest and don't reroute get punished.
    "greedy_trap": True,
    "greedy_trap_step": 2,
    "greedy_trap_penalty": 0.45,   # subtracted from reward at step 2 if on fastest
    "hidden_risk_scale": 0.45,     # moderate hidden risk activation
    # ─────────────────────────────────────────────────────────────────────────
    "reward_weights": {"safety": 0.5, "time": 0.3, "fuel": 0.2},
    "description": "Reach destination while avoiding pre-placed incidents and a greedy trap.",
    "success_condition": {
        "must_reach_destination": True,
        "max_high_severity_encounters": 0,
    },
}
