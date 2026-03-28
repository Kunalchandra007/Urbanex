"""
Grader for Easy task: reach_destination.

Scoring:
  +0.6  if destination reached
  +0.2  if reached within time limit (35 min)
  +0.2  if no unnecessary stops/reroutes
Returns float in [0.0, 1.0]. Deterministic. Sensitive to trajectory differences.
"""
from typing import List


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def grade(trajectory: List[dict]) -> float:
    """
    Grade an episode trajectory for the easy task.
    Each element: {"obs": Observation, "action": Action, "reward": Reward}
    """
    if not trajectory:
        return 0.0

    score = 0.0

    # Check if destination was reached
    last_obs = trajectory[-1]["obs"]
    destination_reached = last_obs.episode_done and not _was_stopped(trajectory)

    # +0.6: reached destination
    if destination_reached:
        score += 0.60

    # +0.2: reached within time limit — infer from step count and route times
    if destination_reached:
        total_time = _estimate_total_time(trajectory)
        if total_time <= 35.0:
            score += 0.20
        else:
            # Partial credit proportional to how close
            ratio = 35.0 / max(total_time, 1.0)
            score += 0.10 * ratio

    # +0.2: no unnecessary reroutes or premature stops
    unnecessary_actions = _count_unnecessary_reroutes(trajectory)
    stop_actions = sum(1 for t in trajectory if t["action"].action_type == "stop")
    if unnecessary_actions == 0 and stop_actions == 0:
        score += 0.20
    else:
        # Partial credit: deduct per unnecessary action
        score += max(0.0, 0.20 - 0.05 * (unnecessary_actions + stop_actions))

    return round(_clamp(score), 4)


def _was_stopped(trajectory: List[dict]) -> bool:
    return any(t["action"].action_type == "stop" for t in trajectory)


def _estimate_total_time(trajectory: List[dict]) -> float:
    """Estimate total travel time from the selected route's base time."""
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            for route in obs.available_routes:
                if route.route_id == action.route_id:
                    return route.estimated_time_min
    # Default: use fastest route time from first obs
    first_obs = trajectory[0]["obs"]
    for route in first_obs.available_routes:
        if route.route_id == "fastest":
            return route.estimated_time_min
    return 999.0


def _count_unnecessary_reroutes(trajectory: List[dict]) -> int:
    """Count reroute actions where the current route had no incidents."""
    count = 0
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        if action.action_type == "reroute" and obs.current_route:
            current = next((r for r in obs.available_routes if r.route_id == obs.current_route), None)
            if current and current.incident_count == 0:
                count += 1
    return count
