"""
Grader for Medium task: avoid_and_optimize — v2.
Adds greedy trap deduction and efficiency ratio.
"""
from typing import List


BASELINE_TIME_MIN = 28.0  # safe route baseline


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def grade(trajectory: List[dict]) -> float:
    """
    +0.5  if destination reached
    +0.3  if zero high-severity incidents encountered on chosen routes
    +0.1  if avg safety score of selected routes > 0.7
    +0.1  if completed under 1.2x baseline time
    -0.15 per high-severity incident encountered
    -0.20 if greedy trap triggered (fastest chosen at trap step)
    -0.10 if efficiency_ratio > 1.5 (took far too many steps)
    """
    if not trajectory:
        return 0.0

    score = 0.0
    last_obs = trajectory[-1]["obs"]

    destination_reached = last_obs.episode_done and not any(
        t["action"].action_type == "stop" for t in trajectory
    )

    # +0.5: destination reached
    if destination_reached:
        score += 0.50

    # High severity count
    high_sev_count = _count_high_severity_encountered(trajectory)

    # +0.3 / -0.15 per high-sev
    if high_sev_count == 0:
        score += 0.30
    else:
        score -= 0.15 * high_sev_count

    # +0.1: avg safety > 0.7
    avg_safety = _avg_selected_route_safety(trajectory)
    if avg_safety > 0.7:
        score += 0.10
    else:
        score += 0.10 * (avg_safety / 0.7)

    # +0.1: completed under 1.2x baseline
    if destination_reached:
        est_time = _estimate_route_time(trajectory)
        if est_time <= BASELINE_TIME_MIN * 1.2:
            score += 0.10
        else:
            score += 0.05 * ((BASELINE_TIME_MIN * 1.2) / max(est_time, 1.0))

    # -0.20: greedy trap triggered (used fastest route at trap step AND got penalised)
    if _greedy_trap_hit(trajectory):
        score -= 0.20

    # Efficiency ratio: penalise taking excessively many steps
    total_steps = len(trajectory)
    efficiency_ratio = total_steps / max(5.0, 1.0)  # expected ~5-8 steps
    if efficiency_ratio > 2.0:
        score -= 0.10

    return round(_clamp(score), 4)


def _count_high_severity_encountered(trajectory: List[dict]) -> int:
    count = 0
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        chosen = action.route_id if action.action_type in ("select_route", "reroute") else obs.current_route
        if chosen:
            for inc in obs.active_incidents:
                if chosen in inc.affects_routes and inc.severity == "high":
                    count += 1
    return count


def _avg_selected_route_safety(trajectory: List[dict]) -> float:
    scores = []
    current = None
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            current = action.route_id
        if current:
            for r in obs.available_routes:
                if r.route_id == current:
                    scores.append(r.safety_score)
    return sum(scores) / len(scores) if scores else 0.0


def _estimate_route_time(trajectory: List[dict]) -> float:
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            for r in obs.available_routes:
                if r.route_id == action.route_id:
                    return r.estimated_time_min
    return 999.0


def _greedy_trap_hit(trajectory: List[dict]) -> bool:
    """
    Returns True if agent chose 'fastest' in the first 3 steps AND
    a reward penalty was recorded (hidden_penalty was applied).
    """
    for step in trajectory[:4]:
        action = step["action"]
        obs = step["obs"]
        reward = step["reward"]
        if action.action_type in ("select_route",) and action.route_id == "fastest":
            # Greedy trap shows as a large penalty in hidden consequence step
            if reward.penalty < -0.30:
                return True
    return False
