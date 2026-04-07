"""
Grader for Hard task: dynamic_rerouting — v2.
Adds decision_stability component.
"""
from typing import List


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _finalize_score(value: float) -> float:
    """
    OpenEnv hackathon validator expects task scores to be strictly inside (0, 1),
    not equal to the boundaries.
    """
    return round(_clamp(value, 0.0001, 0.9999), 4)


def grade(trajectory: List[dict]) -> float:
    """
    +0.4  if destination reached
    +0.2  if rerouted at least once when an incident appeared on current route
    +0.2  if avg safety score > 0.6
    +0.1  if adapted route after traffic escalation (step 5 or 10)
    +0.1  if adapted route after weather change (step 7 or 13)
    -0.1  per step over max_steps (18)
    -0.05 per 'indecision cycle' (switching same route pairs back and forth)
    decision_stability bonus: up to +0.1 for consistent route-keeping under pressure
    """
    if not trajectory:
        return 0.0

    score = 0.0
    last_obs = trajectory[-1]["obs"]
    total_steps = len(trajectory)

    # +0.4: destination reached (not via stop)
    destination_reached = last_obs.episode_done and not any(
        t["action"].action_type == "stop" for t in trajectory
    )
    if destination_reached:
        score += 0.40

    # +0.2: rerouted at least once when incident was on current route
    if _rerouted_on_incident(trajectory):
        score += 0.20

    # +0.2: avg safety > 0.6
    avg_safety = _avg_safety_across_steps(trajectory)
    if avg_safety > 0.6:
        score += 0.20
    else:
        score += 0.20 * (avg_safety / 0.6)

    # +0.1: adapted route after traffic escalation (step 5 or 10)
    if _adapted_after_step(trajectory, change_step=5) or _adapted_after_step(trajectory, change_step=10):
        score += 0.10

    # +0.1: adapted route after weather change (step 7 or 13)
    if _adapted_after_step(trajectory, change_step=7) or _adapted_after_step(trajectory, change_step=13):
        score += 0.10

    # DECISION STABILITY: bonus for consistent route-keeping under pressure
    stability = _decision_stability_score(trajectory)
    score += 0.10 * stability   # up to +0.1 from stability bonus

    # Indecision penalty: repeated zigzag between same two routes
    zigzag = _count_zigzag_patterns(trajectory)
    score -= 0.05 * zigzag

    # -0.1 per step over 18
    if total_steps > 18:
        score -= 0.10 * (total_steps - 18)

    return _finalize_score(score)


def _rerouted_on_incident(trajectory: List[dict]) -> bool:
    for step in trajectory:
        obs = step["obs"]
        action = step["action"]
        if action.action_type == "reroute" and obs.current_route:
            for inc in obs.active_incidents:
                if obs.current_route in inc.affects_routes:
                    return True
    return False


def _avg_safety_across_steps(trajectory: List[dict]) -> float:
    scores = []
    current_route = None
    for step in trajectory:
        action = step["action"]
        obs = step["obs"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            current_route = action.route_id
        if current_route:
            for route in obs.available_routes:
                if route.route_id == current_route:
                    scores.append(route.safety_score)
    return sum(scores) / len(scores) if scores else 0.0


def _adapted_after_step(trajectory: List[dict], change_step: int) -> bool:
    route_before = _get_route_around_step(trajectory, change_step - 1)
    route_after = _get_route_around_step(trajectory, change_step + 2)
    return bool(route_before and route_after and route_before != route_after)


def _get_route_around_step(trajectory: List[dict], target_step: int) -> str:
    best = None
    best_dist = float("inf")
    current_route = None
    for step in trajectory:
        obs = step["obs"]
        action = step["action"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            current_route = action.route_id
        dist = abs(obs.step - target_step)
        if dist < best_dist:
            best_dist = dist
            best = current_route
    return best or ""


def _decision_stability_score(trajectory: List[dict]) -> float:
    """
    Returns 0.0-1.0. Higher = more consistent route choices without zigzagging.
    Measured as: fraction of steps where the route stayed the same as previous step.
    """
    if len(trajectory) < 2:
        return 1.0
    route_sequence = []
    current = None
    for step in trajectory:
        action = step["action"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            current = action.route_id
        if current:
            route_sequence.append(current)
    if len(route_sequence) < 2:
        return 1.0
    stable = sum(1 for a, b in zip(route_sequence, route_sequence[1:]) if a == b)
    return stable / (len(route_sequence) - 1)


def _count_zigzag_patterns(trajectory: List[dict]) -> int:
    """Count instances of A→B→A route switching (indecision zigzag)."""
    routes = []
    for step in trajectory:
        action = step["action"]
        if action.action_type in ("select_route", "reroute") and action.route_id:
            routes.append(action.route_id)
    count = 0
    for i in range(len(routes) - 2):
        if routes[i] == routes[i + 2] and routes[i] != routes[i + 1]:
            count += 1
    return count
