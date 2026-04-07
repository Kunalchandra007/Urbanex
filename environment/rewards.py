"""
Reward computation for URBANEX.

Includes risk exposure, indecision penalty, and recovery bonus terms.
"""
from typing import Optional

from models.action import Action
from models.observation import Observation
from models.reward import Reward


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class RewardCalculator:
    """
    Stateful reward function that tracks recent decisions for indecision penalty.
    """

    def __init__(self):
        self._recent_routes: list = []   # last 3 route selections
        self._reroute_count: int = 0

    def reset(self):
        self._recent_routes = []
        self._reroute_count = 0

    def compute_step_reward(
        self,
        action: Action,
        prev_obs: Observation,
        new_obs: Observation,
        task_config: dict,
        hidden_penalty: float = 0.0,    # latent incident penalty (delayed consequence)
    ) -> Reward:
        """
        Reward breakdown — v2:

        POSITIVE:
          +1.0   reached destination (done=True, no crash)
          +0.3   selected safe route when incidents > 0
          +0.2   reported a real incident correctly
          +0.1   reduced distance to destination this step
          +0.15  chose eco route when fuel_efficiency is task goal
          +0.25  RECOVERY BONUS: rerouted AND next route has lower hidden_risk

        NEGATIVE:
          -0.4 * hidden_risk  RISK EXPOSURE: penalty proportional to chosen route risk
          -0.5  selected route with high-severity incident on it
          -0.3  rerouted unnecessarily (route had no incidents, risk was low)
          -0.25 INDECISION PENALTY: >2 reroutes in last 3 steps
          -0.8  selected route with accident/flooding (severe types)
          -1.0  stop action before reaching destination

        Per-step total clamped to [-1.0, 1.0]. Episode total NOT clamped.
        """
        weights = task_config.get("reward_weights", {"safety": 0.4, "time": 0.4, "fuel": 0.2})
        safety_w = weights.get("safety", 0.4)
        time_w = weights.get("time", 0.4)
        fuel_w = weights.get("fuel", 0.2)

        safety_comp = 0.0
        time_comp = 0.0
        fuel_comp = 0.0
        penalty = 0.0
        reasons = []

        # ── Terminal: reached destination ───────────────────────────────────
        if new_obs.episode_done and action.action_type != "stop":
            safety_comp += 1.0 * safety_w
            time_comp += 1.0 * time_w
            fuel_comp += 1.0 * fuel_w
            reasons.append("reached destination")

        # ── STOP before destination ──────────────────────────────────────────
        if action.action_type == "stop" and not new_obs.episode_done:
            penalty -= 1.0
            reasons.append("abandoned journey early")

        # ── Route selection ──────────────────────────────────────────────────
        if action.action_type in ("select_route", "reroute") and action.route_id:
            selected_route = next(
                (r for r in prev_obs.available_routes if r.route_id == action.route_id),
                None,
            )
            if selected_route:
                # ── RISK EXPOSURE: penalty proportional to hidden risk on chosen route ──
                risk_penalty = -0.4 * selected_route.hidden_risk_prob * safety_w
                penalty += risk_penalty
                if selected_route.hidden_risk_prob > 0.25:
                    reasons.append(f"high hidden risk ({selected_route.hidden_risk_prob:.2f}) on {action.route_id}")

                # ── Positive: chose safe route when there are active incidents ──
                if action.action_type == "select_route" and len(prev_obs.active_incidents) > 0:
                    if action.route_id == "safe":
                        safety_comp += 0.3 * safety_w
                        reasons.append("selected safe route with incidents present")

                # ── RECOVERY BONUS: rerouted to lower-risk route ──────────────
                if action.action_type == "reroute":
                    prev_route = prev_obs.current_route
                    if prev_route:
                        prev_route_opt = next(
                            (r for r in prev_obs.available_routes if r.route_id == prev_route),
                            None,
                        )
                        if prev_route_opt and selected_route.hidden_risk_prob < prev_route_opt.hidden_risk_prob - 0.05:
                            safety_comp += 0.25 * safety_w
                            reasons.append(f"recovery reroute to safer route ({action.route_id})")
                    self._reroute_count += 1

                # ── Eco route bonus ────────────────────────────────────────────
                if action.route_id == "eco" and fuel_w >= 0.2:
                    fuel_comp += 0.15 * fuel_w
                    reasons.append("chose eco route for fuel efficiency")

                # ── High-severity / dangerous incident on chosen route ──────────
                high_sev_types = {"accident", "flooding"}
                severe_on_route = [
                    i for i in prev_obs.active_incidents
                    if action.route_id in i.affects_routes and i.severity == "high"
                ]
                for inc in severe_on_route:
                    if inc.type in high_sev_types:
                        penalty -= 0.8
                        reasons.append(f"chose route with {inc.type} (high severity)")
                    else:
                        penalty -= 0.5
                        reasons.append(f"chose route with high-severity {inc.type}")

                # ── INDECISION PENALTY: churning routes too frequently ─────────
                self._recent_routes.append(action.route_id)
                if len(self._recent_routes) > 3:
                    self._recent_routes = self._recent_routes[-3:]
                if len(self._recent_routes) == 3:
                    unique = len(set(self._recent_routes))
                    if unique == 3:   # changed every single step
                        penalty -= 0.25 * safety_w
                        reasons.append("indecision: switched route every step")

                # ── Unnecessary reroute on clean route ───────────────────────
                if action.action_type == "reroute" and selected_route.incident_count == 0 \
                        and selected_route.hidden_risk_prob < 0.15 \
                        and len(prev_obs.active_incidents) == 0:
                    penalty -= 0.3
                    reasons.append("rerouted unnecessarily from clean route")

        # ── Continue / progress ──────────────────────────────────────────────
        if action.action_type == "continue":
            dist_before = prev_obs.distance_remaining_km
            dist_after = new_obs.distance_remaining_km
            if dist_after < dist_before:
                time_comp += 0.1 * time_w
                reasons.append("progressing toward destination")
            # Risk exposure while continuing on a risky route
            if prev_obs.current_route:
                cur_route_opt = next(
                    (r for r in prev_obs.available_routes if r.route_id == prev_obs.current_route),
                    None,
                )
                if cur_route_opt and cur_route_opt.hidden_risk_prob > 0.3:
                    penalty -= 0.15 * safety_w * cur_route_opt.hidden_risk_prob
                    reasons.append(f"continuing on risky route (hidden_risk={cur_route_opt.hidden_risk_prob:.2f})")

        # ── Report incident ────────────────────────────────────────────────
        if action.action_type == "report_incident" and action.incident_type:
            matching = [i for i in prev_obs.active_incidents if i.type == action.incident_type]
            if matching:
                safety_comp += 0.2 * safety_w
                reasons.append(f"reported real incident: {action.incident_type}")
            else:
                penalty -= 0.15
                reasons.append("false alarm incident report")

        # ── DELAYED CONSEQUENCE: hidden incident from 2 steps ago ────────────
        if hidden_penalty > 0.0:
            penalty -= hidden_penalty
            reasons.append(f"hidden incident consequence (delayed -{hidden_penalty:.2f})")

        # ── Time delay penalty beyond baseline ────────────────────────────────
        if prev_obs.current_route:
            max_steps = task_config.get("max_steps", 10)
            if prev_obs.step > max_steps * 0.6:
                time_comp -= 0.04 * time_w
                reasons.append("delayed beyond expected route time")

        total = _clamp(safety_comp + time_comp + fuel_comp + penalty)

        # Guarantee non-zero signal every step
        if total == 0.0:
            total = 0.01
            reasons.append("small progress signal")

        return Reward(
            total=round(total, 4),
            safety_component=round(safety_comp, 4),
            time_component=round(time_comp, 4),
            fuel_component=round(fuel_comp, 4),
            penalty=round(penalty, 4),
            reason="; ".join(reasons) if reasons else "no significant event",
        )
