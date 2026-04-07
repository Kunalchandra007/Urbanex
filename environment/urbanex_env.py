"""
Main environment class: UrbanexEnv.

This is the core simulation loop for URBANEX. It coordinates the city graph,
incident manager, route calculator, and reward model across reset/step cycles.
"""
import random
from collections import deque
from typing import Deque, Optional, Tuple

from environment.city import CityGraph
from environment.incidents import IncidentManager
from environment.rewards import RewardCalculator
from environment.routes import RouteCalculator
from models.action import Action
from models.observation import Observation
from models.reward import Reward
from tasks.task_easy import EASY_CONFIG
from tasks.task_hard import HARD_CONFIG
from tasks.task_medium import MEDIUM_CONFIG

TASK_CONFIGS = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}


class UrbanexEnv:
    """
    URBANEX urban navigation environment.

    Features:
    - hidden route risk for partial observability
    - delayed penalties from latent incidents
    - a medium-task greedy trap on the fastest route
    - dynamic traffic, weather, and incident changes in hard mode
    """

    def __init__(self, task: str = "easy", seed: int = 42):
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{task}'. Choose: {list(TASK_CONFIGS.keys())}")
        self.task = task
        self.seed = seed
        self.config = TASK_CONFIGS[task]
        self._rng = random.Random(seed)

        self._city = CityGraph(seed=seed)
        self._incidents = IncidentManager(seed=seed)
        self._routes = RouteCalculator()
        self._rewards = RewardCalculator()

        self._step = 0
        self._current_route: Optional[str] = None
        self._done = False
        self._trajectory = []
        self._traffic_level = self.config.get("traffic_level", "low")
        self._weather = self.config.get("weather", "clear")
        self._prev_obs: Optional[Observation] = None

        self._pending_penalties: Deque[Tuple[int, float]] = deque()
        self._greedy_trap_fired = False

    def reset(self, origin_name: Optional[str] = None, dest_name: Optional[str] = None) -> Observation:
        """Reset the environment and return the initial observation."""
        self._rng = random.Random(self.seed)
        self._step = 0
        self._done = False
        self._current_route = None
        self._trajectory = []
        self._traffic_level = self.config.get("traffic_level", "low")
        self._weather = self.config.get("weather", "clear")
        self._pending_penalties = deque()
        self._greedy_trap_fired = False
        self._rewards.reset()

        self._city.reset(origin_name=origin_name, dest_name=dest_name)
        self._incidents.reset()

        for inc_cfg in self.config.get("setup_incidents", []):
            loc = self._city.random_waypoint_near(inc_cfg["affects_routes"][0])
            self._incidents.place_incident(
                incident_type=inc_cfg["type"],
                severity=inc_cfg["severity"],
                affects_routes=inc_cfg["affects_routes"],
                location=loc,
            )

        obs = self._build_observation()
        self._prev_obs = obs
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, dict]:
        """Apply an action and advance the simulation one step."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        prev_obs = self._prev_obs or self._build_observation()
        self._apply_action(action)
        self._advance_city()
        hidden_penalty = self._collect_pending_penalty()

        new_obs = self._build_observation()
        done = self._is_done()
        new_obs = new_obs.model_copy(update={"episode_done": done})
        self._done = done

        reward = self._rewards.compute_step_reward(
            action=action,
            prev_obs=prev_obs,
            new_obs=new_obs,
            task_config=self.config,
            hidden_penalty=hidden_penalty,
        )

        self._prev_obs = new_obs
        self._trajectory.append({"obs": new_obs, "action": action, "reward": reward})

        info = {
            "step": self._step,
            "traffic_level": self._traffic_level,
            "weather": self._weather,
            "distance_remaining_km": new_obs.distance_remaining_km,
            "hidden_penalty_this_step": hidden_penalty,
            "pending_penalties": len(self._pending_penalties),
        }
        return new_obs, reward, done, info

    def state(self) -> dict:
        return {
            "task": self.task,
            "step": self._step,
            "done": self._done,
            "current_route": self._current_route,
            "traffic_level": self._traffic_level,
            "weather": self._weather,
            "current_location": list(self._city.current_location),
            "destination": list(self._city.destination),
            "distance_remaining_km": self._city.distance_remaining_km(),
            "active_incidents": [i.model_dump() for i in self._incidents.get_all_incidents()],
            "trajectory_length": len(self._trajectory),
            "greedy_trap_fired": self._greedy_trap_fired,
        }

    def get_observation(self) -> Observation:
        """Return the current observation without advancing the episode."""
        return self._prev_obs or self._build_observation()

    def _apply_action(self, action: Action) -> None:
        if action.action_type in ("select_route", "reroute"):
            if action.route_id:
                self._current_route = action.route_id
        elif action.action_type == "report_incident":
            if action.incident_type and action.incident_lat and action.incident_lng:
                self._incidents.spawn_incident(
                    incident_type=action.incident_type,
                    location=(action.incident_lat, action.incident_lng),
                    lifetime_steps=3,
                )
        elif action.action_type == "stop":
            self._done = True

    def _advance_city(self) -> None:
        self._step += 1

        if self._current_route:
            self._city.move_towards_destination(fraction=0.30)

        self._incidents.tick()

        if self._current_route:
            routes = self._build_observation().available_routes
            current_route = next((r for r in routes if r.route_id == self._current_route), None)
            if current_route:
                roll = self._rng.random()
                threshold = current_route.hidden_risk_prob * self.config.get("hidden_risk_scale", 0.4)
                if roll < threshold:
                    delay = 2
                    penalty_amount = round(current_route.hidden_risk_prob * 0.5, 3)
                    self._pending_penalties.append((self._step + delay, penalty_amount))

        trap_step = self.config.get("greedy_trap_step")
        if trap_step and self._step == trap_step and not self._greedy_trap_fired:
            if self._current_route == "fastest":
                trap_penalty = self.config.get("greedy_trap_penalty", 0.4)
                self._pending_penalties.append((self._step, trap_penalty))
                self._greedy_trap_fired = True

        if self.config.get("dynamic_incidents", False):
            for spawn_step in self.config.get("spawn_steps", []):
                if self._step == spawn_step:
                    loc = self._city.random_waypoint_near("fastest")
                    routes_to_affect = (
                        ["fastest"]
                        if self._rng.random() < 0.70
                        else self._rng.sample(["fastest", "safe", "eco"], 2)
                    )
                    self._incidents.spawn_incident(
                        location=loc,
                        affects_routes=routes_to_affect,
                        lifetime_steps=4,
                    )

        t1 = self.config.get("traffic_change_step")
        t2 = self.config.get("traffic_change_step_2")
        if t2 and self._step >= t2:
            self._traffic_level = self.config.get("traffic_after_2", self._traffic_level)
        elif t1 and self._step >= t1:
            self._traffic_level = self.config.get("traffic_after", self._traffic_level)

        w1 = self.config.get("weather_change_step")
        w2 = self.config.get("weather_change_step_2")
        if w2 and self._step >= w2:
            self._weather = self.config.get("weather_after_2", self._weather)
        elif w1 and self._step >= w1:
            self._weather = self.config.get("weather_after", self._weather)

    def _collect_pending_penalty(self) -> float:
        """Pop all penalties due at the current step."""
        total = 0.0
        remaining = deque()
        for due_step, amount in self._pending_penalties:
            if due_step <= self._step:
                total += amount
            else:
                remaining.append((due_step, amount))
        self._pending_penalties = remaining
        return round(total, 4)

    def _build_observation(self) -> Observation:
        routes = self._routes.calculate_routes(
            incident_manager=self._incidents,
            traffic_level=self._traffic_level,
            weather=self._weather,
            step=self._step,
            rng=self._rng,
        )
        return Observation(
            step=self._step,
            current_location=list(self._city.current_location),
            destination=list(self._city.destination),
            available_routes=routes,
            active_incidents=self._incidents.get_all_incidents(),
            traffic_level=self._traffic_level,
            weather=self._weather,
            current_route=self._current_route,
            distance_remaining_km=round(self._city.distance_remaining_km(), 3),
            episode_done=self._done,
        )

    def _is_done(self) -> bool:
        if self._done:
            return True
        if self._city.is_at_destination():
            return True
        if self._step >= self.config.get("max_steps", 10):
            return True
        return False
