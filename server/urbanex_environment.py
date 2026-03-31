"""
URBANEX Environment — OpenEnv-compliant wrapper around VeloraEnv.

This module bridges the game logic (environment/) with the OpenEnv server interface.
"""

from uuid import uuid4
from typing import Optional

from openenv.core.env_server import Environment, State

from environment.velora_env import VeloraEnv
from environment.rewards import RewardCalculator
from models import UrbanexAction, UrbanexObservation, RouteOption, Incident, Reward
from graders import grade_easy, grade_medium, grade_hard


class UrbanexEnvironment(Environment):
    """
    OpenEnv-compliant environment wrapper for URBANEX.
    
    Implements the Environment interface required by openenv.core.env_server.create_app()
    """

    def __init__(self):
        """Initialize the environment."""
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._env: Optional[VeloraEnv] = None
        self._state = State(episode_id=self._episode_id, step_count=self._step_count)
        self._last_reward: Reward = Reward()
        self._grader_map = {
            "easy": grade_easy,
            "medium": grade_medium,
            "hard": grade_hard,
        }

    def reset(self, task: str = "easy", seed: int = 42, **kwargs) -> UrbanexObservation:
        """
        Reset the environment for a new episode.
        
        Args:
            task: Task difficulty ("easy", "medium", "hard")
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Initial observation
        """
        # Create new episode
        self._episode_id = str(uuid4())
        self._step_count = 0
        self._state = State(episode_id=self._episode_id, step_count=self._step_count)

        # Initialize environment
        self._env = VeloraEnv(task=task, seed=seed)
        obs = self._env.reset()

        # Convert VeloraEnv observation to OpenEnv observation
        return self._convert_observation(obs, done=False)

    def step(self, action: UrbanexAction) -> tuple[UrbanexObservation, float, bool]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            (observation, reward, done) tuple
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Execute step in VeloraEnv
        obs, reward, done, info = self._env.step(action)

        # Update state
        self._step_count += 1
        self._state.step_count = self._step_count
        self._last_reward = reward

        # Convert observation
        urbanex_obs = self._convert_observation(obs, done=done)

        # Return OpenEnv format: (observation, reward_float, done)
        return urbanex_obs, float(reward.total), done

    @property
    def state(self) -> State:
        """Get current episode state."""
        return self._state

    def _convert_observation(self, velora_obs, done: bool = False) -> UrbanexObservation:
        """Convert VeloraEnv observation to UrbanexObservation."""
        return UrbanexObservation(
            step=velora_obs.step,
            current_location=velora_obs.current_location,
            destination=velora_obs.destination,
            available_routes=[
                RouteOption(
                    route_id=r.route_id,
                    estimated_time_min=r.estimated_time_min,
                    distance_km=getattr(r, "distance_km", 0.0),
                    incident_count=r.incident_count,
                    fuel_cost_score=r.fuel_cost_score,
                    safety_score=r.safety_score,
                    hidden_risk_prob=r.hidden_risk_prob,
                )
                for r in velora_obs.available_routes
            ],
            active_incidents=[
                Incident(
                    incident_id=i.incident_id,
                    type=i.type,
                    severity=i.severity,
                    lat=i.lat,
                    lng=i.lng,
                    affects_routes=i.affects_routes,
                )
                for i in velora_obs.active_incidents
            ],
            traffic_level=velora_obs.traffic_level,
            weather=velora_obs.weather,
            current_route=velora_obs.current_route,
            distance_remaining_km=velora_obs.distance_remaining_km,
            episode_done=velora_obs.episode_done,
            reward=float(self._last_reward.total),
        )


__all__ = ["UrbanexEnvironment"]
