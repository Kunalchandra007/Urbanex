"""
OpenEnv Client for URBANEX environment.

Provides EnvClient wrapper with WebSocket support for use with OpenEnv framework.
"""

from typing import Any, Optional
import asyncio

from openenv.core import EnvClient, StepResult

from models import UrbanexAction, UrbanexObservation


class UrbanexEnv(EnvClient[UrbanexAction, UrbanexObservation, Any]):
    """
    OpenEnv-compliant async client for URBANEX environment.
    
    Connects to a running URBANEX FastAPI server via WebSocket.
    Provides step/reset/state interface per OpenEnv specification.
    """

    action_type = UrbanexAction
    observation_type = UrbanexObservation

    def __init__(self, base_url: str = "http://localhost:7860"):
        """
        Initialize client with server base URL.
        
        Args:
            base_url: Base URL of the URBANEX server (HTTP or HTTPS)
        """
        super().__init__(base_url=base_url)

    def _step_payload(self, action: UrbanexAction) -> dict[str, Any]:
        """Convert action to WebSocket payload."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[UrbanexObservation]:
        """Parse WebSocket response to StepResult."""
        obs_data = payload.get("observation", {})
        observation = UrbanexObservation(**obs_data)
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse state response."""
        return payload


async def demo():
    """Demo of async client usage."""
    async with UrbanexEnv(base_url="http://localhost:7860") as client:
        # Reset to start episode
        obs = await client.reset()
        print(f"Episode started at {obs.current_location}")

        # Play for a few steps
        for step in range(5):
            action = UrbanexAction(action_type="continue")
            result = await client.step(action)
            print(f"Step {step}: reward={result.reward}, done={result.done}")

            if result.done:
                print("Episode complete!")
                break


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo())


__all__ = ["UrbanexEnv", "demo"]

