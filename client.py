"""
OpenEnv Client for URBANEX environment.

Provides EnvClient wrapper around the FastAPI server for use with OpenEnv framework.
"""

from typing import Any, Optional, Dict
import httpx
import json

try:
    from openenv.core import EnvClient, StepResult
except ImportError:
    # Fallback stubs for development without openenv installed
    class StepResult:
        def __init__(self, observation: Any, reward: Optional[float] = None, done: bool = False):
            self.observation = observation
            self.reward = reward
            self.done = done

    class EnvClient:
        def __init__(self, base_url: str = "http://localhost:8000"):
            self.base_url = base_url


from models import Action, Observation


class UrbanexEnv(EnvClient):
    """
    OpenEnv client for URBANEX environment.
    
    Connects to a running URBANEX FastAPI server (local or remote).
    Provides async step/reset/state interface per OpenEnv specification.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client with server base URL."""
        super().__init__(base_url=base_url)
        self._client: Optional[httpx.AsyncClient] = None
        self._task: str = "easy"
        self._seed: int = 42

    async def connect(self) -> None:
        """Establish connection to server."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        # Verify server is reachable with health check
        try:
            response = await self._client.get("/health", follow_redirects=True)
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to URBANEX server at {self.base_url}: {e}")

    async def close(self) -> None:
        """Close connection to server."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection."""
        if self._client is None:
            await self.connect()

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        """Convert Action to JSON payload for server."""
        return action.dict()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Parse server response to StepResult."""
        obs_data = payload.get("observation", {})
        observation = Observation(**obs_data)
        reward = payload.get("reward", 0.0)
        done = payload.get("done", False)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse server state response."""
        return payload

    async def reset(self, task: str = "easy", seed: int = 42, **kwargs) -> Observation:
        """
        Reset environment and return initial observation.
        
        Args:
            task: Task difficulty ('easy', 'medium', or 'hard')
            seed: Random seed for reproducibility
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Initial observation
        """
        await self._ensure_connected()
        self._task = task
        self._seed = seed
        
        try:
            response = await self._client.post(
                "/reset",
                json={"task": task, "seed": seed}
            )
            response.raise_for_status()
            result = response.json()
            obs_data = result.get("observation", {})
            return Observation(**obs_data)
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment: {e}")

    async def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            StepResult with observation, reward, done flag
        """
        await self._ensure_connected()
        
        try:
            payload = self._step_payload(action)
            response = await self._client.post("/step", json={"action": payload})
            response.raise_for_status()
            result = response.json()
            return self._parse_result(result)
        except Exception as e:
            raise RuntimeError(f"Failed to execute step: {e}")

    async def state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            Dictionary with episode metadata
        """
        await self._ensure_connected()
        
        try:
            response = await self._client.get("/state")
            response.raise_for_status()
            result = response.json()
            return self._parse_state(result)
        except Exception as e:
            raise RuntimeError(f"Failed to get state: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def sync(self):
        """Return a synchronous wrapper (stub for compatibility)."""
        # In a full implementation, would return SyncEnvClient wrapper
        # For now, just return self with sync methods
        raise NotImplementedError("Use async interface: async with UrbanexEnv() as env: ...")


__all__ = ["UrbanexEnv"]
