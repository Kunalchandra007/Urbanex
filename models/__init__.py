"""
Shared model exports for URBANEX.

This module keeps the original environment models available at the package
level while also exposing OpenEnv-compatible wrapper models used by the
`client.py` and `server/` integration layer.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
except ModuleNotFoundError:
    class OpenEnvAction(BaseModel):
        """Fallback base model when openenv-core is unavailable."""

        metadata: Dict[str, Any] = Field(default_factory=dict)


    class OpenEnvObservation(BaseModel):
        """Fallback observation model when openenv-core is unavailable."""

        done: bool = Field(default=False)
        reward: float | None = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

from .action import Action
from .observation import Incident, Observation, RouteOption
from .reward import Reward


class UrbanexAction(OpenEnvAction):
    """OpenEnv-compatible action model."""

    action_type: Literal[
        "select_route",
        "reroute",
        "report_incident",
        "continue",
        "stop",
    ] = Field(..., description="Type of action to take")
    route_id: Optional[str] = Field(
        default=None, description="Route ID for select_route / reroute"
    )
    incident_type: Optional[str] = Field(
        default=None, description="Incident type for report_incident"
    )
    incident_lat: Optional[float] = Field(
        default=None, description="Incident latitude"
    )
    incident_lng: Optional[float] = Field(
        default=None, description="Incident longitude"
    )


class UrbanexObservation(OpenEnvObservation):
    """OpenEnv-compatible observation model."""

    step: int = Field(default=0, description="Current step number")
    current_location: list[float] = Field(
        ..., description="Current location as [lat, lng]"
    )
    destination: list[float] = Field(..., description="Destination as [lat, lng]")
    available_routes: list[RouteOption] = Field(
        default_factory=list, description="Available route options"
    )
    active_incidents: list[Incident] = Field(
        default_factory=list, description="Visible incidents"
    )
    traffic_level: Literal["low", "medium", "high"] = Field(
        default="low", description="Current traffic level"
    )
    weather: Literal["clear", "rain", "fog", "heavy_rain"] = Field(
        default="clear", description="Current weather condition"
    )
    current_route: Optional[str] = Field(
        default=None, description="Currently selected route"
    )
    distance_remaining_km: float = Field(
        default=0.0, description="Distance remaining in kilometers"
    )
    episode_done: bool = Field(
        default=False, description="Whether the underlying environment has ended"
    )
    situation_summary: str = Field(
        default="", description="Plain-English summary of the current navigation situation"
    )
    done: bool = Field(default=False, description="OpenEnv episode completion flag")
    reward: float | None = Field(
        default=None, description="Reward associated with the latest transition"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional OpenEnv metadata"
    )


LegacyAction = Action
LegacyObservation = Observation

__all__ = [
    "Action",
    "Observation",
    "RouteOption",
    "Incident",
    "Reward",
    "UrbanexAction",
    "UrbanexObservation",
    "LegacyAction",
    "LegacyObservation",
]
