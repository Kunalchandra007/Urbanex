"""
URBANEX models — OpenEnv-compliant data types.

These extend openenv.core base types for full SDK compatibility.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# Action Type
class UrbanexAction(BaseModel):
    """Action that an agent can take in URBANEX."""
    action_type: Literal[
        "select_route",    # choose a route to follow
        "reroute",         # switch to a different route mid-journey
        "report_incident", # report a new incident
        "continue",        # keep going on current route
        "stop"             # abandon journey
    ] = Field(..., description="Type of action to take")
    route_id: Optional[str] = Field(None, description="Route ID (for select_route/reroute)")
    incident_type: Optional[str] = Field(None, description="Incident type (for report_incident)")
    incident_lat: Optional[float] = Field(None, description="Incident latitude")
    incident_lng: Optional[float] = Field(None, description="Incident longitude")


# Observation Components
class RouteOption(BaseModel):
    """A route option available to the agent."""
    route_id: str = Field(..., description="Route identifier: 'fastest', 'safe', or 'eco'")
    estimated_time_min: float = Field(..., description="Estimated travel time in minutes")
    distance_km: float = Field(default=0.0, description="Distance in kilometers")
    incident_count: int = Field(default=0, description="Number of incidents on this route")
    fuel_cost_score: float = Field(default=0.0, description="Fuel cost score (0.0-1.0)")
    safety_score: float = Field(default=1.0, description="Safety score (0.0-1.0)")
    hidden_risk_prob: float = Field(default=0.0, description="Probability of hidden risks (0.0-1.0)")


class Incident(BaseModel):
    """A traffic incident on the road network."""
    incident_id: str = Field(..., description="Unique incident identifier")
    type: str = Field(..., description="Incident type: accident, roadwork, flooding, etc.")
    severity: Literal["low", "medium", "high"] = Field(..., description="Incident severity")
    lat: float = Field(..., description="Latitude of incident")
    lng: float = Field(..., description="Longitude of incident")
    affects_routes: List[str] = Field(default_factory=list, description="Route IDs this incident affects")


# Observation Type
class UrbanexObservation(BaseModel):
    """Complete observation of the environment state."""
    step: int = Field(default=0, description="Current step number")
    current_location: List[float] = Field(..., description="Current location as [lat, lng]")
    destination: List[float] = Field(..., description="Destination as [lat, lng]")
    available_routes: List[RouteOption] = Field(default_factory=list, description="Available route options")
    active_incidents: List[Incident] = Field(default_factory=list, description="Active incidents")
    traffic_level: Literal["low", "medium", "high"] = Field("low", description="Current traffic level")
    weather: Literal["clear", "rain", "fog"] = Field("clear", description="Current weather")
    current_route: Optional[str] = Field(None, description="Currently selected route ID")
    distance_remaining_km: float = Field(default=0.0, description="Distance to destination in km")
    episode_done: bool = Field(default=False, description="Whether episode is complete")
    reward: float = Field(default=0.0, description="Reward for this step")


# Reward breakdown (for transparency)
class Reward(BaseModel):
    """Detailed reward breakdown."""
    total: float = Field(default=0.0, description="Total reward")
    safety_component: float = Field(default=0.0, description="Safety reward component")
    time_component: float = Field(default=0.0, description="Time efficiency component")
    fuel_component: float = Field(default=0.0, description="Fuel efficiency component")
    penalty: float = Field(default=0.0, description="Applied penalties")
    reason: str = Field(default="", description="Reward reason/explanation")


__all__ = [
    "UrbanexAction",
    "UrbanexObservation",
    "RouteOption",
    "Incident",
    "Reward",
]
