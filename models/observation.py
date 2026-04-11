from pydantic import BaseModel
from typing import List, Optional


class RouteOption(BaseModel):
    route_id: str                  # "fastest" | "safe" | "eco"
    estimated_time_min: float      # e.g. 20.5
    incident_count: int            # number of incidents on this route
    fuel_cost_score: float         # 0.0 (low) to 1.0 (high)
    safety_score: float            # 0.0 (dangerous) to 1.0 (safe)
    hidden_risk_prob: float = 0.0  # PARTIAL OBSERVABILITY: probability of a hidden
                                   # undetected incident on this route (0.0–1.0).
                                   # Agent sees this signal but NOT the actual incident.


class Incident(BaseModel):
    incident_id: str
    type: str                      # "pothole" | "accident" | "flooding" | "construction"
    severity: str                  # "low" | "medium" | "high"
    lat: float
    lng: float
    affects_routes: List[str]      # which route_ids this incident is on


class Observation(BaseModel):
    step: int
    current_location: List[float]  # [lat, lng]
    destination: List[float]       # [lat, lng]
    available_routes: List[RouteOption]
    active_incidents: List[Incident]
    traffic_level: str             # "low" | "medium" | "high"
    weather: str                   # "clear" | "rain" | "fog" | "heavy_rain"
    current_route: Optional[str]   # currently selected route_id or None
    distance_remaining_km: float
    episode_done: bool
    situation_summary: str = ""
