"""
Incident types, spawning, and management for URBANEX.

Ported from the original app-side incident and navigation providers.
"""
import random
import uuid
from typing import Dict, List, Optional, Tuple

from models.observation import Incident

INCIDENT_TYPES: Dict[str, Dict] = {
    "pothole":      {"severity_range": ["low", "medium"],    "time_penalty_min": 2,  "safety_impact": 0.10},
    "accident":     {"severity_range": ["medium", "high"],   "time_penalty_min": 8,  "safety_impact": 0.40},
    "flooding":     {"severity_range": ["medium", "high"],   "time_penalty_min": 12, "safety_impact": 0.50},
    "construction": {"severity_range": ["low", "medium"],    "time_penalty_min": 5,  "safety_impact": 0.15},
    "breakdown":    {"severity_range": ["low"],              "time_penalty_min": 3,  "safety_impact": 0.05},
}

SEVERITY_WEIGHTS = {"low": 0, "medium": 1, "high": 2}
ALL_ROUTE_IDS = ["fastest", "safe", "eco"]


def _pick_severity(severity_range: List[str], rng: random.Random) -> str:
    return rng.choice(severity_range)


class IncidentManager:
    """
    Manages active incidents: spawning, aging, querying by route.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._incidents: List[Incident] = []
        self._incident_lifetimes: Dict[str, int] = {}  # incident_id -> steps remaining

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._incidents = []
        self._incident_lifetimes = {}

    def spawn_incident(
        self,
        incident_type: Optional[str] = None,
        affects_routes: Optional[List[str]] = None,
        location: Optional[Tuple[float, float]] = None,
        lifetime_steps: int = 5,
    ) -> Incident:
        """Spawn a new incident with given or random properties."""
        if incident_type is None:
            incident_type = self._rng.choice(list(INCIDENT_TYPES.keys()))
        profile = INCIDENT_TYPES[incident_type]
        severity = _pick_severity(profile["severity_range"], self._rng)

        if affects_routes is None:
            # Random subset of routes
            n = self._rng.randint(1, len(ALL_ROUTE_IDS))
            affects_routes = self._rng.sample(ALL_ROUTE_IDS, n)

        if location is None:
            # Default near Bangalore center with small random offset
            lat = 12.9716 + self._rng.uniform(-0.05, 0.05)
            lng = 77.5946 + self._rng.uniform(-0.05, 0.05)
            location = (round(lat, 4), round(lng, 4))

        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            type=incident_type,
            severity=severity,
            lat=location[0],
            lng=location[1],
            affects_routes=affects_routes,
        )
        self._incidents.append(incident)
        self._incident_lifetimes[incident.incident_id] = lifetime_steps
        return incident

    def place_incident(
        self,
        incident_type: str,
        severity: str,
        affects_routes: List[str],
        location: Tuple[float, float],
    ) -> Incident:
        """Place a pre-configured incident (used for deterministic task setup)."""
        incident = Incident(
            incident_id=str(uuid.uuid4())[:8],
            type=incident_type,
            severity=severity,
            lat=location[0],
            lng=location[1],
            affects_routes=affects_routes,
        )
        self._incidents.append(incident)
        self._incident_lifetimes[incident.incident_id] = 999  # persistent unless cleared
        return incident

    def tick(self) -> None:
        """Advance time — reduce lifetimes and remove expired incidents."""
        expired = []
        for inc_id, lifetime in self._incident_lifetimes.items():
            self._incident_lifetimes[inc_id] = lifetime - 1
            if self._incident_lifetimes[inc_id] <= 0:
                expired.append(inc_id)
        self._incidents = [i for i in self._incidents if i.incident_id not in expired]
        for inc_id in expired:
            del self._incident_lifetimes[inc_id]

    def get_incidents_on_route(self, route_id: str) -> List[Incident]:
        return [i for i in self._incidents if route_id in i.affects_routes]

    def get_all_incidents(self) -> List[Incident]:
        return list(self._incidents)

    def clear(self) -> None:
        self._incidents = []
        self._incident_lifetimes = {}

    def max_severity_on_route(self, route_id: str) -> Optional[str]:
        incidents = self.get_incidents_on_route(route_id)
        if not incidents:
            return None
        return max(incidents, key=lambda i: SEVERITY_WEIGHTS.get(i.severity, 0)).severity

    def total_time_penalty_on_route(self, route_id: str) -> float:
        """Sum of time penalties for all incidents on a given route."""
        total = 0.0
        for inc in self.get_incidents_on_route(route_id):
            total += INCIDENT_TYPES.get(inc.type, {}).get("time_penalty_min", 0)
        return total

    def total_safety_impact_on_route(self, route_id: str) -> float:
        """Cumulative safety impact (0→1) from incidents on a route."""
        impact = 0.0
        for inc in self.get_incidents_on_route(route_id):
            impact += INCIDENT_TYPES.get(inc.type, {}).get("safety_impact", 0)
        return min(impact, 1.0)
