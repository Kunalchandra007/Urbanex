"""
City simulation module for URBANEX.

Holds Bangalore waypoints, distance computation, and mutable city state.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

BANGALORE_WAYPOINTS: Dict[str, Tuple[float, float]] = {
    "majestic":        (12.9767, 77.5713),
    "koramangala":     (12.9352, 77.6245),
    "indiranagar":     (12.9784, 77.6408),
    "whitefield":      (12.9698, 77.7499),
    "electronic_city": (12.8399, 77.6770),
    "jayanagar":       (12.9250, 77.5938),
    "hebbal":          (13.0350, 77.5970),
    "mg_road":         (12.9756, 77.6097),
    "jp_nagar":        (12.9102, 77.5850),
    "marathahalli":    (12.9591, 77.6974),
}

# Approximate road graph (adjacency) — distances in km
ROAD_CONNECTIONS: List[Tuple[str, str, float]] = [
    ("majestic",        "mg_road",         3.2),
    ("majestic",        "jayanagar",        4.8),
    ("majestic",        "hebbal",           8.5),
    ("mg_road",         "indiranagar",      4.0),
    ("mg_road",         "koramangala",      7.0),
    ("indiranagar",     "marathahalli",     8.5),
    ("indiranagar",     "whitefield",       16.0),
    ("koramangala",     "jayanagar",        5.0),
    ("koramangala",     "electronic_city",  15.0),
    ("jayanagar",       "jp_nagar",         3.5),
    ("jp_nagar",        "electronic_city",  12.0),
    ("marathahalli",    "whitefield",       8.0),
    ("hebbal",          "indiranagar",      9.0),
]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute great-circle distance between two lat/lng points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class CityGraph:
    """
    Simulated city graph over Bangalore waypoints.
    Manages agent position, incidents, and time advancement.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self.waypoints = BANGALORE_WAYPOINTS
        self.waypoint_names = list(self.waypoints.keys())
        self.current_location: Tuple[float, float] = (12.9716, 77.5946)
        self.destination: Tuple[float, float] = (12.9716, 77.5946)
        self.origin_name: str = "majestic"
        self.destination_name: str = "koramangala"
        self.step: int = 0

    def reset(self, origin_name: Optional[str] = None, dest_name: Optional[str] = None) -> None:
        """Reset city state. Picks random origin/dest if not specified."""
        self._rng = random.Random(self._seed)  # deterministic reset
        self.step = 0

        names = self.waypoint_names
        if origin_name is None or dest_name is None:
            pair = self._rng.sample(names, 2)
            origin_name = pair[0]
            dest_name = pair[1]

        # Ensure origin != dest
        while dest_name == origin_name:
            dest_name = self._rng.choice(names)

        self.origin_name = origin_name
        self.destination_name = dest_name
        self.current_location = self.waypoints[origin_name]
        self.destination = self.waypoints[dest_name]

    def distance_remaining_km(self) -> float:
        """Haversine distance from current location to destination."""
        return haversine_km(
            self.current_location[0], self.current_location[1],
            self.destination[0], self.destination[1],
        )

    def total_distance_km(self) -> float:
        """Haversine distance from origin to destination."""
        origin = self.waypoints[self.origin_name]
        return haversine_km(origin[0], origin[1], self.destination[0], self.destination[1])

    def move_towards_destination(self, fraction: float = 0.25) -> None:
        """
        Move agent a fraction of remaining distance toward destination.
        fraction=0.25 means 25% of remaining distance per step.
        """
        lat1, lng1 = self.current_location
        lat2, lng2 = self.destination
        new_lat = lat1 + (lat2 - lat1) * fraction
        new_lng = lng1 + (lng2 - lng1) * fraction
        self.current_location = (new_lat, new_lng)
        self.step += 1

    def is_at_destination(self, threshold_km: float = 0.5) -> bool:
        return self.distance_remaining_km() <= threshold_km

    def random_waypoint_near(self, route_id: str) -> Tuple[float, float]:
        """Return a random lat/lng somewhere between origin and destination."""
        lat1, lng1 = self.waypoints[self.origin_name]
        lat2, lng2 = self.destination
        t = self._rng.uniform(0.2, 0.8)
        offset_lat = self._rng.uniform(-0.01, 0.01)
        offset_lng = self._rng.uniform(-0.01, 0.01)
        lat = lat1 + (lat2 - lat1) * t + offset_lat
        lng = lng1 + (lng2 - lng1) * t + offset_lng
        return (round(lat, 4), round(lng, 4))
