"""
Route generation and scoring logic for URBANEX.

Builds the three route profiles and assigns time, safety, fuel, and hidden risk.
"""
import random
from typing import List, Optional

from models.observation import RouteOption

ROUTE_PROFILES = {
    "fastest": {
        "base_time_min": 20,
        "fuel_multiplier": 1.3,       # uses more fuel
        "incident_exposure": 1.0,     # full exposure to incidents
        "base_hidden_risk": 0.35,     # HIGH hidden risk — greedy trap
        "description": "Shortest time, highest incident exposure",
    },
    "safe": {
        "base_time_min": 28,
        "fuel_multiplier": 1.0,
        "incident_exposure": 0.2,     # avoids most incident zones
        "base_hidden_risk": 0.05,     # very low hidden risk
        "description": "Longer but avoids high-risk areas",
    },
    "eco": {
        "base_time_min": 25,
        "fuel_multiplier": 0.7,       # best fuel efficiency
        "incident_exposure": 0.6,
        "base_hidden_risk": 0.18,     # moderate hidden risk
        "description": "Balanced — moderate time, lower fuel",
    },
}

# Maximum possible fuel cost so we can normalise to 0–1
_MAX_FUEL = max(p["fuel_multiplier"] for p in ROUTE_PROFILES.values())

# Traffic and weather multipliers
TRAFFIC_MULTIPLIERS = {"low": 1.0, "medium": 1.15, "high": 1.30}
WEATHER_MULTIPLIERS = {"clear": 1.0, "rain": 1.10, "fog": 1.15, "heavy_rain": 1.25}

# Weather amplifies hidden risk
WEATHER_RISK_MULTIPLIERS = {"clear": 1.0, "rain": 1.35, "fog": 1.50, "heavy_rain": 1.80}
TRAFFIC_RISK_MULTIPLIERS = {"low": 1.0, "medium": 1.20, "high": 1.55}


class RouteCalculator:
    """
    Computes RouteOption objects for each of the three route profiles.
    Includes hidden_risk_prob reflecting latent unobservable danger.
    """

    def calculate_routes(
        self,
        incident_manager,
        traffic_level: str = "low",
        weather: str = "clear",
        step: int = 0,
        rng: Optional[random.Random] = None,
    ) -> List[RouteOption]:
        """
        Returns a RouteOption for each profile with time/safety/fuel/hidden_risk
        adjusted for current traffic, weather, and active incidents.
        """
        if rng is None:
            rng = random.Random(step * 137 + 42)

        routes: List[RouteOption] = []
        traffic_mult = TRAFFIC_MULTIPLIERS.get(traffic_level, 1.0)
        weather_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
        risk_traffic = TRAFFIC_RISK_MULTIPLIERS.get(traffic_level, 1.0)
        risk_weather = WEATHER_RISK_MULTIPLIERS.get(weather, 1.0)

        for route_id, profile in ROUTE_PROFILES.items():
            base_time = profile["base_time_min"]
            exposure = profile["incident_exposure"]

            # Time adjustments
            time_penalty = incident_manager.total_time_penalty_on_route(route_id) * exposure
            adjusted_time = (base_time + time_penalty) * traffic_mult * weather_mult

            # Safety score: start at 1.0, subtract safety impact scaled by exposure
            safety_impact = incident_manager.total_safety_impact_on_route(route_id) * exposure
            safety_score = max(0.0, 1.0 - safety_impact)
            if weather == "heavy_rain":
                safety_score = max(0.0, safety_score - 0.3)

            # Incident count (full count for agent awareness)
            incident_count = len(incident_manager.get_incidents_on_route(route_id))

            # Fuel cost score normalised to 0–1
            fuel_cost_score = round(profile["fuel_multiplier"] / _MAX_FUEL, 3)

            # ── PARTIAL OBSERVABILITY: hidden risk ──────────────────────────
            # Base hidden risk from route profile, amplified by traffic/weather
            # and perturbed with small per-step noise so it varies naturally.
            base_hidden = profile["base_hidden_risk"]
            noise = rng.uniform(-0.05, 0.07)
            hidden_risk = base_hidden * risk_traffic * risk_weather + noise
            # Incidents on this route add to hidden risk (other incidents may lurk)
            hidden_risk += incident_count * 0.04
            hidden_risk = round(max(0.0, min(1.0, hidden_risk)), 3)
            # ────────────────────────────────────────────────────────────────

            routes.append(RouteOption(
                route_id=route_id,
                estimated_time_min=round(adjusted_time, 2),
                incident_count=incident_count,
                fuel_cost_score=fuel_cost_score,
                safety_score=round(safety_score, 3),
                hidden_risk_prob=hidden_risk,
            ))

        return routes
