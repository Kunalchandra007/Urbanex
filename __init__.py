"""
URBANEX (Urban + Exploration) — City Navigation RL Environment for OpenEnv.

An agentic RL environment where agents act as smart routing assistants,
selecting routes while avoiding incidents, responding to dynamic conditions,
and optimizing for safety, time, and fuel efficiency.
"""

__version__ = "1.0.0"
__author__ = "URBANEX Team"

from models import Action, Observation, RouteOption, Incident, Reward

__all__ = ["Action", "Observation", "RouteOption", "Incident", "Reward"]
