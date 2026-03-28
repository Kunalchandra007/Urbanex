"""
FastAPI server for Velora OpenEnv.
Exposes all required endpoints: /reset, /step, /state, /tasks, /grader, /baseline.
"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment.velora_env import VeloraEnv
from graders import grade_easy, grade_medium, grade_hard
from models.action import Action
from models.observation import Observation
from models.reward import Reward
from tasks import EASY_CONFIG, MEDIUM_CONFIG, HARD_CONFIG

# ---------------------------------------------------------------------------
# Global environment singleton (one per server process)
# ---------------------------------------------------------------------------

_env: Optional[VeloraEnv] = None


def _get_env() -> VeloraEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call POST /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GraderRequest(BaseModel):
    task: str
    trajectory: List[Dict[str, Any]]


class GraderResponse(BaseModel):
    score: float
    breakdown: Dict[str, Any]


class BaselineRequest(BaseModel):
    task: str = "easy"
    agent: str = "rule_based"
    seed: int = 42


class BaselineResponse(BaseModel):
    task: str
    agent: str
    score: float
    steps: int
    total_reward: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="URBANEX",
    description=(
        "URBANEX (Urban + Exploration) — city navigation RL environment for Bangalore. "
        "Urban navigation meets the RL explore-vs-exploit problem."
    ),
    version="1.0.0",
)

TASK_CONFIGS = {
    "easy":   EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard":   HARD_CONFIG,
}

GRADER_MAP = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset_env(request: ResetRequest):
    """Reset the environment and return the initial observation."""
    global _env
    if request.task not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'. Choose: easy, medium, hard")
    _env = VeloraEnv(task=request.task, seed=request.seed)
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    """Take one step in the environment."""
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state():
    """Return full internal state for debugging/logging."""
    env = _get_env()
    return env.state()


@app.get("/tasks")
def list_tasks():
    """Return all task definitions and the action schema."""
    return {
        "tasks": list(TASK_CONFIGS.values()),
        "action_schema": {
            "action_type": ["select_route", "reroute", "report_incident", "continue", "stop"],
            "route_id": "Optional[str] — for select_route / reroute",
            "incident_type": "Optional[str] — for report_incident",
            "incident_lat": "Optional[float]",
            "incident_lng": "Optional[float]",
        },
    }


@app.post("/grader", response_model=GraderResponse)
def grade_trajectory(request: GraderRequest):
    """Grade a completed episode trajectory."""
    if request.task not in GRADER_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'")

    # Reconstruct trajectory dicts with Pydantic objects where possible
    trajectory = _deserialize_trajectory(request.trajectory)

    grader_fn = GRADER_MAP[request.task]
    score = grader_fn(trajectory)

    return GraderResponse(
        score=score,
        breakdown={
            "task": request.task,
            "steps": len(trajectory),
            "score": score,
        },
    )


@app.post("/baseline", response_model=BaselineResponse)
def run_baseline(request: BaselineRequest):
    """Run the baseline agent and return its score."""
    from baseline.baseline_agent import run_baseline as _run_baseline

    if request.task not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{request.task}'")

    result = _run_baseline(task=request.task, agent=request.agent, seed=request.seed)
    return BaselineResponse(**result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deserialize_trajectory(raw: List[Dict[str, Any]]) -> List[dict]:
    """Best-effort deserialization of trajectory for grading."""
    from models.observation import Observation
    from models.action import Action
    from models.reward import Reward

    trajectory = []
    for step in raw:
        try:
            obs = Observation.model_validate(step["obs"]) if isinstance(step.get("obs"), dict) else step.get("obs")
            action = Action.model_validate(step["action"]) if isinstance(step.get("action"), dict) else step.get("action")
            reward = Reward.model_validate(step["reward"]) if isinstance(step.get("reward"), dict) else step.get("reward")
            trajectory.append({"obs": obs, "action": action, "reward": reward})
        except Exception:
            trajectory.append(step)
    return trajectory
