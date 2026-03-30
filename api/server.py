"""
FastAPI server for Velora OpenEnv.
Exposes all required endpoints: /reset, /step, /state, /tasks, /grader, /baseline, /ws (WebSocket).
"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import json

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
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


class ResetResponse(BaseModel):
    """Standard OpenEnv reset response format."""
    observation: Observation
    info: Dict[str, Any] = {}


class StepResponse(BaseModel):
    """Standard OpenEnv step response format."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


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

@app.get("/")
def root():
    """Root endpoint — required for Hugging Face Spaces health check."""
    return {
        "name": "URBANEX",
        "version": "1.0.0",
        "status": "ok",
        "description": "Urban navigation RL environment",
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for OpenEnv client connections.
    Handles reset, step, and state messages as per OpenEnv spec.
    """
    await websocket.accept()
    env: Optional[VeloraEnv] = None
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "reset":
                # Reset the environment
                task = data.get("task", "easy")
                seed = data.get("seed", 42)
                env = VeloraEnv(task=task, seed=seed)
                obs = env.reset()
                await websocket.send_json({
                    "type": "reset",
                    "observation": obs.dict(),
                    "episode_id": str(seed)
                })
            
            elif msg_type == "step":
                # Take a step in the environment
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first."
                    })
                    continue
                
                try:
                    action_data = data.get("action", {})
                    action = Action(**action_data)
                    obs, reward, done, info = env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "observation": obs.dict(),
                        "reward": float(reward.total),
                        "done": done,
                        "info": info
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif msg_type == "state":
                # Get current state
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first."
                    })
                    continue
                
                obs = env.get_observation()
                await websocket.send_json({
                    "type": "state",
                    "observation": obs.dict()
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/health")
def health():
    """Standard OpenEnv health check endpoint."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset_env(
    request: Optional[ResetRequest] = Body(default=None),
    task_id: Optional[str] = Query(default=None),
    task: Optional[str] = Query(default=None),
    seed: Optional[int] = Query(default=None),
):
    """
    Reset the environment and return the initial observation.
    Accepts both JSON body and query parameters for flexibility.
    Supports both 'task' and 'task_id' parameter names.
    
    Returns standard OpenEnv format:
    {
        "observation": {...},
        "info": {}
    }
    """
    global _env
    
    # Priority: JSON body > query parameters
    if request is not None:
        task_name = request.task
        seed_val = request.seed
    else:
        # Use query parameters (support both task_id and task)
        task_name = task_id or task or "easy"
        seed_val = seed if seed is not None else 42
    
    if task_name not in TASK_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'. Choose: easy, medium, hard")
    
    _env = VeloraEnv(task=task_name, seed=seed_val)
    obs = _env.reset()
    return ResetResponse(observation=obs, info={})


@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    """
    Take one step in the environment.
    
    Returns standard OpenEnv format:
    {
        "observation": {...},
        "reward": float,
        "done": bool,
        "info": {}
    }
    """
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        observation=obs,
        reward=float(reward.total),
        done=done,
        info=info,
    )


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
