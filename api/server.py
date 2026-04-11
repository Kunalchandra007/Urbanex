"""
FastAPI server for URBANEX.

Exposes the HTTP and WebSocket interfaces used by local development,
OpenEnv validation, and the Hugging Face Space deployment.
"""
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from environment.urbanex_env import UrbanexEnv
from graders import grade_easy, grade_medium, grade_hard
from models.action import Action
from models.observation import Observation
from models.reward import Reward
from tasks import EASY_CONFIG, MEDIUM_CONFIG, HARD_CONFIG

# ---------------------------------------------------------------------------
# Global environment singleton (one per server process)
# ---------------------------------------------------------------------------

_env: Optional[UrbanexEnv] = None


def _get_env() -> UrbanexEnv:
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


class StateSchema(BaseModel):
    """Schema used by /schema without requiring openenv-core at runtime."""

    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)


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


def _clamp_open_score(score: float) -> float:
    """Clamp scores to the validator-required open interval."""
    return round(max(0.05, min(0.95, float(score))), 4)


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


@app.get("/metadata")
def metadata():
    """Minimal OpenEnv metadata endpoint."""
    return {
        "name": "URBANEX",
        "description": "Urban navigation RL environment for route selection under uncertainty",
        "version": "1.0.0",
    }


@app.get("/schema")
def schema():
    """Expose action / observation / state schemas for OpenEnv validators."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": StateSchema.model_json_schema(),
    }


@app.post("/mcp")
def mcp():
    """Return a JSON-RPC payload so OpenEnv runtime validation succeeds."""
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32601,
            "message": "MCP is not implemented for this simulation environment.",
        },
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for OpenEnv client connections.
    Handles reset, step, and state messages as per OpenEnv spec.
    """
    await websocket.accept()
    env: Optional[UrbanexEnv] = None
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            is_legacy_message = "data" not in data
            payload = data if is_legacy_message else data.get("data", {})
            
            if msg_type == "reset":
                # Reset the environment
                task = payload.get("task", payload.get("task_id", "easy"))
                seed = payload.get("seed", 42)
                env = UrbanexEnv(task=task, seed=seed)
                obs = env.reset()
                obs_payload = obs.model_dump()
                obs_payload["done"] = obs.episode_done
                obs_payload["reward"] = None
                response = {
                    "type": "reset" if is_legacy_message else "observation",
                    "data": obs_payload,
                    "observation": obs_payload,
                    "episode_id": str(seed),
                }
                await websocket.send_json(response)
            
            elif msg_type == "step":
                # Take a step in the environment
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first.",
                        "data": {
                            "message": "Environment not initialized. Call reset first.",
                            "code": "EXECUTION_ERROR",
                        },
                    })
                    continue
                
                try:
                    action_data = payload if not is_legacy_message else data.get("action", {})
                    action = Action(**action_data)
                    obs, reward, done, info = env.step(action)
                    obs_payload = obs.model_dump()
                    obs_payload["done"] = done
                    obs_payload["reward"] = float(reward.total)
                    obs_payload["metadata"] = {"info": info}
                    response = {
                        "type": "step" if is_legacy_message else "observation",
                        "data": obs_payload,
                        "observation": obs_payload,
                        "reward": float(reward.total),
                        "done": done,
                        "info": info,
                    }
                    await websocket.send_json(response)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                        "data": {
                            "message": str(e),
                            "code": "EXECUTION_ERROR",
                        },
                    })
            
            elif msg_type == "state":
                # Get current state
                if env is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Environment not initialized. Call reset first.",
                        "data": {
                            "message": "Environment not initialized. Call reset first.",
                            "code": "EXECUTION_ERROR",
                        },
                    })
                    continue
                
                state_payload = env.state()
                obs = env.get_observation()
                await websocket.send_json({
                    "type": "state",
                    "data": state_payload,
                    "observation": obs.model_dump(),
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "data": {
                        "message": f"Unknown message type: {msg_type}",
                        "code": "UNKNOWN_TYPE",
                    },
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/health")
def health():
    """Standard OpenEnv health check endpoint."""
    return {"status": "healthy"}


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
    
    _env = UrbanexEnv(task=task_name, seed=seed_val)
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
    score = _clamp_open_score(grader_fn(trajectory))

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
    result["score"] = _clamp_open_score(result["score"])
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
