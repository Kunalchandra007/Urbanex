---
title: URBANEX
emoji: 🗺️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
license: mit
tags:
  - openenv
  - reinforcement-learning
  - urban-navigation
  - simulation
---

# URBANEX

URBANEX is an urban navigation reinforcement learning environment where an agent acts like a smart routing assistant for Bangalore. It must choose between `fastest`, `safe`, and `eco` routes while reacting to traffic incidents, hidden risks, weather shifts, and changing traffic conditions.

The project was built for the Meta + Hugging Face + PyTorch OpenEnv hackathon and is designed to work both as:

- a local FastAPI environment for testing and development
- a Dockerized Hugging Face Space
- an OpenEnv-compatible environment with a standalone `inference.py`

## Why this project is interesting

URBANEX is not just shortest-path routing. The agent has to reason under uncertainty.

- Some hazards are visible: potholes, accidents, flooding, and construction.
- Some hazards are hidden: each route exposes a `hidden_risk_prob` but not the exact future incident.
- Rewards are multi-objective: safety, time, and fuel efficiency all matter.
- Hard mode is dynamic: incidents spawn during the episode and traffic/weather conditions change mid-run.

That makes the environment a good benchmark for planning, risk tradeoffs, and adaptive decision-making.

## Core idea

Each episode follows the usual RL loop:

1. The environment returns an observation.
2. The agent chooses an action.
3. The environment advances one step.
4. It returns a new observation, a scalar reward, and a `done` flag.

In URBANEX:

- Observation includes current location, destination, available routes, active incidents, traffic, weather, current route, and remaining distance.
- Actions include `select_route`, `reroute`, `continue`, `report_incident`, and `stop`.
- Rewards favor safe and efficient progress while penalizing risky or unstable decisions.

## Tasks

| Task | Goal | What makes it hard |
|------|------|--------------------|
| `easy` | Reach destination in a clean city | Straightforward routing |
| `medium` | Reach destination while avoiding incidents | Hidden-risk greedy trap |
| `hard` | Reach destination while the city changes around you | Dynamic incidents, traffic escalation, weather changes |

## Project architecture

```text
URBANEX/
├── api/
│   └── server.py
├── baseline/
│   ├── baseline_agent.py
│   └── visualizer.py
├── environment/
│   ├── city.py
│   ├── incidents.py
│   ├── rewards.py
│   ├── routes.py
│   └── velora_env.py
├── graders/
│   ├── grader_easy.py
│   ├── grader_medium.py
│   └── grader_hard.py
├── models/
│   ├── __init__.py
│   ├── action.py
│   ├── observation.py
│   └── reward.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── urbanex_environment.py
├── tasks/
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
├── tests/
│   ├── test_env.py
│   └── test_graders.py
├── client.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── run.py
└── uv.lock
```

### Key files

- `environment/velora_env.py`
  Main environment loop. Handles reset, step progression, delayed hidden penalties, and done conditions.
- `environment/routes.py`
  Generates the route options an agent can choose from.
- `environment/incidents.py`
  Creates and manages visible incidents.
- `environment/rewards.py`
  Computes the scalar reward and reward breakdown.
- `api/server.py`
  FastAPI app exposing HTTP endpoints and WebSocket support.
- `server/app.py`
  Standard OpenEnv-style entry point used by `openenv validate`.
- `inference.py`
  Standalone agent script expected by the hackathon validator.

## API surface

### HTTP endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/reset` | Start an episode |
| `POST` | `/step` | Apply one action |
| `GET` | `/state` | Inspect environment state |
| `GET` | `/tasks` | List tasks and action schema |
| `POST` | `/grader` | Score a trajectory |
| `POST` | `/baseline` | Run the built-in baseline |
| `GET` | `/health` | Health endpoint |
| `GET` | `/metadata` | OpenEnv metadata |
| `GET` | `/schema` | Action/observation/state schemas |
| `POST` | `/mcp` | JSON-RPC-compatible stub response |

### WebSocket endpoint

`/ws` supports:

- OpenEnv-style messages using `{"type": "...", "data": {...}}`
- older internal/local test message shapes for backward compatibility

## Example observation

An observation includes fields like:

- `step`
- `current_location`
- `destination`
- `available_routes`
- `active_incidents`
- `traffic_level`
- `weather`
- `current_route`
- `distance_remaining_km`
- `episode_done`

Each route option includes:

- `route_id`
- `estimated_time_min`
- `incident_count`
- `fuel_cost_score`
- `safety_score`
- `hidden_risk_prob`

## Reward design

The reward system encourages:

- reaching the destination
- choosing safer routes when incidents are present
- making progress consistently
- fuel efficiency when appropriate

It penalizes:

- staying on risky routes
- high-severity incident exposure
- indecisive rerouting
- stopping early
- delayed consequences from hidden risk

Per-step rewards are clamped to `[-1.0, 1.0]`.

## Inference pipeline

The hackathon expects a root-level `inference.py`, and this project provides one.

It does three important things:

1. Talks to the running environment through `SPACE_URL`
2. Uses the OpenAI Python client for LLM calls
3. Falls back to deterministic heuristics when the remote LLM is unavailable

### Environment variables used by `inference.py`

| Variable | Purpose |
|----------|---------|
| `SPACE_URL` | Base URL for the running environment |
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API token |

### Current default model path

The inference script currently uses Hugging Face's OpenAI-compatible API by default:

- `API_BASE_URL=https://api-inference.huggingface.co/v1`
- `MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3`

This still satisfies the important validator requirement because the code uses the `OpenAI(...)` client interface.

## Local development

### Install

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
python run.py
```

or

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

### Run tests

```bash
pytest -q tests
```

### Run OpenEnv validation

```bash
openenv validate
```

### Run inference locally against the local server

```bash
set SPACE_URL=http://localhost:7860
python inference.py
```

On PowerShell:

```powershell
$env:SPACE_URL="http://localhost:7860"
python inference.py
```

## Docker

Build:

```bash
docker build -t urbanex .
```

Run:

```bash
docker run -p 7860:7860 urbanex
```

The Docker image starts:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

## Hugging Face Space

Repository target:

- Space URL: `https://huggingface.co/spaces/kunalchandra007/urbanexx`

Runtime URL:

- `https://kunalchandra007-urbanexx.hf.space`

Suggested Space secrets:

| Secret | Value |
|--------|-------|
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `HF_TOKEN` | your Hugging Face token |

## Validation notes

The repo is set up to satisfy the local OpenEnv structural validation:

- `openenv.yaml` exists
- `server/app.py` exists
- `uv.lock` exists
- `[project.scripts].server` exists in `pyproject.toml`
- root-level `inference.py` exists

The project also includes compatibility paths for:

- `client.UrbanexEnv`
- `models.Action`
- `models.Observation`

## Baseline and grading

The repo includes:

- a local baseline agent in `baseline/baseline_agent.py`
- task-specific graders in `graders/`

This makes it easy to compare:

- environment rollout behavior
- heuristic agent behavior
- LLM-driven behavior

## Why the repo is organized this way

The split between `environment/`, `api/`, `server/`, `models/`, `graders/`, and `tasks/` keeps the project easy to reason about:

- `environment/` contains the simulation logic
- `api/` contains the serving layer
- `server/` contains OpenEnv-oriented wrappers and entry points
- `models/` contains schemas
- `tasks/` contains difficulty configuration
- `graders/` contains scoring

That makes the codebase easier to test, deploy, and extend.

## Status

URBANEX is currently set up as:

- a local FastAPI environment
- a Dockerized deployment target
- an OpenEnv-compatible submission layout
- a root-level LLM inference script using the OpenAI client

If you want a deeper internal walkthrough, see `urbanex_deep_dive.md`.
