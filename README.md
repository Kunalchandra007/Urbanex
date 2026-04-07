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

URBANEX is a Bangalore traffic-routing reinforcement learning environment built for the Meta x Scaler x Hugging Face OpenEnv hackathon. An agent must decide when to take the `fastest`, `safe`, or `eco` route while reacting to visible incidents, hidden route risk, traffic escalation, and changing weather.

The repository is designed to work as:

- a standalone RL environment
- a FastAPI/OpenEnv server
- a Dockerized Hugging Face Space
- a root-level `inference.py` submission

## Project summary

Each episode is a city trip between Bangalore waypoints. At every step the agent sees route choices, active incidents, traffic, weather, and remaining distance, then picks an action such as selecting a route, rerouting, continuing, or stopping.

What makes URBANEX interesting is that it is not just shortest-path planning:

- route quality is multi-objective: time, safety, and fuel all matter
- some hazards are visible, but some risk is latent via `hidden_risk_prob`
- the hard task changes the city state while the episode is running
- the medium task contains a greedy trap that punishes naive fastest-route policies

## Task suite

| Task | Core idea | Conditions | What it tests |
|------|-----------|------------|---------------|
| `easy` | Reach the destination cleanly | No incidents, low traffic, clear weather | Basic route selection and steady progress |
| `medium` | Avoid the greedy trap | Pre-placed incidents, medium traffic, hidden penalty on `fastest` | Risk-aware planning under partial observability |
| `hard` | Adapt as the city changes | Dynamic incidents, two-stage traffic escalation, two-stage weather change | Safe rerouting and decision stability |

### Easy

- `max_steps = 10`
- `time_limit_min = 35`
- reward weights: safety `0.3`, time `0.5`, fuel `0.2`
- goal: reach the destination in a clean city

### Medium

- `max_steps = 15`
- `time_limit_min = 45`
- pre-placed incidents bias the map against `fastest`
- greedy trap fires at step `2` if the agent commits to `fastest`
- reward weights: safety `0.5`, time `0.3`, fuel `0.2`

### Hard

- `max_steps = 18`
- `time_limit_min = 55`
- incident spawns at steps `2, 4, 6, 9, 12, 15`
- traffic escalates at steps `5` and `10`
- weather changes at steps `7` and `13`
- reward weights: safety `0.5`, time `0.3`, fuel `0.2`

## Observation space

Each observation includes:

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

## Action space

The environment supports five actions:

- `select_route`
- `reroute`
- `continue`
- `report_incident`
- `stop`

## Reward design

The step reward combines safety, time, fuel, and penalty terms.

Positive signals:

- reaching the destination
- choosing the safe route when incidents are present
- making forward progress
- selecting `eco` when fuel efficiency is valuable
- rerouting to a clearly safer route

Negative signals:

- selecting or staying on risky routes
- choosing a route with severe incidents
- unnecessary rerouting
- indecisive route switching
- delayed hidden-risk penalties
- stopping before the destination

Per-step rewards are clamped to `[-1.0, 1.0]`. Final grader scores are kept strictly inside `(0, 1)` because the hackathon validator rejects exact `0.0` and `1.0`.

## Grading

There is one grader per task:

- `graders/grader_easy.py`
- `graders/grader_medium.py`
- `graders/grader_hard.py`

At a high level:

- `easy` rewards reaching the destination quickly with minimal wasted actions
- `medium` rewards avoiding high-severity exposure and resisting the greedy trap
- `hard` rewards safe arrival, adaptation after city changes, and stable decisions

## Reference baseline scores

The repo includes a bundled `rule_based` baseline agent. Under the current task and grader setup, the expected task-level reference scores are:

| Task | Reference baseline score |
|------|--------------------------|
| `easy` | `0.9999` |
| `medium` | `0.9999` |
| `hard` | `0.7000` |

These are the main baseline scores judges should expect from the built-in heuristic policy.

## Repository structure

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
│   └── urbanex_env.py
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

- `environment/urbanex_env.py`
  Core simulation loop and task progression.
- `environment/routes.py`
  Route scoring for time, fuel, safety, and hidden risk.
- `environment/incidents.py`
  Incident lifecycle and route exposure logic.
- `environment/rewards.py`
  Step reward shaping.
- `api/server.py`
  HTTP and WebSocket serving layer.
- `server/app.py`
  OpenEnv entry point used by `openenv validate`.
- `inference.py`
  Root-level hackathon submission script.

## API surface

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Apply one action |
| `GET` | `/state` | Inspect server-side state |
| `GET` | `/tasks` | List task definitions and action schema |
| `POST` | `/grader` | Score a trajectory |
| `POST` | `/baseline` | Run the bundled heuristic baseline |
| `GET` | `/health` | Liveness check |
| `GET` | `/metadata` | OpenEnv metadata |
| `GET` | `/schema` | Action, observation, and state schemas |
| `POST` | `/mcp` | JSON-RPC-compatible stub for validation |

WebSocket support is available at `/ws` using the OpenEnv-style `{"type": "...", "data": {...}}` message shape.

## Inference script

The hackathon requires a root-level `inference.py`, and this repository provides one that:

- calls the environment through `SPACE_URL`
- uses the OpenAI Python client for all LLM requests
- falls back to deterministic routing heuristics if the model call fails
- prints validator-friendly `[START]`, `[STEP]`, and `[END]` trace blocks

### Environment variables used by `inference.py`

| Variable | Purpose |
|----------|---------|
| `SPACE_URL` | Base URL of the running URBANEX server |
| `API_BASE_URL` | OpenAI-compatible model endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face token / API key |

Default OpenAI-compatible endpoint:

- `API_BASE_URL=https://api-inference.huggingface.co/v1`
- `MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3`

## Local development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
python run.py
```

Alternative:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

Run tests:

```bash
pytest -q tests
```

Run OpenEnv validation:

```bash
openenv validate
```

Run inference against a local server:

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

Container entrypoint:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

## Submission compatibility

This repo is laid out to satisfy the hackathon submission contract:

- `Dockerfile` at repo root
- `inference.py` at repo root
- `openenv.yaml` present
- `server/app.py` present
- `uv.lock` present
- `pyproject.toml` exposes the `server` entrypoint

Compatibility shims are also included for:

- `client.UrbanexEnv`
- `models.Action`
- `models.Observation`

## Status

URBANEX is currently set up as:

- a local FastAPI environment
- a Docker-based Hugging Face Space
- an OpenEnv-compatible submission
- a judge-readable project repository
