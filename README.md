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

URBANEX is a reinforcement learning benchmark for urban navigation in Bangalore, where agents must route under real-world-style stakes: unsafe roads, delayed hazards, weather disruption, and conflicting objectives across safety, travel time, and fuel efficiency. Unlike toy gridworlds, it blends partial observability with decision pressure, making it a compact but meaningful testbed for both classical RL policies and general-purpose LLM agents.

## Observation Space

| Field | Type | Description |
|------|------|-------------|
| `step` | `int` | Current timestep within the episode |
| `current_location` | `list[float]` | Agent position as `[lat, lng]` |
| `destination` | `list[float]` | Goal position as `[lat, lng]` |
| `available_routes` | `list[RouteOption]` | Candidate route plans: `fastest`, `safe`, `eco` |
| `active_incidents` | `list[Incident]` | Visible incidents currently affecting the city |
| `traffic_level` | `str` | Traffic state: `low`, `medium`, or `high` |
| `weather` | `str` | Weather state: `clear`, `rain`, `fog`, or `heavy_rain` |
| `current_route` | `str | null` | Route currently selected by the agent |
| `distance_remaining_km` | `float` | Remaining straight-line distance to destination |
| `episode_done` | `bool` | Whether the episode has terminated |
| `situation_summary` | `str` | Plain-English summary of the current situation for LLM-friendly reasoning |

### RouteOption

| Field | Type | Description |
|------|------|-------------|
| `route_id` | `str` | One of `fastest`, `safe`, `eco` |
| `estimated_time_min` | `float` | Estimated travel time for that route |
| `incident_count` | `int` | Number of visible incidents affecting the route |
| `fuel_cost_score` | `float` | Relative fuel cost from `0` to `1` |
| `safety_score` | `float` | Relative safety from `0` to `1` |
| `hidden_risk_prob` | `float` | Probability of concealed downstream danger |

## Action Space

| Action | Parameters | When to use |
|------|------|-------------|
| `select_route` | `route_id` | Choose an initial route when none is active |
| `reroute` | `route_id` | Switch away from a risky or degraded route |
| `continue` | none | Stay on the current route and make progress |
| `report_incident` | `incident_type`, `incident_lat`, `incident_lng` | Report a visible incident correctly |
| `stop` | none | Terminate the journey early |

## Reward Function

URBANEX uses a weighted reward with progress bonuses and explicit penalties:

```text
R_t = w_s * safety_t + w_t * time_t + w_f * fuel_t + progress_bonus_t - penalties_t
```

Where:

- `w_s`, `w_t`, `w_f` are task-specific weights
- `progress_bonus_t` includes positive credit for reducing distance
- `penalties_t` include severe-incident exposure, hidden-risk decisions, delayed consequences, unnecessary reroutes, and early stopping

Key shaped signals:

- reward for reaching the destination
- reward for selecting safer routes under active incidents
- `+0.05` when distance decreases
- penalty for selecting a route with `hidden_risk_prob > 0.3` despite no visible justification
- penalty for staying on `eco` after the hard-task fuel-critical trigger

Per-step reward is clamped to `[-1, 1]`. Final grader scores are constrained to the open interval `(0, 1)` for validator compatibility.

## Tasks

| Task | Max Steps | Traffic | Weather | What makes it hard |
|------|-----------|---------|---------|--------------------|
| `easy` | `10` | Low | Clear | Straightforward route selection with no incident pressure |
| `medium` | `15` | Medium | Clear | Greedy trap on the fastest route plus pre-placed incidents |
| `hard` | `18` | Low → Medium → High | Clear → Heavy Rain | Forced high-severity accident on `fastest`, late fuel pressure on `eco`, dynamic incident spawns, and safety collapse under heavy rain |

## Baseline Scores

Rule-based baseline agent, shown in rounded human-readable form:

| Task | Score |
|------|-------|
| `easy` | `1.0` |
| `medium` | `1.0` |
| `hard` | `0.7` |

## Why URBANEX Challenges LLMs

URBANEX is difficult for LLM agents because route safety is only partially observable. The `hidden_risk_prob` field hints that a route may be dangerous without revealing the exact failure mode, so the model must reason under uncertainty rather than simply reacting to visible incidents. The hard task adds delayed consequences: an apparently attractive route can degrade several steps later as incidents spawn or weather worsens. On top of that, the environment is explicitly multi-objective, so the shortest route is often tempting but strategically wrong when safety and fuel tradeoffs are taken seriously.

## Setup

### Local

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python run.py
```

Run tests:

```bash
pytest tests/ -v
```

Run inference against a local server:

```powershell
$env:SPACE_URL="http://localhost:7860"
python inference.py
```

### Docker

Build:

```bash
docker build -t urbanex .
```

Run:

```bash
docker run -p 7860:7860 urbanex
```

### OpenEnv Compatibility

This repository includes:

- root `Dockerfile`
- root `inference.py`
- `openenv.yaml`
- `server/app.py`
- `uv.lock`
- OpenEnv-compatible schemas and client/server wrappers
