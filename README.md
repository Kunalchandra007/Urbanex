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

# URBANEX — Urban Exploration RL Environment

> **URBANEX** = **Urban** + **Exploration** — a double meaning by design.  
> *Exploring a city's road network* AND *the core RL explore-vs-exploit dilemma* that every agent must solve navigating it.  
> Sounds like a product. Because it is one.

> *Bangalore has over 12 million residents, 8 million registered vehicles, and a pothole every 50 meters. Getting from point A to B sounds simple. For an AI agent, it's a minefield.*

## What this environment models

Velora simulates urban navigation decision-making over a graph of Bangalore's real road network. An AI agent receives structured observations about road conditions — potholes, accidents, flooding, construction — and must choose from three route types to reach a destination both safely and efficiently.

The twist: **not everything is visible**. Every route carries a `hidden_risk_prob` — the probability that an undetected incident is lurking. A greedy agent that always picks the fastest route will get burned by a hidden accident two steps in. A cautious agent that reroutes at every shadow will be penalized for indecision. The optimal agent reasons under uncertainty.

## Why this domain matters for AI agents

Modern LLMs are being evaluated on real-world agentic tasks. Urban routing is:
- **Multi-objective**: minimize time AND maximize safety AND save fuel — simultaneously
- **Partially observable**: not all hazards are reported; sensors lie; reports are delayed
- **Non-stationary**: traffic escalates mid-journey; rain starts at step 7; new incidents spawn
- **Consequential**: a single wrong route choice causes a penalty 2 steps later

This makes Velora a natural test of chain-of-thought reasoning, risk calibration, and adaptive planning.

## Action space

| action_type | Parameters | Description |
|-------------|------------|-------------|
| `select_route` | `route_id` | Choose a route to follow at episode start |
| `reroute` | `route_id` | Switch route mid-journey (triggers recovery bonus if safer) |
| `report_incident` | `incident_type`, `incident_lat`, `incident_lng` | Report an incident (rewarded if real, penalized if false) |
| `continue` | — | Stay on current route (penalized if hidden risk is high) |
| `stop` | — | Abandon journey (−1.0 penalty) |

## Observation space

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current episode step |
| `current_location` | [float, float] | Agent lat/lng |
| `destination` | [float, float] | Target lat/lng |
| `available_routes` | List[RouteOption] | fastest / safe / eco with time, safety, fuel, **hidden_risk_prob** |
| `active_incidents` | List[Incident] | Visible incidents (pothole, accident, flooding, construction) |
| `traffic_level` | str | "low" / "medium" / "high" (changes dynamically in hard task) |
| `weather` | str | "clear" / "rain" / "fog" (changes mid-episode in hard task) |
| `current_route` | Optional[str] | Currently selected route |
| `distance_remaining_km` | float | Haversine distance to destination |
| `episode_done` | bool | Terminal state flag |

**Key partial observability field**: `hidden_risk_prob` on each `RouteOption`. The agent sees the probability of a lurking incident but never knows exactly what it is or when it fires. The consequence arrives 2 steps later as a reward penalty.

## Reward function

| Event | Reward | Reason |
|-------|--------|--------|
| Reached destination | +1.0 × weights | Primary goal |
| Safe route chosen with incidents present | +0.3 × safety_w | Proactive risk avoidance |
| Recovery reroute (moved to lower-risk route) | +0.25 × safety_w | Adaptive replanning bonus |
| Progress toward destination | +0.1 × time_w | Dense step signal |
| Eco route chosen | +0.15 × fuel_w | Fuel efficiency |
| High hidden risk on chosen route | −0.4 × hidden_risk × safety_w | Risk exposure |
| High-severity incident on chosen route | −0.5 to −0.8 | Safety penalty |
| Continuing on risky route | −0.15 × safety_w × risk | Sustained exposure |
| Indecision (route switch every step) | −0.25 × safety_w | Decision instability |
| Unnecessary reroute | −0.3 | Efficiency penalty |
| Delayed hidden incident fires | −`penalty` (arrives 2 steps later) | Consequence of bad choice |
| Stop before destination | −1.0 | Episode failure |

Per-step total is clamped to [−1.0, 1.0]. Episode total is not clamped.

## Tasks

### Easy — reach_destination
- **Goal**: Reach destination within 35 minutes
- **City state**: No incidents, low traffic, clear weather, low hidden risk (scale 0.10)
- **Max steps**: 10
- **Designed for**: Agents that can follow a basic route
- **Reward weights**: safety 0.3, time 0.5, fuel 0.2

### Medium — avoid_and_optimize
- **Goal**: Reach destination while avoiding 3 pre-placed incidents on the fastest route
- **Greedy trap**: Fastest route appears clear at step 0 but a hidden accident fires at step 2 — agents that don't reason about `hidden_risk_prob` are punished
- **City state**: Medium traffic, clear weather, hidden risk scale 0.45
- **Max steps**: 15
- **Reward weights**: safety 0.5, time 0.3, fuel 0.2

### Hard — dynamic_rerouting
- **Goal**: Reach destination as the city changes around you
- **Dynamic events**: 6 incident spawns (steps 2, 4, 6, 9, 12, 15) — 70% biased on fastest
- **Two-stage traffic**: low → medium at step 5 → high at step 10
- **Two-stage weather**: clear → rain at step 7 → fog at step 13
- **Pre-placed**: 2 incidents (flooding on eco, construction on fastest) from the start
- **Max steps**: 18 (tight)
- **Success condition**: Reach destination **and** maintain avg safety score > 0.6
- **Reward weights**: safety 0.5, time 0.3, fuel 0.2

### Example episode walkthrough (Hard task)
```
Step 01 | SELECT [S] safe       | [#####.....] +0.149 | 2 inc | 5.17km | SUN/TLO
Step 02 | CONT   [S] safe       | [#####.....] +0.030 | 3 inc | 3.62km | SUN/TLO  ← incident spawns
Step 04 | CONT   [S] safe       | [#####.....] +0.030 | 4 inc | 1.77km | SUN/TLO
Step 05 | CONT   [S] safe       | [#####.....] +0.030 | 4 inc | 1.24km | SUN/TMD  ← traffic: medium
Step 07 | CONT   [S] safe       | [#####.....] +0.030 | 4 inc | 0.61km | RAN/TMD  ← weather: rain
Step 08 | CONT   [S] safe       | [##########] +1.000 | 3 inc | 0.43km | RAN/TMD  ← destination!
>> REACHED DESTINATION  |  Grader Score: 0.7000
```
A frontier LLM agent that greedily picks `fastest` will trigger the construction incident penalty and get hit by 4+ dynamic spawns without rerouting — expected score: ~0.3.

## Setup

### Local
```bash
cd C:\Projects\URBANEX
pip install -r requirements.txt

# Start server
python run.py
# OR: uvicorn api.server:app --host 0.0.0.0 --port 7860 --reload
```

Run tests:
```bash
python -m pytest tests/ -v
```

Run baseline with visual episode log:
```bash
python baseline/baseline_agent.py hard rule_based
# Set VELORA_RENDER=1 to print step-by-step table
```

### Docker
```bash
docker build -t urbanex .
docker run -p 7860:7860 urbanex
```

Test endpoints after build:
```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy","seed":42}'
curl http://localhost:7860/tasks
curl http://localhost:7860/state
curl -X POST http://localhost:7860/baseline -H "Content-Type: application/json" -d '{"task":"easy"}'
```

### Hugging Face Spaces
```bash
cd C:\Projects\URBANEX
git init
git add .
git commit -m "initial: URBANEX OpenEnv environment"
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/urbanex
git push hf main
```
Once the Space shows **Running** (green badge), test:
```bash
curl -X POST https://YOUR_HF_USERNAME-urbanex.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"easy","seed":42}'
```

## Baseline scores

| Task | Agent | Score | Notes |
|------|-------|-------|-------|
| easy | rule_based | 1.00 | Clean city, straightforward |
| medium | rule_based | 1.00 | Smart agent avoids greedy trap |
| hard | rule_based | 0.70 | Partial adaptation to dynamics |

*A naive LLM agent (greedy-fastest picker) is expected to score: easy ~0.90, medium ~0.60, hard ~0.30.*

*(All scores deterministic with seed=42. Run `baseline/baseline_agent.py` to reproduce.)*

## API reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `POST` | `/reset` | `{task, seed}` | `Observation` |
| `POST` | `/step` | `Action` | `{observation, reward, done, info}` |
| `GET` | `/state` | — | Full internal state dict |
| `GET` | `/tasks` | — | Task definitions + action schema |
| `POST` | `/grader` | `{task, trajectory}` | `{score, breakdown}` |
| `POST` | `/baseline` | `{task, agent, seed}` | `{task, score, steps, total_reward}` |
