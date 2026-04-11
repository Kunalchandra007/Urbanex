import json
import os
import re

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_URL = os.getenv("SPACE_URL", "https://kunalchandra007-urbanexx.hf.space")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def llm_action(obs: dict) -> dict:
    try:
        routes = obs.get("available_routes", [])
        current = obs.get("current_route")
        incidents = obs.get("active_incidents", [])
        route_details = "; ".join(
            f"{route['route_id']}: {route.get('estimated_time_min', 0):.1f} min, "
            f"{route.get('incident_count', 0)} incidents, safety={route.get('safety_score', 0):.2f}, "
            f"hidden_risk={route.get('hidden_risk_prob', 0):.2f}"
            for route in routes
        ) or "none"
        current_route_incidents = [
            f"{incident.get('severity', 'unknown')} {incident.get('type', 'incident')}"
            for incident in incidents
            if current and current in incident.get("affects_routes", [])
        ]
        current_incident_text = ", ".join(current_route_incidents) if current_route_incidents else "none"
        prompt = f"""You are an AI navigation agent in Bangalore. Your goal is to reach the destination safely and efficiently.
Current situation: {obs.get('situation_summary', 'No summary available.')}
Full observation: routes available are {route_details}.
Incidents on your current route: {current_incident_text}.

Choose the single best action. Reply with only valid JSON on one line.
Valid action_types: select_route, continue, reroute
Valid route_ids: fastest, safe, eco
Example: {{"action_type":"continue"}}"""
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        text = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return _fallback(obs)


def _fallback(obs: dict) -> dict:
    routes = obs.get("available_routes", [])
    current = obs.get("current_route")
    if current is None:
        clean = [r for r in routes if r.get("incident_count", 0) == 0]
        pool = clean if clean else routes
        if not pool:
            return {"action_type": "continue"}
        best = min(pool, key=lambda r: r.get("estimated_time_min", 999))
        return {"action_type": "select_route", "route_id": best["route_id"]}
    return {"action_type": "continue"}


def run_episode(task: str) -> None:
    env = httpx.Client(base_url=SPACE_URL, timeout=60)
    rewards: list[float] = []
    step = 0
    done = False
    success = False

    print(f"[START] task={task} env=urbanex model={MODEL_NAME}", flush=True)

    try:
        reset_result = env.post("/reset", json={"task": task, "seed": 42}).json()
        obs = reset_result.get("observation", reset_result)
    except Exception:
        print("[END] success=false steps=0 rewards=", flush=True)
        env.close()
        return

    while not done and step < 20:
        action = llm_action(obs)
        action_str = action.get("action_type", "continue")
        if "route_id" in action and action["route_id"]:
            action_str += f"({action['route_id']})"

        try:
            result = env.post("/step", json=action).json()
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", obs)
            error = "null"
            if done and obs.get("episode_done"):
                success = True
        except Exception as exc:
            reward = 0.0
            done = True
            error = str(exc)

        rewards.append(reward)
        step += 1
        print(
            f"[STEP] step={step} action={action_str} reward={reward:.2f} "
            f"done={str(done).lower()} error={error}",
            flush=True,
        )

    env.close()
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task_name in ["easy", "medium", "hard"]:
        run_episode(task_name)
