import os
import json
import httpx
from openai import OpenAI

BASE_URL = os.getenv("SPACE_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN support
openai_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "sk-dummy")
)


def llm_decide_action(obs: dict) -> dict:
    """Use LLM to decide the next action based on observation."""
    try:
        # Enhanced prompt for better hard task performance
        available_routes = obs.get('available_routes', [])
        active_incidents = obs.get('active_incidents', [])
        current_route = obs.get('current_route')
        distance_remaining = obs.get('distance_remaining_km', 0)
        
        # Analyze route incident severity
        route_summaries = []
        for route in available_routes:
            incident_count = route.get('incident_count', 0)
            route_summaries.append(f"{route['route_id']}: {route.get('estimated_time_min', 0)} min, {incident_count} incidents, safety={route.get('safety_score', 0):.2f}")
        
        # Check current route risk
        current_risk = "LOW"
        if current_route:
            route_incidents = [i for i in active_incidents if current_route in i.get('affects_routes', [])]
            high_sev = [i for i in route_incidents if i.get('severity') == 'high']
            if high_sev:
                current_risk = f"HIGH ({len(high_sev)} high-severity incidents)"
            elif route_incidents:
                current_risk = f"MEDIUM ({len(route_incidents)} incidents)"
        
        prompt = f"""You are an expert routing algorithm for urban navigation. MAXIMIZE safety and efficiency.

CURRENT STATE:
- Step: {obs.get('step')} (distance remaining: {distance_remaining} km)
- Current route: {current_route if current_route else 'NOT SELECTED'}
- Current route risk: {current_risk}
- Available routes: {'; '.join(route_summaries)}
- Active incidents: {len(active_incidents)} total

PRIORITY RULES (in order):
1. SAFETY FIRST: Avoid high-severity incidents at all costs
2. If on dangerous route (high-severity incidents): REROUTE to safest alternative immediately
3. If no route selected: Pick SAFEST route (highest safety_score), break ties by time
4. If route is safe: CONTINUE (don't waste steps on redundant selection)
5. If destination reached: STOP

RESPONSE: Select ONE action. Reply with ONLY valid JSON:
{{"action_type": "select_route|continue|reroute|stop", "route_id": "fastest|safe|eco|null"}}

Think step-by-step:
- Safety > Time > Fuel
- Avoid incidents > Continue safe → Select once, continue until unsafe """

        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        
        # Parse LLM response
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        action = json.loads(content)
        return action
    except Exception as e:
        # Fallback: use simple heuristics if LLM fails
        print(f"LLM decision failed ({e}), using fallback heuristic")
        return _fallback_decide_action(obs)


def _fallback_decide_action(obs: dict) -> dict:
    """Fallback rule-based decision when LLM is unavailable."""
    current_route = obs.get("current_route")
    routes = obs.get("available_routes", [])
    incidents = obs.get("active_incidents", [])
    distance = obs.get("distance_remaining_km", 0)
    
    # If destination reached
    if distance <= 0 or obs.get("episode_done"):
        return {"action_type": "stop"}
    
    # No route selected yet
    if current_route is None:
        if not routes:
            return {"action_type": "continue"}
        # Pick safest+fastest clean route
        clean = [r for r in routes if r.get("incident_count", 0) == 0]
        pool = clean if clean else routes
        best = min(pool, key=lambda r: r.get("estimated_time_min", 999))
        return {"action_type": "select_route", "route_id": best["route_id"]}
    
    # Check for high-severity incidents on current route
    route_incidents = [
        i for i in incidents
        if current_route in i.get("affects_routes", [])
    ]
    high_sev = [i for i in route_incidents if i.get("severity") == "high"]
    
    if high_sev:
        # Reroute to cleanest option
        clean = [r for r in routes if r.get("incident_count", 0) == 0]
        if clean and clean[0]["route_id"] != current_route:
            best = min(clean, key=lambda r: r.get("estimated_time_min", 999))
            return {"action_type": "reroute", "route_id": best["route_id"]}
    
    # Route is clean — keep going
    return {"action_type": "continue"}


def run_inference(task: str = "easy", seed: int = 42):
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    # Reset — extracts observation from wrapped response
    reset_response = client.post("/reset", json={"task": task, "seed": seed}).json()
    obs = reset_response["observation"]

    trajectory = []
    done = False
    step = 0

    while not done and step < 20:
        # Decide action using LLM (with fallback to heuristics)
        action = llm_decide_action(obs)

        # Take the step
        step_response = client.post("/step", json=action).json()

        # Extract observation from wrapped response
        next_obs = step_response["observation"]
        trajectory.append({
            "obs": next_obs,
            "action": action,
            "reward": step_response["reward"],
        })
        obs = next_obs
        done = step_response["done"]
        step += 1

    # Grade
    score_result = client.post(
        "/grader", json={"task": task, "trajectory": trajectory}
    ).json()
    score = score_result.get("score", 0.0)

    return {"task": task, "score": score, "steps": step}


if __name__ == "__main__":
    results = []
    for task in ["easy", "medium", "hard"]:
        result = run_inference(task=task)
        results.append(result)
        # Print human-readable log for debugging
        print(f"Task: {task} | Steps: {result['steps']} | Score: {result['score']}")
    
    # Output JSON for validator to parse
    print("\n" + json.dumps(results, indent=2))
