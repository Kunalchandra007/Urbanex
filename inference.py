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
        prompt = f"""You are a smart navigation agent for city routing. Analyze this observation and decide the best action.

Observation:
- Current step: {obs.get('step')}
- Current location: {obs.get('current_location')}
- Destination: {obs.get('destination')}
- Current route: {obs.get('current_route')}
- Available routes: {len(obs.get('available_routes', []))} options
- Active incidents: {len(obs.get('active_incidents', []))} incidents
- Traffic level: {obs.get('traffic_level')}
- Distance remaining: {obs.get('distance_remaining_km')} km
- Episode done: {obs.get('episode_done')}

If no route is selected yet, pick the safest route with lowest incidents.
If on a route with high-severity incidents, reroute to a clean route.
Otherwise, continue on the current route.
If destination reached or no distance remaining, stop.

Respond with ONLY valid JSON (no markdown, no extra text):
{{"action_type": "select_route|continue|reroute|stop", "route_id": "optional_route_id"}}"""

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

    print(f"Task: {task} | Steps: {step} | Score: {score}")
    return {"task": task, "score": score, "steps": step}


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task=task)
