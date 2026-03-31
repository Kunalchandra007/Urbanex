import os
import json
import httpx

BASE_URL = os.getenv("SPACE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# HuggingFace Inference API URL
HF_API_URL = "https://api-inference.huggingface.co/models"

# Check if HF token is available for LLM calls
llm_available = bool(HF_TOKEN and HF_TOKEN.strip())


def llm_decide_action(obs: dict) -> dict:
    """Use HuggingFace Inference API to decide the next action."""
    if not llm_available:
        return _fallback_decide_action(obs)
    
    try:
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
2. If on dangerous route: REROUTE to safest alternative immediately
3. If no route selected: Pick SAFEST route (highest safety_score), break ties by time
4. If route is safe: CONTINUE (don't waste steps on redundant selection)
5. If destination reached: STOP

RESPONSE: Select ONE action. Reply with ONLY valid JSON:
{{"action_type": "select_route|continue|reroute|stop", "route_id": "fastest|safe|eco|null"}}

Think step-by-step: Safety > Time > Fuel. Avoid incidents > Continue safe → Select once."""

        # Call HuggingFace Inference API
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 100,
                "temperature": 0.1,
                "do_sample": False
            }
        }
        
        response = httpx.post(
            f"{HF_API_URL}/{MODEL_NAME}",
            json=payload,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract text from HF response
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            else:
                content = str(result)
            
            # Extract JSON from response
            if "{" in content and "}" in content:
                json_start = content.rfind("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]
                action = json.loads(json_str)
                return action
        
        # If HF API call failed, use fallback
        return _fallback_decide_action(obs)
        
    except Exception as e:
        print(f"[INFO] LLM decision failed ({type(e).__name__}), using fallback heuristic")
        return _fallback_decide_action(obs)


def _fallback_decide_action(obs: dict) -> dict:
    """Fallback rule-based decision when LLM is unavailable.
    Enhanced for hard task: dynamic_rerouting."""
    current_route = obs.get("current_route")
    routes = obs.get("available_routes", [])
    incidents = obs.get("active_incidents", [])
    distance = obs.get("distance_remaining_km", 0)
    step = obs.get("step", 0)
    
    # If destination reached
    if distance <= 0 or obs.get("episode_done"):
        return {"action_type": "stop"}
    
    # No route selected yet - pick safest
    if current_route is None:
        if not routes:
            return {"action_type": "continue"}
        # For hard task: prioritize safety heavily
        # Sort by: incident_count (ascending), then safety_score (descending)
        clean = [r for r in routes if r.get("incident_count", 0) == 0]
        if clean:
            best = max(clean, key=lambda r: r.get("safety_score", 0))
        else:
            # No clean route - pick lowest incident count, highest safety
            sorted_routes = sorted(
                routes,
                key=lambda r: (r.get("incident_count", 999), -r.get("safety_score", 0))
            )
            best = sorted_routes[0]
        return {"action_type": "select_route", "route_id": best["route_id"]}
    
    # Check for incidents on current route
    route_incidents = [
        i for i in incidents
        if current_route in i.get("affects_routes", [])
    ]
    high_sev = [i for i in route_incidents if i.get("severity") == "high"]
    
    # Hard task: be aggressive about rerouting on ANY incident
    if high_sev or (len(route_incidents) > 0 and step > 2):
        # Look for clean routes or safest alternative
        clean = [r for r in routes if r.get("incident_count", 0) == 0 and r["route_id"] != current_route]
        if clean:
            best = max(clean, key=lambda r: r.get("safety_score", 0))
            return {"action_type": "reroute", "route_id": best["route_id"]}
        
        # No clean route - pick highest safety among alternatives
        alternatives = [r for r in routes if r["route_id"] != current_route]
        if alternatives:
            best = max(alternatives, key=lambda r: (
                -r.get("incident_count", 0),
                r.get("safety_score", 0)
            ))
            return {"action_type": "reroute", "route_id": best["route_id"]}
    
    # Route is safe — continue
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
