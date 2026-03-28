import os
import httpx

BASE_URL = os.getenv("SPACE_URL", "http://localhost:7860")


def run_inference(task: str = "easy", seed: int = 42):
    client = httpx.Client(base_url=BASE_URL, timeout=30)

    # Reset
    obs = client.post("/reset", json={"task": task, "seed": seed}).json()

    trajectory = []
    done = False
    step = 0

    while not done and step < 20:
        routes = obs.get("available_routes", [])
        incidents = obs.get("active_incidents", [])
        current_route = obs.get("current_route")

        # Mirror the rule-based agent logic:
        if current_route is None:
            # No route selected yet — pick safest+fastest
            clean = [r for r in routes if r.get("incident_count", 0) == 0]
            pool = clean if clean else routes
            if pool:
                best = min(pool, key=lambda r: r.get("estimated_time_min", 999))
                action = {"action_type": "select_route", "route_id": best["route_id"]}
            else:
                action = {"action_type": "continue"}
        else:
            # Check if current route has high-severity incidents
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
                    action = {"action_type": "reroute", "route_id": best["route_id"]}
                else:
                    action = {"action_type": "continue"}
            else:
                # Route is clean — keep going
                action = {"action_type": "continue"}

        result = client.post("/step", json=action).json()

        # Append POST-step observation (matches baseline_agent.py behaviour)
        next_obs = result["observation"]
        trajectory.append({
            "obs": next_obs,     # ← post-step obs, so episode_done is correct
            "action": action,
            "reward": result["reward"],
        })
        obs = next_obs
        done = result["done"]
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
