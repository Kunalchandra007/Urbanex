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
        # Rule-based agent: pick safest fastest route
        routes = obs.get("available_routes", [])
        safe_routes = [r for r in routes if r.get("incident_count", 0) == 0]

        if safe_routes:
            # Lowest risk, then fastest
            best = min(safe_routes, key=lambda r: (r.get("risk_score", 1.0), r.get("estimated_time_min", 999)))
            action = {"action_type": "select_route", "route_id": best["route_id"]}
        elif routes:
            # All risky — pick lowest risk
            best = min(routes, key=lambda r: r.get("risk_score", 1.0))
            action = {"action_type": "select_route", "route_id": best["route_id"]}
        else:
            action = {"action_type": "continue"}

        result = client.post("/step", json=action).json()

        # Use "obs" key — matches what the grader expects
        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": result["reward"],
        })
        obs = result["observation"]
        done = result["done"]
        step += 1

    # Grade
    score_result = client.post("/grader", json={"task": task, "trajectory": trajectory}).json()
    score = score_result.get("score", 0.0)

    print(f"Task: {task} | Steps: {step} | Score: {score}")
    return {"task": task, "score": score, "steps": step}


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task=task)
