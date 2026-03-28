"""
Baseline agents for Velora OpenEnv.
- rule_based_agent: deterministic policy, no API key required (PRIMARY)
- llm_agent: GPT-4o-mini agent (requires OPENAI_API_KEY)
- run_baseline: run full episode and return graded score
"""
import os
from typing import Optional

from environment.velora_env import VeloraEnv
from graders import grade_easy, grade_medium, grade_hard
from models.action import Action
from models.observation import Observation

GRADER_MAP = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


# ---------------------------------------------------------------------------
# Rule-based agent (no API key needed)
# ---------------------------------------------------------------------------

def rule_based_agent(observation: Observation) -> Action:
    """
    Simple deterministic policy:
    1. If not yet on a route → select best available route.
    2. If current route has a new high-severity incident → reroute to safe.
    3. If on a route with no incidents → continue.
    4. Otherwise → pick route with highest safety_score.
    """
    current_route = observation.current_route
    routes = observation.available_routes
    incidents = observation.active_incidents

    if not routes:
        return Action(action_type="continue")

    # Find incidents affecting the current route
    current_route_incidents = []
    if current_route:
        current_route_incidents = [
            i for i in incidents if current_route in i.affects_routes
        ]

    # If no route selected yet → pick best option
    if current_route is None:
        best_route = _pick_best_route(routes, incidents)
        return Action(action_type="select_route", route_id=best_route)

    # If high-severity incident on current route → reroute to safe
    high_sev = [i for i in current_route_incidents if i.severity == "high"]
    if high_sev:
        # Check safe route isn't also compromised
        safe_incidents = [i for i in incidents if "safe" in i.affects_routes and i.severity == "high"]
        if not safe_incidents:
            return Action(action_type="reroute", route_id="safe")
        else:
            # Pick whichever route has the best safety score
            best = _pick_best_route(routes, incidents)
            if best != current_route:
                return Action(action_type="reroute", route_id=best)

    # If current route is clean → continue
    if not current_route_incidents:
        return Action(action_type="continue")

    # Medium severity on current route → reroute to cleaner option
    safer = _pick_best_route(routes, incidents)
    if safer != current_route:
        return Action(action_type="reroute", route_id=safer)

    return Action(action_type="continue")


def _pick_best_route(routes, incidents) -> str:
    """
    Pick route with zero incidents (prefer lowest time),
    otherwise pick highest safety_score.
    """
    clean_routes = [r for r in routes if r.incident_count == 0]
    if clean_routes:
        return min(clean_routes, key=lambda r: r.estimated_time_min).route_id
    return max(routes, key=lambda r: r.safety_score).route_id


# ---------------------------------------------------------------------------
# LLM-based agent (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def llm_agent(observation: Observation) -> Action:
    """
    Calls OpenAI API (gpt-4o-mini) with the observation as JSON.
    Falls back to rule_based_agent if API key is not available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[llm_agent] OPENAI_API_KEY not set. Falling back to rule_based_agent.")
        return rule_based_agent(observation)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        obs_json = observation.model_dump_json(indent=2)

        system_prompt = (
            "You are a smart city navigation agent for Bangalore. "
            "Given the current navigation observation as JSON, choose the best action "
            "to reach the destination safely and efficiently. "
            "Respond ONLY with a valid JSON object matching this schema:\n"
            '{"action_type": "select_route"|"reroute"|"report_incident"|"continue"|"stop", '
            '"route_id": "fastest"|"safe"|"eco"|null, '
            '"incident_type": string|null, '
            '"incident_lat": number|null, '
            '"incident_lng": number|null}'
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Observation:\n{obs_json}"},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        # Parse JSON response
        import json
        action_dict = json.loads(content)
        return Action(**action_dict)

    except Exception as e:
        print(f"[llm_agent] Error calling OpenAI: {e}. Falling back to rule_based_agent.")
        return rule_based_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_baseline(task: str = "easy", agent: str = "rule_based", seed: int = 42, visualize: bool = False) -> dict:
    """
    Run a full episode with the specified agent and return graded score.

    Args:
        task: "easy" | "medium" | "hard"
        agent: "rule_based" | "llm"
        seed: random seed for reproducibility
        visualize: print step-by-step CLI table (or set VELORA_RENDER=1)

    Returns:
        dict with keys: task, agent, score, steps, total_reward
    """
    from baseline.visualizer import render_step, render_episode_header, render_episode_footer, should_render
    show = visualize or should_render()

    env = VeloraEnv(task=task, seed=seed)
    obs = env.reset()

    agent_fn = llm_agent if agent == "llm" else rule_based_agent

    trajectory = []
    total_reward = 0.0
    done = False
    steps = 0

    if show:
        render_episode_header(task, seed)

    while not done:
        action = agent_fn(obs)
        new_obs, reward, done, info = env.step(action)
        trajectory.append({"obs": new_obs, "action": action, "reward": reward})
        total_reward += reward.total
        steps += 1

        if show:
            render_step(steps, new_obs, action, reward, info)

        obs = new_obs

    grader_fn = GRADER_MAP.get(task, grade_easy)
    score = grader_fn(trajectory)

    if show:
        render_episode_footer(steps, total_reward, score, done)

    print(f"[baseline] task={task} agent={agent} seed={seed} steps={steps} "
          f"total_reward={total_reward:.3f} score={score:.4f}")

    return {
        "task": task,
        "agent": agent,
        "score": score,
        "steps": steps,
        "total_reward": round(total_reward, 4),
    }


if __name__ == "__main__":
    import sys
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "easy"
    agent_arg = sys.argv[2] if len(sys.argv) > 2 else "rule_based"
    run_baseline(task=task_arg, agent=agent_arg, visualize=True)
