"""
CLI visualizer for URBANEX episodes.

Activated via `URBANEX_RENDER=1` or `visualize=True` in `run_baseline()`.
Shows a compact ASCII summary of route, reward, incidents, and distance.
"""
import os

ROUTE_ICONS = {"fastest": "[F]", "safe": "[S]", "eco": "[E]", None: "[?]"}
SEVERITY_ICONS = {"low": "LOW", "medium": "MED", "high": "HIGH"}
WEATHER_ICONS = {"clear": "SUN", "rain": "RAN", "fog": "FOG"}
TRAFFIC_ICONS = {"low": "TLO", "medium": "TMD", "high": "THI"}


def _reward_bar(value: float, width: int = 10) -> str:
    """Render a simple ASCII bar chart for reward (-1 to +1)."""
    filled = int((value + 1.0) / 2.0 * width)
    filled = max(0, min(width, filled))
    bar = "#" * filled + "." * (width - filled)
    sign = "+" if value >= 0 else ""
    return f"[{bar}] {sign}{value:.3f}"


def render_step(step: int, obs, action, reward, info: dict = None) -> None:
    route_icon = ROUTE_ICONS.get(action.route_id or obs.current_route, "[?]")
    route_name = action.route_id or obs.current_route or "none"
    action_abbr = {
        "select_route": "SELECT",
        "reroute":      "REROUTE",
        "continue":     "CONT  ",
        "stop":         "STOP  ",
        "report_incident": "REPORT",
    }.get(action.action_type, action.action_type[:6].upper())

    inc_count = len(obs.active_incidents)
    inc_str = f"! {inc_count}" if inc_count > 0 else "0 inc"

    weather = WEATHER_ICONS.get(obs.weather, "?")
    traffic = TRAFFIC_ICONS.get(obs.traffic_level, "?")
    dist = obs.distance_remaining_km
    bar = _reward_bar(reward.total)
    hidden_pen = info.get("hidden_penalty_this_step", 0.0) if info else 0.0
    hidden_str = f"  HIDDEN_PEN={hidden_pen:.3f}" if hidden_pen > 0 else ""

    print(
        f"  Step {step:02d} | {action_abbr} {route_icon} {route_name:<10} | "
        f"{bar} | {inc_str} | {dist:5.2f}km | "
        f"{weather}/{traffic}{hidden_str}"
    )


def render_episode_header(task: str, seed: int) -> None:
    print()
    print("=" * 80)
    print(f"  URBANEX -- Task: {task.upper()}   seed={seed}")
    print("=" * 80)
    print(f"  {'Step':6} | {'Action':<22} | {'Reward':14} | {'Inc':6} | {'Dist':6} | Env")
    print(f"  {'-'*72}")


def render_episode_footer(steps: int, total_reward: float, score: float, done: bool) -> None:
    outcome = "REACHED DESTINATION" if done else "FAILED (timeout/stop)"
    print(f"  {'-'*72}")
    print(f"  >> {outcome}")
    print(f"  Steps: {steps}  |  Total Reward: {total_reward:+.4f}  |  Grader Score: {score:.4f}")
    print("=" * 80)
    print()


def should_render() -> bool:
    return os.environ.get("URBANEX_RENDER", "0").strip() in ("1", "true", "yes")
