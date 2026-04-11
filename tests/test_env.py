"""
Tests for the URBANEX environment.
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from environment.urbanex_env import UrbanexEnv
from models.action import Action


def _make_env(task="easy") -> UrbanexEnv:
    env = UrbanexEnv(task=task, seed=42)
    return env


# Test 1: reset() returns valid Observation
def test_reset_returns_valid_observation():
    env = _make_env("easy")
    obs = env.reset()
    assert obs is not None
    assert obs.step == 0
    assert len(obs.current_location) == 2
    assert len(obs.destination) == 2
    assert len(obs.available_routes) == 3
    assert obs.distance_remaining_km > 0
    assert obs.episode_done is False
    assert isinstance(obs.situation_summary, str)
    assert obs.situation_summary


# Test 2: step() with valid action returns (obs, reward, done, info)
def test_step_returns_valid_tuple():
    env = _make_env("easy")
    env.reset()
    action = Action(action_type="select_route", route_id="safe")
    obs, reward, done, info = env.step(action)
    assert obs is not None
    assert reward is not None
    assert isinstance(done, bool)
    assert isinstance(info, dict)


# Test 3: episode terminates within max_steps
def test_episode_terminates_within_max_steps():
    for task in ["easy", "medium", "hard"]:
        env = _make_env(task)
        obs = env.reset()
        done = False
        steps = 0
        max_steps = env.config["max_steps"]
        while not done and steps < max_steps + 5:
            action = Action(action_type="continue")
            obs, reward, done, info = env.step(action)
            steps += 1
        assert done, f"Episode did not terminate for task '{task}'"
        assert steps <= max_steps + 1


# Test 4: reward is never None
def test_reward_is_never_none():
    env = _make_env("easy")
    env.reset()
    actions = [
        Action(action_type="select_route", route_id="fastest"),
        Action(action_type="continue"),
        Action(action_type="reroute", route_id="safe"),
    ]
    for action in actions:
        try:
            _, reward, done, _ = env.step(action)
            assert reward is not None
            assert reward.total is not None
            if done:
                break
        except RuntimeError:
            break


# Test 5: done=True when destination reached
def test_done_true_when_destination_reached():
    env = _make_env("easy")
    env.reset()
    # Force city to be at destination
    env._city.current_location = env._city.destination
    action = Action(action_type="continue")
    obs, reward, done, info = env.step(action)
    assert done is True


# Test 6: all three tasks can be instantiated and run 5 steps without error
@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_all_tasks_run_5_steps(task):
    env = _make_env(task)
    obs = env.reset()
    assert obs is not None
    for i in range(5):
        action = Action(action_type="select_route", route_id="safe") if i == 0 else Action(action_type="continue")
        obs, reward, done, info = env.step(action)
        assert reward is not None
        if done:
            break


def test_hard_task_forced_incident_and_weather_shift():
    env = _make_env("hard")
    obs = env.reset()
    obs, reward, done, info = env.step(Action(action_type="select_route", route_id="safe"))
    while obs.step < 8 and not done:
        obs, reward, done, info = env.step(Action(action_type="continue"))

    assert any(
        incident.type == "accident"
        and incident.severity == "high"
        and "fastest" in incident.affects_routes
        for incident in obs.active_incidents
    )
    assert obs.weather == "heavy_rain"


# Test 7: reset() clears state between episodes (no leaking)
def test_reset_clears_state():
    env = _make_env("easy")
    obs1 = env.reset()
    # Step a few times
    for _ in range(3):
        try:
            env.step(Action(action_type="continue"))
        except RuntimeError:
            break
    # Reset again
    obs2 = env.reset()
    # Step should be 0 after reset
    assert obs2.step == 0
    assert obs2.episode_done is False
