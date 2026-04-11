"""
Tests for the URBANEX graders.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from graders import grade_easy, grade_medium, grade_hard
from models.action import Action
from models.observation import Observation, RouteOption, Incident
from models.reward import Reward


def _make_route(route_id="safe", time=28.0, incidents=0, fuel=0.77, safety=0.9):
    return RouteOption(
        route_id=route_id,
        estimated_time_min=time,
        incident_count=incidents,
        fuel_cost_score=fuel,
        safety_score=safety,
    )


def _make_obs(step=0, done=False, current_route=None, incidents=None, distance=5.0):
    routes = [
        _make_route("fastest", 20.0, 0, 1.0, 0.8),
        _make_route("safe",    28.0, 0, 0.77, 0.95),
        _make_route("eco",     25.0, 0, 0.54, 0.85),
    ]
    return Observation(
        step=step,
        current_location=[12.97, 77.59],
        destination=[12.93, 77.62],
        available_routes=routes,
        active_incidents=incidents or [],
        traffic_level="low",
        weather="clear",
        current_route=current_route,
        distance_remaining_km=distance,
        episode_done=done,
    )


def _make_action(action_type="continue", route_id=None):
    return Action(action_type=action_type, route_id=route_id)


def _make_reward(total=0.1):
    return Reward(total=total, safety_component=0.0, time_component=total, fuel_component=0.0, penalty=0.0, reason="test")


def _successful_trajectory():
    return [
        {"obs": _make_obs(0, False, None, distance=8.0),       "action": _make_action("select_route", "safe"), "reward": _make_reward(0.2)},
        {"obs": _make_obs(1, False, "safe", distance=6.0),     "action": _make_action("continue"),             "reward": _make_reward(0.1)},
        {"obs": _make_obs(2, False, "safe", distance=4.0),     "action": _make_action("continue"),             "reward": _make_reward(0.1)},
        {"obs": _make_obs(3, True,  "safe", distance=0.3),     "action": _make_action("continue"),             "reward": _make_reward(1.0)},
    ]


def _failed_trajectory():
    return [
        {"obs": _make_obs(0, False, None, distance=8.0),       "action": _make_action("select_route", "fastest"), "reward": _make_reward(0.1)},
        {"obs": _make_obs(1, False, "fastest", distance=6.0),  "action": _make_action("stop"),                    "reward": _make_reward(-1.0)},
        {"obs": _make_obs(2, True,  "fastest", distance=6.0),  "action": _make_action("stop"),                    "reward": _make_reward(-1.0)},
    ]


# Test 1: grader returns float in [0.0, 1.0]
@pytest.mark.parametrize("grader", [grade_easy, grade_medium, grade_hard])
def test_grader_returns_float_in_range(grader):
    traj = _successful_trajectory()
    score = grader(traj)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# Test 2: successful trajectory scores higher than failed trajectory
@pytest.mark.parametrize("grader", [grade_easy, grade_medium, grade_hard])
def test_successful_scores_higher_than_failed(grader):
    success_score = grader(_successful_trajectory())
    fail_score = grader(_failed_trajectory())
    assert success_score > fail_score, (
        f"{grader.__module__}: expected success ({success_score}) > fail ({fail_score})"
    )


# Test 3: grader is deterministic – same trajectory = same score
@pytest.mark.parametrize("grader", [grade_easy, grade_medium, grade_hard])
def test_grader_is_deterministic(grader):
    traj = _successful_trajectory()
    assert grader(traj) == grader(traj)


# Test 4: grader never returns same score for wildly different trajectories
@pytest.mark.parametrize("grader", [grade_easy, grade_medium, grade_hard])
def test_grader_differentiates_trajectories(grader):
    success_score = grader(_successful_trajectory())
    fail_score = grader(_failed_trajectory())
    assert success_score != fail_score, (
        f"{grader.__module__}: scores are identical ({success_score}) for different trajectories!"
    )


# Test 5: empty trajectory returns the minimum open-interval score
@pytest.mark.parametrize("grader", [grade_easy, grade_medium, grade_hard])
def test_empty_trajectory_returns_zero(grader):
    assert grader([]) == 0.05
