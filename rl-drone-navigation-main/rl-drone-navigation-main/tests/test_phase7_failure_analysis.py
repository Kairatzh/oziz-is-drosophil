"""
Tests for the Phase 7 failure-analysis helpers. The interesting thing to
unit-test here is the corridor-clearance placement geometry
(place_obstacle_at_clearance) -- it had a real bug during development
(an unclamped offset let very negative "clearance" values wrap around to
the OPPOSITE, less-occluded side of the line, silently corrupting the
sweep), and the states/actions length contract that render_drone_gif
depends on.
"""

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.phase7_failure_analysis import (
    CHECKPOINT,
    place_obstacle_at_clearance,
    run_clearance_episode,
    run_episode,
)


def _true_clearance(env):
    x, z = env.state[0], env.state[1]
    return env._goal_los_clearance(x, z, env.goal)


def test_place_obstacle_at_clearance_matches_requested_value():
    env = QuadrotorEnv(seed=0)
    rng = np.random.default_rng(0)
    for target in (-0.4, -0.2, 0.0, 0.2, 0.5, 1.0):
        for _ in range(10):
            place_obstacle_at_clearance(env, rng, radius=0.8, clearance=target, min_start_goal_dist=4.0)
            assert np.isclose(_true_clearance(env), target, atol=1e-6)


def test_place_obstacle_at_clearance_clamps_beyond_center_on_line():
    """Requesting clearance more negative than -radius is geometrically
    impossible (the obstacle center can't be a negative distance from the
    line) -- it should clamp to the "center exactly on the line" case,
    clearance == -radius, not wrap around to a less-occluded geometry."""
    env = QuadrotorEnv(seed=0)
    rng = np.random.default_rng(0)
    for target in (-1.2, -0.9, -0.8):
        place_obstacle_at_clearance(env, rng, radius=0.8, clearance=target, min_start_goal_dist=4.0)
        assert np.isclose(_true_clearance(env), -0.8, atol=1e-6)


def test_place_obstacle_at_clearance_respects_min_start_goal_distance():
    env = QuadrotorEnv(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(10):
        place_obstacle_at_clearance(env, rng, radius=0.8, clearance=0.0, min_start_goal_dist=4.0)
        sx, sz = env.state[0], env.state[1]
        gx, gz = env.goal
        assert np.hypot(gx - sx, gz - sz) >= 4.0


def test_run_episode_actions_and_states_are_same_length():
    """Regression test: render_drone_gif indexes actions[i] for every
    states[i], so a length mismatch crashes GIF export (see git history --
    this broke once when actions wasn't seeded to match states[0])."""
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)
    env = QuadrotorEnv(seed=0)
    ep = run_episode(agent, env)
    assert len(ep["actions"]) == len(ep["states"])
    assert len(ep["min_lidar"]) == len(ep["states"])


def test_run_clearance_episode_actions_and_states_are_same_length():
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)
    env = QuadrotorEnv(seed=0)
    rng = np.random.default_rng(0)
    ep = run_clearance_episode(agent, env, rng, clearance=0.2)
    assert len(ep["actions"]) == len(ep["states"])
