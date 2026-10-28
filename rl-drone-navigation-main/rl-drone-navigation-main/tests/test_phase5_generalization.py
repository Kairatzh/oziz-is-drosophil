"""
Tests for the Phase 5 generalization test conditions. As with
tests/test_curriculum.py, the actual evaluation run is the real validation
of the numbers reported in PHASE5_GENERALIZATION_RESULTS.md; what's worth
unit-testing here is the *configuration* -- the new random-obstacle
generators and the fixed wall formation must reliably produce valid,
non-degenerate episodes, and must actually differ from the Level 1.5
training generator along the axis they claim to test (no guaranteed clear
line of sight).
"""

import numpy as np
import pytest

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.phase5_generalization_eval import (
    CHECKPOINT,
    TEST_CONDITIONS,
    WALL_OBSTACLES,
    make_random_obstacles_fn,
)


@pytest.mark.parametrize("cond", TEST_CONDITIONS, ids=[c["name"] for c in TEST_CONDITIONS])
def test_condition_reset_reliably_finds_free_start_and_goal(cond):
    env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
    for _ in range(50):
        env.reset()
        x, z = env.state[0], env.state[1]
        gx, gz = env.goal
        for obs in env.obstacles:
            assert obs.distance_to_point(x, z) > env.DRONE_RADIUS
            assert obs.distance_to_point(gx, gz) > env.GOAL_RADIUS


@pytest.mark.parametrize("cond", TEST_CONDITIONS, ids=[c["name"] for c in TEST_CONDITIONS])
def test_condition_obstacles_stay_within_world_bounds(cond):
    env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
    for _ in range(20):
        env.reset()
        for obs in env.obstacles:
            assert 0 <= obs.x - obs.radius and obs.x + obs.radius <= env.WORLD_WIDTH
            assert 0 <= obs.z - obs.radius and obs.z + obs.radius <= env.WORLD_HEIGHT


def test_random_obstacles_fn_places_requested_count():
    fn = make_random_obstacles_fn(3, radius=0.8)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obstacles = fn(rng, (1.0, 1.0), (9.0, 7.0))
        assert len(obstacles) == 3


def test_random_obstacles_fn_does_not_guarantee_clear_line_of_sight():
    """The whole point of this generator (unlike make_level15_obstacle_fn)
    is that it does NOT keep the straight start->goal line clear -- verify
    that occlusion actually happens at least some of the time, otherwise
    the "no LOS guarantee" condition wouldn't be testing anything new."""
    env = QuadrotorEnv(obstacle_fn=make_random_obstacles_fn(1, radius=0.8), seed=1000)
    occluded_count = 0
    for _ in range(100):
        env.reset()
        start = env.state[:2]
        if env._goal_los_clearance(start[0], start[1], env.goal) <= 0:
            occluded_count += 1
    assert occluded_count > 0


def test_wall_obstacles_leave_navigable_gaps():
    """The wall formation is meant to be hard, not impossible -- each gap
    (including to the world edges) must be wide enough for the drone body
    to physically fit through."""
    env = QuadrotorEnv(obstacles=WALL_OBSTACLES)
    zs = sorted(o.z - o.radius for o in WALL_OBSTACLES) + [env.WORLD_HEIGHT]
    prev_top = 0.0
    gaps = []
    for obs in sorted(WALL_OBSTACLES, key=lambda o: o.z):
        gaps.append(obs.z - obs.radius - prev_top)
        prev_top = obs.z + obs.radius
    gaps.append(env.WORLD_HEIGHT - prev_top)
    assert all(gap > 2 * env.DRONE_RADIUS for gap in gaps)


def test_primary_checkpoint_loads_and_matches_env_dims():
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)  # raises if shapes stored in the checkpoint don't match
    env = QuadrotorEnv(seed=0)
    obs = env.reset()
    action = agent.act_deterministic(obs)
    assert action.shape == (2,)
