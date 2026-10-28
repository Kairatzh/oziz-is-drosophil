"""
Tests for the Phase 4 curriculum configuration. The actual multi-minute
training run is the real validation of the progression logic (identical
control flow to Phase 3's already-tested training loop); what's unique
and worth unit-testing here is the curriculum *data* -- the four
obstacle layouts must each leave enough free space for reset() to work
reliably, and the level definitions must be well-formed.
"""

import pytest

from env.quadrotor_env import QuadrotorEnv
from scripts.train_drone_curriculum import (
    CURRICULUM,
    EVAL_EPISODES,
    MAX_UPDATES_PER_LEVEL,
    SUCCESS_THRESHOLD,
)


def test_curriculum_level_names_are_unique():
    names = [level["name"] for level in CURRICULUM]
    assert len(names) == len(set(names))


def test_curriculum_has_five_levels_in_increasing_obstacle_count():
    assert len(CURRICULUM) == 5
    obstacle_counts = [len(level["obstacles"]) for level in CURRICULUM]
    assert obstacle_counts == sorted(obstacle_counts)  # non-decreasing difficulty
    assert obstacle_counts[0] == 0  # level 1 truly has no obstacles


@pytest.mark.parametrize("level_config", CURRICULUM, ids=[l["name"] for l in CURRICULUM])
def test_level_reset_reliably_finds_free_start_and_goal(level_config):
    env = QuadrotorEnv(obstacles=level_config["obstacles"], seed=0)
    for _ in range(100):
        env.reset()
        x, z = env.state[0], env.state[1]
        gx, gz = env.goal
        for obs in level_config["obstacles"]:
            assert obs.distance_to_point(x, z) > env.DRONE_RADIUS
            assert obs.distance_to_point(gx, gz) > env.GOAL_RADIUS


@pytest.mark.parametrize("level_config", CURRICULUM, ids=[l["name"] for l in CURRICULUM])
def test_level_obstacles_stay_within_world_bounds(level_config):
    env = QuadrotorEnv(obstacles=level_config["obstacles"])
    for obs in level_config["obstacles"]:
        assert 0 <= obs.x - obs.radius and obs.x + obs.radius <= env.WORLD_WIDTH
        assert 0 <= obs.z - obs.radius and obs.z + obs.radius <= env.WORLD_HEIGHT


def test_progression_criterion_constants_are_sane():
    assert 0.0 < SUCCESS_THRESHOLD <= 1.0
    assert EVAL_EPISODES > 0
    assert MAX_UPDATES_PER_LEVEL > 0
