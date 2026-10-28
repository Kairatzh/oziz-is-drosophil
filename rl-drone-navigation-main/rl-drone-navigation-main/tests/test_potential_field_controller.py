"""
Tests for the Phase 6 hand-coded potential-field baseline. As with
tests/test_quadrotor_env.py's physics sanity checks, these validate the
controller's building blocks (attractive/repulsive fields, the
force-to-attitude inversion) in isolation, plus one closed-loop regression
test that it can actually fly the drone to a goal.
"""

import numpy as np
import pytest

from baseline.potential_field_controller import PotentialFieldController
from env.quadrotor_env import CircleObstacle, QuadrotorEnv


def test_action_always_within_bounds():
    ctrl = PotentialFieldController()
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, z = rng.uniform(0, 10), rng.uniform(0, 8)
        vx, vz = rng.uniform(-3, 3), rng.uniform(-3, 3)
        theta, omega = rng.uniform(-np.pi, np.pi), rng.uniform(-5, 5)
        goal = (rng.uniform(0, 10), rng.uniform(0, 8))
        obstacles = [CircleObstacle(x=rng.uniform(0, 10), z=rng.uniform(0, 8), radius=0.8)]
        action = ctrl.act(x, z, vx, vz, theta, omega, goal, obstacles)
        assert action.shape == (2,)
        assert np.all(action >= 0.0) and np.all(action <= 1.0)


def test_attractive_accel_points_toward_goal_when_stationary():
    ctrl = PotentialFieldController()
    a = ctrl._attractive_accel(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    # accel should point in the same direction as (goal - pos) = (3, 4)
    assert a[0] > 0 and a[1] > 0
    assert np.isclose(a[0] / a[1], 3.0 / 4.0)


def test_attractive_accel_capped_at_long_range():
    ctrl = PotentialFieldController(max_att_accel=3.0)
    a = ctrl._attractive_accel(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([100.0, 0.0]))
    assert np.hypot(*a) <= 3.0 + 1e-9


def test_repulsive_accel_zero_outside_influence_margin():
    ctrl = PotentialFieldController(influence_margin=1.0)
    obstacles = [CircleObstacle(x=5.0, z=5.0, radius=0.8)]
    a = ctrl._repulsive_accel(np.array([5.0, 5.0 + 0.8 + 1.5]), obstacles)  # 1.5m clear of the surface
    assert np.allclose(a, 0.0)


def test_repulsive_accel_pushes_away_from_obstacle_center():
    ctrl = PotentialFieldController(influence_margin=1.0)
    obstacles = [CircleObstacle(x=5.0, z=5.0, radius=0.8)]
    pos = np.array([5.0, 5.0 + 0.8 + 0.3])  # 0.3m from the surface, inside the influence margin
    a = ctrl._repulsive_accel(pos, obstacles)
    assert a[1] > 0  # pushed in +z, directly away from the obstacle center
    assert np.isclose(a[0], 0.0, atol=1e-9)


@pytest.mark.parametrize("goal_dir,expected_theta_sign", [
    ((1.0, 0.0), -1),  # desired thrust toward +x needs a negative tilt
    ((-1.0, 0.0), 1),  # desired thrust toward -x needs a positive tilt
])
def test_force_to_attitude_inversion_sign_convention(goal_dir, expected_theta_sign):
    """A pure horizontal desired force should tilt the drone in the
    direction that the dynamics (fx = -total_thrust*sin(theta)) actually
    require -- see QuadrotorEnv's module docstring on sign conventions."""
    ctrl = PotentialFieldController(k_att=10.0, max_att_accel=10.0, k_vel_damp=0.0)
    goal = np.array(goal_dir) * 5.0
    action = ctrl.act(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, goal, obstacles=[])
    # a rightward (+x) push means left thrust > right thrust in this sign
    # convention (see QuadrotorEnv module docstring), i.e. theta becomes negative.
    if expected_theta_sign < 0:
        assert action[0] > action[1]
    else:
        assert action[1] > action[0]


def test_controller_reaches_goal_with_no_obstacles():
    ctrl = PotentialFieldController()
    env = QuadrotorEnv(seed=0)
    env.reset_to(state=np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0]), goal=np.array([7.0, 6.0]))
    done, info = False, {"outcome": "running"}
    steps = 0
    while not done and steps < QuadrotorEnv.MAX_STEPS:
        x, z, vx, vz, theta, omega = env.state
        action = ctrl.act(x, z, vx, vz, theta, omega, env.goal, env.obstacles)
        _, _, done, info = env.step(action)
        steps += 1
    assert info["outcome"] == "success"


def test_controller_routes_around_a_single_obstacle():
    # start/obstacle/goal deliberately NOT collinear -- a perfectly symmetric
    # approach is the one well-known potential-field degenerate case (the
    # repulsive force has zero lateral component, so the drone can fly
    # straight into an exact-center obstacle; see PHASE6_BASELINE_COMPARISON_RESULTS.md).
    ctrl = PotentialFieldController()
    obstacles = [CircleObstacle(x=4.5, z=4.0, radius=0.8)]
    env = QuadrotorEnv(obstacles=obstacles, seed=0)
    env.reset_to(state=np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]), goal=np.array([8.0, 2.5]))
    done, info = False, {"outcome": "running"}
    steps = 0
    while not done and steps < QuadrotorEnv.MAX_STEPS:
        x, z, vx, vz, theta, omega = env.state
        action = ctrl.act(x, z, vx, vz, theta, omega, env.goal, env.obstacles)
        _, _, done, info = env.step(action)
        steps += 1
    assert info["outcome"] == "success"
