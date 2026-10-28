"""
Tests for the Phase 1 physics environment.

These tests are the "sanity checks" described in PROJECT_SPEC.md: before
any learning happens, we need hard evidence that the physics itself is
correct. If the physics is wrong, no amount of RL training will produce
sensible flight -- the agent would just be learning to exploit a broken
simulator.
"""

import numpy as np
import pytest

from env.quadrotor_env import CircleObstacle, QuadrotorEnv, RectObstacle


def make_env(obstacles=None):
    return QuadrotorEnv(obstacles=obstacles, seed=0)


# ---------------------------------------------------------------------------
# Obstacle geometry
# ---------------------------------------------------------------------------

def test_circle_obstacle_distance():
    obs = CircleObstacle(x=5.0, z=5.0, radius=1.0)
    assert obs.distance_to_point(5.0, 5.0) == pytest.approx(-1.0)   # center: inside
    assert obs.distance_to_point(6.0, 5.0) == pytest.approx(0.0)    # exactly on surface
    assert obs.distance_to_point(8.0, 5.0) == pytest.approx(2.0)    # outside


def test_rect_obstacle_distance():
    obs = RectObstacle(x=5.0, z=5.0, width=2.0, height=2.0)  # spans x:[4,6], z:[4,6]
    assert obs.distance_to_point(5.0, 5.0) == pytest.approx(0.0)    # inside -> 0
    assert obs.distance_to_point(6.0, 5.0) == pytest.approx(0.0)    # on edge
    assert obs.distance_to_point(7.0, 5.0) == pytest.approx(1.0)    # 1m past the right edge
    assert obs.distance_to_point(7.0, 8.0) == pytest.approx(np.hypot(1.0, 2.0))  # outside a corner


# ---------------------------------------------------------------------------
# Hover sanity check: constant, symmetric thrust should hold altitude
# ---------------------------------------------------------------------------

def test_hover_maintains_altitude_and_levelness():
    env = make_env()
    start_z = 4.0
    # Goal is placed far away so the "reached goal" termination can't
    # trigger just because the drone starts at rest -- this test only
    # cares about the physics, not the episode/goal logic.
    env.reset_to(state=[5.0, start_z, 0.0, 0.0, 0.0, 0.0], goal=[0.2, 0.2])

    hover_action = [env.HOVER_ACTION, env.HOVER_ACTION]
    for _ in range(150):  # 3 seconds of simulated flight
        obs, reward, done, info = env.step(hover_action)
        assert not done

    x, z, vx, vz, theta, omega = env.state
    assert z == pytest.approx(start_z, abs=0.05)
    assert x == pytest.approx(5.0, abs=0.05)
    assert vz == pytest.approx(0.0, abs=0.05)
    assert theta == pytest.approx(0.0, abs=1e-6)
    assert omega == pytest.approx(0.0, abs=1e-6)


def test_zero_thrust_falls_under_gravity():
    env = make_env()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[5.0, 4.0])
    env.step([0.0, 0.0])
    _, z, _, vz, _, _ = env.state
    assert vz < 0.0        # falling
    assert z < 4.0          # descended


# ---------------------------------------------------------------------------
# Differential thrust: verifies rotation direction and the resulting
# (initially counterintuitive) direction of lateral travel.
# ---------------------------------------------------------------------------

def test_left_greater_than_right_rotates_and_moves_right():
    env = make_env()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[5.0, 4.0])

    action = [env.HOVER_ACTION + 0.15, env.HOVER_ACTION - 0.15]  # left > right
    for _ in range(10):
        env.step(action)
    x, z, vx, vz, theta, omega = env.state

    assert omega < 0.0     # torque = (right-left)*L < 0 -> clockwise spin-up
    assert theta < 0.0     # body has rotated clockwise (negative)
    assert vx > 0.0        # net thrust vector tilts toward +x -> moves right


def test_right_greater_than_left_rotates_and_moves_left():
    env = make_env()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[5.0, 4.0])

    action = [env.HOVER_ACTION - 0.15, env.HOVER_ACTION + 0.15]  # right > left
    for _ in range(10):
        env.step(action)
    x, z, vx, vz, theta, omega = env.state

    assert omega > 0.0
    assert theta > 0.0
    assert vx < 0.0


def test_torque_independent_of_current_orientation():
    """Torque = (right - left) * arm_length should not depend on theta,
    since both rotor forces are always perpendicular to the rod."""
    env = make_env()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.7, 0.0], goal=[5.0, 4.0])  # start tilted
    action = [env.HOVER_ACTION - 0.1, env.HOVER_ACTION + 0.1]
    env.step(action)
    expected_torque = (action[1] - action[0]) * env.MAX_THRUST_PER_ROTOR * env.ARM_LENGTH
    expected_alpha = expected_torque / env.INERTIA
    assert env.state[5] == pytest.approx(expected_alpha * env.DT, abs=1e-9)


# ---------------------------------------------------------------------------
# Episode termination and reward
# ---------------------------------------------------------------------------

def test_reaching_goal_slowly_gives_success_and_bonus():
    env = make_env()
    # Start essentially at the goal, at rest -> should immediately succeed.
    env.reset_to(state=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0], goal=[5.0, 5.0])
    obs, reward, done, info = env.step([env.HOVER_ACTION, env.HOVER_ACTION])
    assert done
    assert info["outcome"] == "success"
    assert reward > env.R_GOAL - 1.0  # dominated by the terminal bonus


def test_reaching_goal_too_fast_does_not_count_as_success():
    env = make_env()
    env.reset_to(state=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], goal=[5.05, 5.0])  # near goal, fast
    obs, reward, done, info = env.step([env.HOVER_ACTION, env.HOVER_ACTION])
    assert info["outcome"] != "success"


def test_out_of_bounds_terminates_with_crash_penalty():
    env = make_env()
    env.reset_to(state=[5.0, 0.5, 0.0, 0.0, 0.0, 0.0], goal=[5.0, 7.5])
    done = False
    info = {}
    reward = 0.0
    for _ in range(100):  # free-fall through the floor, zero thrust
        obs, reward, done, info = env.step([0.0, 0.0])
        if done:
            break
    assert done
    assert info["outcome"] == "out_of_bounds"
    assert reward < 0


def test_collision_with_obstacle_terminates_with_crash_penalty():
    obstacle = CircleObstacle(x=5.2, z=4.0, radius=0.3)
    env = make_env(obstacles=[obstacle])
    env.reset_to(state=[5.0, 4.0, 2.0, 0.0, 0.0, 0.0], goal=[8.0, 4.0])  # flying straight at it
    done = False
    info = {}
    for _ in range(20):
        obs, reward, done, info = env.step([env.HOVER_ACTION, env.HOVER_ACTION])
        if done:
            break
    assert done
    assert info["outcome"] == "collision"


def test_timeout_terminates_without_bonus_or_penalty():
    env = make_env()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[9.9, 7.9])  # far away
    env.MAX_STEPS = 5  # shrink for a fast test
    done = False
    info = {}
    for _ in range(5):
        obs, reward, done, info = env.step([env.HOVER_ACTION, env.HOVER_ACTION])
    assert done
    assert info["outcome"] == "timeout"


# ---------------------------------------------------------------------------
# Obstacle-clearance shaping (Phase 4)
# ---------------------------------------------------------------------------
#
# The clearance term is the only reward component that depends on the
# obstacle layout for a NON-terminal step, so we isolate it by diffing the
# reward of an env with a nearby obstacle against an otherwise identical
# obstacle-free env at the same state/goal/action. Obstacles do not affect
# the physics, so both envs step to the exact same state and every other
# reward term cancels in the difference.

def _clearance_penalty(state, goal, action, obstacle):
    """(reward_free - reward_with_obstacle) at one step == the magnitude of
    the clearance penalty, provided the step is non-terminal (no crash)."""
    env_free = make_env(obstacles=[])
    env_free.reset_to(state=state, goal=goal)
    _, r_free, done_free, _ = env_free.step(action)

    env_obs = make_env(obstacles=[obstacle])
    env_obs.reset_to(state=state, goal=goal)
    _, r_obs, done_obs, info = env_obs.step(action)

    assert not done_free and not done_obs and info["outcome"] == "running"
    return r_free - r_obs


def test_clearance_penalty_zero_in_open_space():
    # Obstacle well beyond lidar range (all rays read 1.0 > threshold):
    # the clearance term must contribute exactly nothing.
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    far = CircleObstacle(x=1.0, z=7.0, radius=0.3)  # ~4.7 m away, past LIDAR_MAX_RANGE
    penalty = _clearance_penalty(state, [8.0, 4.0], [QuadrotorEnv.HOVER_ACTION] * 2, far)
    assert penalty == pytest.approx(0.0, abs=1e-9)


def test_clearance_penalty_negative_near_obstacle():
    # Obstacle ~0.2 m from the drone's surface (inside the 0.9 m band but
    # not colliding): the clearance term must reduce the reward, by exactly
    # W_CLEARANCE * (threshold - min_ray).
    env = make_env()
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    goal = [8.0, 4.0]
    near = CircleObstacle(x=5.5, z=4.0, radius=0.3)  # surface 0.2 m from (5,4); > DRONE_RADIUS, no crash
    penalty = _clearance_penalty(state, goal, [env.HOVER_ACTION] * 2, near)
    assert penalty > 0.0  # reward strictly reduced by the clearance term

    # ...and it matches the exact formula, using the post-step position the
    # two envs necessarily share (physics is obstacle-independent).
    env_obs = make_env(obstacles=[near])
    env_obs.reset_to(state=state, goal=goal)
    env_obs.step([env.HOVER_ACTION] * 2)
    min_ray = float(env_obs._lidar_scan(env_obs.state[0], env_obs.state[1]).min())
    expected = env.W_CLEARANCE * (env.LIDAR_CLEAR_THRESHOLD - min_ray)
    assert penalty == pytest.approx(expected)


def test_clearance_penalty_scales_with_proximity():
    # Closer obstacle -> larger penalty. Both obstacles are in-band and
    # non-colliding, at different distances straight ahead (+x) of the drone.
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    goal = [8.0, 4.0]
    action = [QuadrotorEnv.HOVER_ACTION] * 2
    closer = CircleObstacle(x=5.5, z=4.0, radius=0.3)   # surface ~0.2 m
    farther = CircleObstacle(x=5.9, z=4.0, radius=0.3)  # surface ~0.6 m
    pen_closer = _clearance_penalty(state, goal, action, closer)
    pen_farther = _clearance_penalty(state, goal, action, farther)
    assert pen_closer > pen_farther > 0.0


# ---------------------------------------------------------------------------
# Goal-proximity bonus (Phase 4)
# ---------------------------------------------------------------------------
#
# Isolated by evaluating the reward at exact hand-set states via
# _compute_reward_and_done (no physics integration): drone at rest, level,
# with _prev_dist == dist so the progress term is zero and the velocity term
# is zero. In open space every term except the distance penalty and the
# goal-proximity bonus is then constant across goal distance, so reward
# differences expose the bonus directly.

def _reward_at_distance(env, dist):
    """Reward with the drone at rest, `dist` metres from the goal, no obstacles."""
    env.state = np.array([5.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    env.goal = np.array([5.0 + dist, 4.0])
    env._prev_dist = dist  # -> progress term is exactly zero
    reward, done, info = env._compute_reward_and_done(env.HOVER_ACTION, env.HOVER_ACTION)
    return reward


def test_goal_proximity_bonus_zero_beyond_range():
    # Two distances both beyond GOAL_PROX_RANGE: the bonus is off for both,
    # so the reward difference must be exactly the linear distance-penalty
    # difference -- nothing extra from the proximity term.
    env = make_env()
    d1, d2 = env.GOAL_PROX_RANGE + 0.5, env.GOAL_PROX_RANGE + 1.5
    diff = _reward_at_distance(env, d1) - _reward_at_distance(env, d2)
    assert diff == pytest.approx(env.W_DIST * (d2 - d1))


def test_goal_proximity_bonus_grows_as_drone_nears_goal():
    # Inside the range, total reward strictly increases as the drone gets
    # closer -- the proximity bonus dominates the (tiny) distance penalty.
    env = make_env()
    dists = [1.4, 1.0, 0.6, 0.4]
    rewards = [_reward_at_distance(env, d) for d in dists]
    assert all(rewards[i] < rewards[i + 1] for i in range(len(rewards) - 1))


def test_goal_proximity_bonus_matches_formula_and_is_positive():
    # Isolate the bonus at the 0.39 m stall distance by differencing against a
    # point just beyond the range (bonus == 0 there), correcting for the
    # linear distance penalty that also varies with distance.
    env = make_env()
    d = 0.39
    d_ref = env.GOAL_PROX_RANGE + 0.5
    isolated_bonus = (_reward_at_distance(env, d) - _reward_at_distance(env, d_ref)) - env.W_DIST * (d_ref - d)
    expected = env.W_GOAL_PROX * (env.GOAL_PROX_RANGE - d)
    assert isolated_bonus == pytest.approx(expected)
    assert isolated_bonus > 0.0


def _reward_at_state(env, state, goal):
    """Exact reward at a hand-set state (no physics step), at rest with
    _prev_dist == dist so progress and velocity terms are zero."""
    env.state = np.array(state, dtype=float)
    env.goal = np.array(goal, dtype=float)
    env._prev_dist = env._distance_to_goal()
    reward, done, info = env._compute_reward_and_done(env.HOVER_ACTION, env.HOVER_ACTION)
    return reward


def test_goal_proximity_bonus_full_strength_when_obstacle_outside_band():
    # Obstacle present but well beyond the clearance band (min_ray >= 0.3):
    # prox_factor == 1, so the reward is identical to the obstacle-free case.
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    goal = [5.0, 4.6]  # 0.6 m away, within GOAL_PROX_RANGE
    far_obstacle = CircleObstacle(x=5.0, z=1.0, radius=0.3)  # ~2.7 m below -> min_ray ~0.9
    r_far = _reward_at_state(make_env(obstacles=[far_obstacle]), state, goal)
    r_free = _reward_at_state(make_env(obstacles=[]), state, goal)
    assert r_far == pytest.approx(r_free)


def test_goal_proximity_bonus_suppressed_near_obstacle():
    # Obstacle ~0.2 m from the drone (min_ray < 0.3): the goal-proximity
    # bonus must be scaled by prox_factor = (min_ray/threshold)**2, strongly
    # suppressing it so the clearance penalty dominates.
    env = make_env()
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    goal = [5.0, 4.6]  # 0.6 m away, within range; not toward the obstacle
    near = CircleObstacle(x=5.5, z=4.0, radius=0.3)  # surface 0.2 m from (5,4); no crash
    env_obs = make_env(obstacles=[near])

    r_obs = _reward_at_state(env_obs, state, goal)
    r_free = _reward_at_state(make_env(obstacles=[]), state, goal)

    min_ray = float(env_obs._lidar_scan(5.0, 4.0).min())
    assert min_ray < env.LIDAR_CLEAR_THRESHOLD  # sanity: obstacle is inside the band
    factor = min(1.0, min_ray / env.LIDAR_CLEAR_THRESHOLD) ** 2
    full_bonus = env.W_GOAL_PROX * (env.GOAL_PROX_RANGE - 0.6)
    suppressed_bonus = full_bonus * factor
    clearance_pen = env.W_CLEARANCE * (env.LIDAR_CLEAR_THRESHOLD - min_ray)

    # r_obs vs r_free differ only by (suppressed - full) goal-prox and the
    # added clearance penalty; everything else is identical.
    assert (r_obs - r_free) == pytest.approx((suppressed_bonus - full_bonus) - clearance_pen)
    assert factor < 0.1                          # bonus strongly suppressed
    assert suppressed_bonus < clearance_pen      # clearance dominates near the obstacle


def test_goal_los_clearance_detects_occlusion():
    # Obstacle squarely between drone (5,4) and goal (7,4): clearance == -radius.
    env = make_env(obstacles=[CircleObstacle(x=6.0, z=4.0, radius=0.5)])
    assert env._goal_los_clearance(5.0, 4.0, np.array([7.0, 4.0])) == pytest.approx(-0.5)
    # Same obstacle, but the goal is off to the side (line of sight clear).
    assert env._goal_los_clearance(5.0, 4.0, np.array([5.0, 6.0])) == float("inf")
    # Obstacle behind the drone (opposite the goal direction) does not count.
    env_behind = make_env(obstacles=[CircleObstacle(x=3.0, z=4.0, radius=0.5)])
    assert env_behind._goal_los_clearance(5.0, 4.0, np.array([7.0, 4.0])) == float("inf")


def test_goal_proximity_bonus_suppressed_when_occluded_at_moderate_range():
    # A small obstacle on the line to the goal, positioned so the drone's
    # min_ray stays at/above the clearance threshold (proximity alone would
    # NOT suppress the bonus), but the straight line is blocked -> the
    # line-of-sight factor suppresses it. This is the case the proximity-only
    # version missed.
    obstacle = CircleObstacle(x=6.3, z=4.0, radius=0.3)  # ~1.0 m ahead of the drone
    env = make_env(obstacles=[obstacle])
    state = [5.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    goal = [6.45, 4.0]  # 1.45 m away, directly behind the obstacle, within range

    min_ray = float(env._lidar_scan(5.0, 4.0).min())
    assert min_ray >= env.LIDAR_CLEAR_THRESHOLD          # proximity alone would NOT suppress
    assert env._goal_los_clearance(5.0, 4.0, np.array(goal)) < 0.0  # line of sight blocked

    r_obs = _reward_at_state(env, state, goal)
    r_free = _reward_at_state(make_env(obstacles=[]), state, goal)
    # min_ray >= threshold means no clearance penalty in either env, so the ONLY
    # difference is the goal-proximity bonus: full in open space, ~0 when occluded.
    full_bonus = env.W_GOAL_PROX * (env.GOAL_PROX_RANGE - 1.45)
    assert (r_free - r_obs) == pytest.approx(full_bonus)


# ---------------------------------------------------------------------------
# Level 1.5 curriculum layout (one obstacle, guaranteed clear line of sight)
# ---------------------------------------------------------------------------

def test_curriculum_has_level1_5_between_levels_1_and_2():
    from scripts.train_drone_curriculum import CURRICULUM
    names = [c["name"] for c in CURRICULUM]
    assert names[:3] == [
        "level1_no_obstacles", "level1_5_single_clear_obstacle", "level2_wide_gaps",
    ]


def test_level1_5_obstacle_keeps_line_of_sight_clear():
    # Over many resets the single obstacle must always be radius 0.8, inside
    # the world, clear of the start->goal line of sight, and not overlapping
    # the start or goal. It teaches "obstacles exist" without ever blocking the
    # straight path.
    from scripts.train_drone_curriculum import make_level15_obstacle_fn
    env = QuadrotorEnv(obstacle_fn=make_level15_obstacle_fn(), seed=0)
    placed = 0
    for _ in range(300):
        env.reset()
        assert len(env.obstacles) <= 1
        if not env.obstacles:
            continue
        placed += 1
        o = env.obstacles[0]
        assert o.radius == pytest.approx(0.8)
        assert 0.8 <= o.x <= QuadrotorEnv.WORLD_WIDTH - 0.8
        assert 0.8 <= o.z <= QuadrotorEnv.WORLD_HEIGHT - 0.8
        # straight start->goal line of sight is clear (not occluded)
        assert env._goal_los_clearance(env.state[0], env.state[1], env.goal) > 0.0
        # obstacle does not swallow the start or the goal
        assert o.distance_to_point(env.state[0], env.state[1]) > QuadrotorEnv.DRONE_RADIUS
        assert o.distance_to_point(env.goal[0], env.goal[1]) > QuadrotorEnv.GOAL_RADIUS
    assert placed >= 285  # a clear spot is found in essentially every episode


def test_progress_reward_is_potential_based_and_telescopes():
    """Sum of progress-only rewards over an episode should equal
    (start_distance - end_distance), regardless of the path taken."""
    env = make_env()
    env.reset_to(state=[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], goal=[1.0, 1.0])
    start_dist = env._distance_to_goal()

    total_progress = 0.0
    action = [env.HOVER_ACTION + 0.05, env.HOVER_ACTION - 0.05]
    for _ in range(30):
        dist_before = env._distance_to_goal()
        env.step(action)
        dist_after = env._distance_to_goal()
        total_progress += dist_before - dist_after

    end_dist = env._distance_to_goal()
    assert total_progress == pytest.approx(start_dist - end_dist, abs=1e-9)


# ---------------------------------------------------------------------------
# Observation / reset sanity
# ---------------------------------------------------------------------------

def test_reset_avoids_obstacles():
    obstacle = CircleObstacle(x=5.0, z=4.0, radius=1.5)
    env = QuadrotorEnv(obstacles=[obstacle], seed=1)
    for _ in range(20):
        obs = env.reset()
        x, z = env.state[0], env.state[1]
        assert obstacle.distance_to_point(x, z) > env.DRONE_RADIUS
        gx, gz = env.goal
        assert obstacle.distance_to_point(gx, gz) > env.GOAL_RADIUS


def test_observation_shape_and_goal_offset():
    env = make_env()
    obs = env.reset_to(state=[2.0, 3.0, 0.1, 0.2, 0.0, 0.0], goal=[5.0, 7.0])
    assert obs.shape == (env.OBS_DIM,)
    x, z, vx, vz, theta, omega, dx, dz = obs[:8]
    assert dx == pytest.approx(3.0)
    assert dz == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Lidar-style obstacle sensing (added after Phase 4 failure analysis showed
# the observation previously carried zero obstacle information)
# ---------------------------------------------------------------------------

def test_circle_ray_intersection_distance_matches_geometry():
    obs = CircleObstacle(x=5.0, z=0.0, radius=1.0)
    # Ray straight along +x from the origin should hit the near edge at x=4.
    dist = obs.ray_intersection_distance(ox=0.0, oz=0.0, dx=1.0, dz=0.0)
    assert dist == pytest.approx(4.0)


def test_circle_ray_intersection_returns_none_when_ray_misses():
    obs = CircleObstacle(x=5.0, z=5.0, radius=1.0)
    dist = obs.ray_intersection_distance(ox=0.0, oz=0.0, dx=1.0, dz=0.0)  # passes well below the circle
    assert dist is None


def test_circle_ray_intersection_returns_none_when_circle_is_behind():
    obs = CircleObstacle(x=-5.0, z=0.0, radius=1.0)
    dist = obs.ray_intersection_distance(ox=0.0, oz=0.0, dx=1.0, dz=0.0)
    assert dist is None


def test_rect_ray_intersection_distance_matches_geometry():
    obs = RectObstacle(x=5.0, z=0.0, width=2.0, height=2.0)  # spans x:[4,6], z:[-1,1]
    dist = obs.ray_intersection_distance(ox=0.0, oz=0.0, dx=1.0, dz=0.0)
    assert dist == pytest.approx(4.0)


def test_rect_ray_intersection_returns_none_when_ray_misses():
    obs = RectObstacle(x=5.0, z=5.0, width=2.0, height=2.0)
    dist = obs.ray_intersection_distance(ox=0.0, oz=0.0, dx=1.0, dz=0.0)
    assert dist is None


def test_lidar_scan_shape_and_normalized_range():
    env = make_env(obstacles=[CircleObstacle(x=6.0, z=4.0, radius=1.0)])
    readings = env._lidar_scan(x=4.0, z=4.0)
    assert readings.shape == (env.LIDAR_NUM_RAYS,)
    assert np.all(readings >= 0.0) and np.all(readings <= 1.0)


def test_lidar_returns_max_range_when_no_obstacle_nearby():
    env = make_env(obstacles=[])
    readings = env._lidar_scan(x=5.0, z=4.0)
    np.testing.assert_allclose(readings, 1.0)  # nothing in range -> every ray reads "clear"


def test_lidar_distance_decreases_as_drone_approaches_obstacle():
    """The specific regression test for the Phase 4 root-cause fix: as
    the drone gets closer to an obstacle sitting on one of the ray
    directions, that ray's normalized reading must shrink monotonically
    toward 0, giving the policy an actual signal to react to."""
    obstacle = CircleObstacle(x=8.0, z=4.0, radius=0.5)
    env = make_env(obstacles=[obstacle])

    # Ray index 0 points along world +x (angle 0), so approaching the
    # obstacle by increasing x at fixed z=4.0 walks straight down that ray.
    # All positions chosen within LIDAR_MAX_RANGE (3m) of the obstacle's
    # edge (x=7.5) so every reading reflects a real detection, not the
    # "nothing in range" ceiling of 1.0.
    ray_index = 0
    positions_x = [5.0, 5.5, 6.0, 6.5, 7.0]  # each closer to the obstacle at x=8.0
    readings = [env._lidar_scan(x=px, z=4.0)[ray_index] for px in positions_x]

    for earlier, later in zip(readings, readings[1:]):
        assert later < earlier, f"lidar reading should shrink as drone approaches: {readings}"


def test_lidar_hits_nearest_of_multiple_obstacles():
    near = CircleObstacle(x=6.0, z=4.0, radius=0.5)
    far = CircleObstacle(x=9.0, z=4.0, radius=0.5)
    env = make_env(obstacles=[far, near])  # deliberately listed far-first
    readings = env._lidar_scan(x=4.0, z=4.0)
    expected_normalized = (near.x - near.radius - 4.0) / env.LIDAR_MAX_RANGE
    assert readings[0] == pytest.approx(expected_normalized, abs=1e-4)


def test_get_obs_includes_16_values_with_lidar_segment_matching_direct_scan():
    obstacle = CircleObstacle(x=8.0, z=4.0, radius=0.5)
    env = make_env(obstacles=[obstacle])
    obs = env.reset_to(state=[4.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[1.0, 1.0])
    assert obs.shape == (16,)
    expected_lidar = env._lidar_scan(4.0, 4.0)
    np.testing.assert_allclose(obs[8:], expected_lidar)
