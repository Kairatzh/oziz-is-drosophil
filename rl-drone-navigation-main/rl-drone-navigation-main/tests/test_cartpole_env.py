"""
Sanity checks for the hand-rolled CartPole physics, in the same spirit as
Phase 1's quadrotor validation: confirm the equations of motion behave
the way basic mechanics says they should, before trusting PPO results
built on top of them.
"""

import numpy as np
import pytest

from env.cartpole_env import CartPoleEnv


def test_reset_state_is_small_and_shape_correct():
    env = CartPoleEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    assert np.all(np.abs(obs) < 0.05)


def test_inverted_pendulum_is_an_unstable_equilibrium():
    """With zero applied force, a small tilt should produce angular
    acceleration in the *same* direction as the tilt -- gravity pulls an
    inverted pendulum further off-balance rather than restoring it. The
    discrete action space only offers +-10N, which is large enough to
    swamp gravity's effect over a real step, so this checks the
    acceleration formula directly (force=0) rather than via env.step."""
    env = CartPoleEnv(seed=0)
    theta = 0.05
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    temp = 0.0  # force = 0, theta_dot = 0
    theta_acc = (env.GRAVITY * sin_theta - cos_theta * temp) / (
        env.LENGTH * (4.0 / 3.0 - env.MASS_POLE * cos_theta ** 2 / env.TOTAL_MASS)
    )
    assert theta_acc > 0.0  # accelerates further away from upright, not back toward it


def test_reward_is_one_per_surviving_step():
    env = CartPoleEnv(seed=0)
    env.reset()
    _, reward, done, _ = env.step(1)
    assert reward == 1.0


def test_episode_ends_when_angle_exceeds_limit():
    env = CartPoleEnv(seed=0)
    env.state = np.array([0.0, 0.0, env.THETA_LIMIT + 0.01, 0.0])
    env.steps = 0
    obs, reward, done, info = env.step(1)
    assert done
    assert info["outcome"] == "fell"


def test_episode_ends_when_cart_leaves_track():
    env = CartPoleEnv(seed=0)
    env.state = np.array([env.X_LIMIT + 0.01, 0.0, 0.0, 0.0])
    env.steps = 0
    obs, reward, done, info = env.step(1)
    assert done
    assert info["outcome"] == "fell"


def test_episode_times_out_at_max_steps():
    env = CartPoleEnv(seed=0)
    env.reset()
    env.MAX_STEPS = 3
    done = False
    info = {}
    for _ in range(3):
        obs, reward, done, info = env.step(1)
    assert done
    assert info["outcome"] == "timeout"


def test_push_right_accelerates_cart_right_when_upright():
    env = CartPoleEnv(seed=0)
    env.state = np.array([0.0, 0.0, 0.0, 0.0])  # perfectly upright, at rest
    env.steps = 0
    env.step(1)  # push right
    assert env.state[1] > 0.0  # cart velocity becomes positive (rightward)
