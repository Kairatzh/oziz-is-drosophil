"""
Hand-rolled CartPole environment, used only to validate the from-scratch
PPO implementation before connecting it to the (much harder) quadrotor
environment in Phase 3.

This is the classic cart-pole benchmark: a pole is hinged to a cart that
can move left/right along a frictionless track, and the only control is
a fixed-magnitude push left or right. The physics equations below are the
standard ones from Barto, Sutton & Anderson (1983), also used by OpenAI
Gym's CartPole-v1 -- they are well-documented and easy to sanity-check
independently, which is exactly why this is a good "does my PPO even
work" test case: if the environment is right (small, well-understood,
easy to solve) and PPO still can't learn, the bug is in PPO.

State: [x, x_dot, theta, theta_dot]
  x        - cart position (m)
  x_dot    - cart velocity (m/s)
  theta    - pole angle from vertical (rad), 0 = balanced upright
  theta_dot - pole angular velocity (rad/s)

Action: discrete, 0 = push cart left, 1 = push cart right (fixed force
magnitude either way).

Reward: +1 for every timestep the pole remains balanced -- this is a
strictly sparser, simpler reward than the drone's, on purpose.
"""

from typing import Optional, Tuple

import numpy as np


class CartPoleEnv:
    GRAVITY = 9.8
    MASS_CART = 1.0
    MASS_POLE = 0.1
    TOTAL_MASS = MASS_CART + MASS_POLE
    LENGTH = 0.5                                  # half the pole's length (m)
    POLE_MASS_LENGTH = MASS_POLE * LENGTH
    FORCE_MAG = 10.0
    DT = 0.02

    X_LIMIT = 2.4                                 # track edges (m)
    THETA_LIMIT = 12 * np.pi / 180                # ~0.209 rad (12 degrees)
    MAX_STEPS = 500

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        # Small random perturbation near the balanced-upright state,
        # matching the classic benchmark's reset distribution.
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert action in (0, 1), f"action must be 0 or 1, got {action}"
        x, x_dot, theta, theta_dot = self.state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Standard cart-pole equations of motion (Barto, Sutton & Anderson 1983).
        temp = (force + self.POLE_MASS_LENGTH * theta_dot ** 2 * sin_theta) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASS_POLE * cos_theta ** 2 / self.TOTAL_MASS)
        )
        x_acc = temp - self.POLE_MASS_LENGTH * theta_acc * cos_theta / self.TOTAL_MASS

        # Semi-implicit Euler, consistent with env/quadrotor_env.py.
        x_dot += self.DT * x_acc
        x += self.DT * x_dot
        theta_dot += self.DT * theta_acc
        theta += self.DT * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(
            abs(x) > self.X_LIMIT
            or abs(theta) > self.THETA_LIMIT
            or self.steps >= self.MAX_STEPS
        )
        reward = 1.0
        info = {"outcome": "timeout" if self.steps >= self.MAX_STEPS and not done else "running"}
        if done and self.steps < self.MAX_STEPS:
            info["outcome"] = "fell"
        elif done:
            info["outcome"] = "timeout"

        return self.state.astype(np.float32), reward, done, info
