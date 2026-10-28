"""
Read-only: deterministic outcome breakdown (success/collision/out_of_bounds/
timeout) for a given checkpoint on Level 2, over N eval episodes. This is the
CLEAN measure of what the learned policy does -- unlike the training-rollout
breakdown, which is inflated by exploration noise. Does not train.

Usage: python -m scripts.eval_outcome_breakdown <checkpoint.pt> [n_episodes]
"""

import os
import sys
from collections import Counter

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM

EVAL_SEED = 1000


def breakdown(ckpt_path, n_episodes=100):
    level2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(ckpt_path)

    outcomes = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        info = {"outcome": "running"}
        while not done:
            obs, _, done, info = env.step(agent.act_deterministic(obs))
        outcomes.append(info["outcome"])

    counts = Counter(outcomes)
    print(f"deterministic {n_episodes}-ep breakdown for {os.path.basename(ckpt_path)} (seed {EVAL_SEED}):")
    for k in ("success", "collision", "out_of_bounds", "timeout"):
        c = counts.get(k, 0)
        print(f"  {k:14s}: {c:3d}  ({c / n_episodes:5.1%})")


if __name__ == "__main__":
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    breakdown(ckpt, n)
