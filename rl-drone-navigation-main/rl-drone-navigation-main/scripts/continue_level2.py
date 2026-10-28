"""
Continue Level 2 training (7-term LOS-aware reward) from an existing
checkpoint for one block of updates, then report -- built to run the
"is it undertraining?" test: does the DETERMINISTIC collision rate trend
down from 34% toward the ~15% baseline as training continues?

The block-end metric is a 100-episode deterministic eval on a fresh
seed-1000 env (draws 1..100), IDENTICAL to scripts/eval_outcome_breakdown,
so the collision % is directly comparable across blocks. 20-ep pulses give
cheap intra-block texture only. entropy_coef default 0.003.

Reuses the curriculum's Level 2 config and shared components. No reward/env
changes. Per CLAUDE.md: no plt.show(); CPU only.
"""

import argparse
import os
import time
from collections import Counter

import numpy as np
import torch

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer
from training.utils import run_evaluation
from scripts.train_drone_curriculum import (
    CURRICULUM, ROLLOUT_STEPS, SEED, SUCCESS_THRESHOLD, MOVING_AVG_WINDOW,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def deterministic_eval(agent, env, n):
    """Success rate, avg reward, and full outcome breakdown over n
    deterministic episodes (the clean measure -- no exploration noise)."""
    succ, tot_r, outcomes = 0, 0.0, []
    for _ in range(n):
        obs = env.reset()
        done, r, info = False, 0.0, {"outcome": "running"}
        while not done:
            obs, rew, done, info = env.step(agent.act_deterministic(obs))
            r += rew
        outcomes.append(info["outcome"])
        succ += int(info["outcome"] == "success")
        tot_r += r
    return succ / n, tot_r / n, Counter(outcomes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--offset", type=int, required=True, help="cumulative update count of the resume checkpoint")
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--name", default="phase4_level2_v2")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    level2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")

    print("=" * 76)
    print(f"CONTINUE LEVEL 2 -- updates {args.offset + 1}..{args.offset + args.updates} "
          f"(from {os.path.basename(args.resume_checkpoint)})")
    print(f"  entropy_coef={args.entropy_coef}; key metric = deterministic collision % "
          f"(100-ep, seed {SEED + 1000})")
    print("=" * 76)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=args.entropy_coef),
    )
    agent.load(args.resume_checkpoint)

    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED)
    pulse_env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED + 2000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards, episode_outcomes, entropy_trend, pulses = [], [], [], []
    current = 0.0
    obs = env.reset()
    t0 = time.time()

    for u in range(1, args.updates + 1):
        cumulative = args.offset + u
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=done, value=value)
            current += reward
            obs = next_obs
            if done:
                episode_rewards.append(current)
                episode_outcomes.append(info["outcome"])
                current = 0.0
                obs = env.reset()

        diag = agent.update(buffer, agent.get_value(obs))
        entropy_trend.append((cumulative, diag["entropy"]))
        recent = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_sr = np.mean([o == "success" for o in recent]) if recent else 0.0
        print(f"  update {cumulative:3d}  episodes:{len(episode_rewards):5d}  "
              f"train_pulse:{train_sr:5.1%}  entropy:{diag['entropy']:+.3f}")

        if u % 10 == 0:
            sr, ar = run_evaluation(agent, pulse_env, 20)
            pulses.append((cumulative, sr, ar))
            print(f"      [pulse 20-ep @ {cumulative}] success:{sr:5.1%}")

    agent.save(os.path.join(CHECKPOINT_DIR, f"{args.name}_update{args.offset + args.updates:04d}.pt"))
    elapsed = time.time() - t0

    # ---- block-end 100-ep DETERMINISTIC decision eval + breakdown ----
    dec_env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED + 1000)
    sr, ar, counts = deterministic_eval(agent, dec_env, 100)
    n = 100

    line = "=" * 76
    print("\n" + line)
    print(f"REPORT -- Level 2 at cumulative update {args.offset + args.updates}")
    print(line)
    print(f"\n100-EP DETERMINISTIC DECISION EVAL (seed {SEED + 1000}):")
    print(f"  success rate: {sr:.1%}   avg reward: {ar:.2f}")
    print(f"\nKEY METRIC -- deterministic outcome breakdown:")
    for k in OUTCOME_KEYS:
        c = counts.get(k, 0)
        flag = "   <== KEY" if k == "collision" else ""
        print(f"  {k:14s}: {c:3d}  ({c / n:5.1%}){flag}")

    print(f"\nENTROPY TREND:")
    u0, e0 = entropy_trend[0]
    u1, e1 = entropy_trend[-1]
    print(f"  start (u{u0}): {e0:+.3f}   end (u{u1}): {e1:+.3f}   delta: {e1 - e0:+.3f}")

    print(f"\nTRAINING OUTCOME MIX (rollouts, exploration-inflated):")
    tot = len(episode_outcomes)
    for k in OUTCOME_KEYS:
        c = episode_outcomes.count(k)
        print(f"  {k:14s}: {c:5d}  ({(c / tot if tot else 0):5.1%})")

    if pulses:
        print(f"\n20-ep pulses: " + "  ".join(f"u{u}:{s:.0%}" for u, s, _ in pulses))
    print(f"\nwall time: {elapsed:.1f}s   checkpoint: {args.name}_update{args.offset + args.updates:04d}.pt")
    print(line)


if __name__ == "__main__":
    main()
