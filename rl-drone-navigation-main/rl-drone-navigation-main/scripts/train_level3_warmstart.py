"""
Level 3 (3-5 moderate-gap obstacles), warm-started from the accepted Level 2
checkpoint (phase4_level2_from15_update0050.pt: 61% success / 34% collision,
plateau diagnosed and accepted per the user's decision -- see memory
level2-plateau-diagnosis). Advancing per the standard curriculum rule: carry
weights + optimizer state forward, only the obstacle field changes.

Two-tier evaluation, per the project standard ([[eval-batch-size]]):
  - 100-episode deterministic eval = the PROGRESSION DECISION (>= 90% solves).
  - 20-episode deterministic eval = a cheap mid-training PULSE only.
Held-out seeds: decision uses SEED+1000, pulse uses SEED+2000.

Runs in fixed-size blocks (default 50 updates) and STOPS at the end of each
block to report and wait for confirmation, per the user's request to check in
between blocks rather than run straight through the 250-update safety cap.
Use --offset to resume with cumulative update numbering across blocks (same
convention as continue_level2.py) -- e.g. --resume-checkpoint
phase4_level3_from_level2_update0050.pt --offset 50 --updates 50 continues to
update 100. entropy_coef=0.003 by default.
Per CLAUDE.md: plt.show() is never called; CPU training only.
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
    MAX_UPDATES_PER_LEVEL,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")
DEFAULT_WARM_START = os.path.join(CHECKPOINT_DIR, "phase4_level2_from15_update0050.pt")
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
    p.add_argument("--resume-checkpoint", default=DEFAULT_WARM_START)
    p.add_argument("--updates", type=int, default=50)
    p.add_argument("--offset", type=int, default=0,
                   help="cumulative update count the resume checkpoint represents")
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--name", default="phase4_level3_from_level2")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    level3 = next(c for c in CURRICULUM if c["name"] == "level3_moderate_gaps")

    print("=" * 76)
    print(f"LEVEL 3 -- warm-started from {os.path.basename(args.resume_checkpoint)}")
    print(f"  cumulative updates {args.offset + 1}..{args.offset + args.updates} "
          f"(safety cap {MAX_UPDATES_PER_LEVEL} per level)")
    print(f"  entropy_coef={args.entropy_coef}")
    print(f"  decision eval = 100 ep (seed {SEED + 1000}); pulse = 20 ep (seed {SEED + 2000})")
    print("=" * 76)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=args.entropy_coef),
    )
    agent.load(args.resume_checkpoint)
    print(f"warm-started from {os.path.basename(args.resume_checkpoint)}\n")

    env = QuadrotorEnv(obstacles=level3["obstacles"], seed=SEED)
    decision_env = QuadrotorEnv(obstacles=level3["obstacles"], seed=SEED + 1000)
    pulse_env = QuadrotorEnv(obstacles=level3["obstacles"], seed=SEED + 2000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards, episode_outcomes = [], []
    entropy_trend = []      # (update, entropy)
    eval_history = []       # (update, kind, success_rate, avg_reward)
    reached_90_at = None
    current = 0.0
    obs = env.reset()
    start_time = time.time()

    for i in range(1, args.updates + 1):
        update = args.offset + i
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

        diagnostics = agent.update(buffer, agent.get_value(obs))
        entropy_trend.append((update, diagnostics["entropy"]))
        recent = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_sr = np.mean([o == "success" for o in recent]) if recent else 0.0
        recent_r = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        print(f"  update {update:3d}  episodes:{len(episode_rewards):5d}  "
              f"moving_avg_reward:{np.mean(recent_r):8.2f}  train_pulse:{train_sr:5.1%}  "
              f"entropy:{diagnostics['entropy']:+.3f}")

        if update % 10 == 0:
            esr, ear = run_evaluation(agent, decision_env, 100)
            eval_history.append((update, "decision-100", esr, ear))
            print(f"      [DECISION 100-ep @ update {update}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")
            if esr >= SUCCESS_THRESHOLD and reached_90_at is None:
                reached_90_at = update
                print(f"      *** reached progression threshold {SUCCESS_THRESHOLD:.0%} at update {update} "
                      f"(not advancing -- holding as instructed) ***")
        elif update % 5 == 0:
            esr, ear = run_evaluation(agent, pulse_env, 20)
            eval_history.append((update, "pulse-20", esr, ear))
            print(f"      [pulse 20-ep @ update {update}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")

        if update % 10 == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"{args.name}_update{update:04d}.pt"))

    elapsed = time.time() - start_time
    final_cumulative = args.offset + args.updates
    agent.save(os.path.join(CHECKPOINT_DIR, f"{args.name}_update{final_cumulative:04d}.pt"))

    # ---- block-end 100-ep DETERMINISTIC decision eval + full breakdown ----
    sr, ar, counts = deterministic_eval(agent, decision_env, 100)
    n = 100

    line = "=" * 76
    print("\n" + line)
    print(f"LEVEL 3 (warm-started) REPORT -- cumulative updates {args.offset + 1}..{final_cumulative}")
    print(line)

    print(f"\n100-EP DETERMINISTIC DECISION EVAL (seed {SEED + 1000}):")
    print(f"  success rate: {sr:.1%}   avg reward: {ar:.2f}")
    print(f"\nKEY METRIC -- deterministic outcome breakdown:")
    for k in OUTCOME_KEYS:
        c = counts.get(k, 0)
        flag = "   <== KEY" if k == "collision" else ""
        print(f"  {k:14s}: {c:3d}  ({c / n:5.1%}){flag}")

    print("\nEVAL HISTORY  (100-ep = decision, 20-ep = pulse only)")
    print(f"   {'update':>6}  {'eval type':>13}  {'success':>8}  {'avg_reward':>10}")
    for u, kind, esr, ear in eval_history:
        print(f"   {u:>6}  {kind:>13}  {esr:>7.1%}  {ear:>10.2f}")

    print("\nENTROPY TREND")
    u0, e0 = entropy_trend[0]
    u1, e1 = entropy_trend[-1]
    emin = min(entropy_trend, key=lambda t: t[1])
    emax = max(entropy_trend, key=lambda t: t[1])
    print(f"   start (u{u0}): {e0:+.3f}    end (u{u1}): {e1:+.3f}    delta: {e1 - e0:+.3f}")
    print(f"   min: {emin[1]:+.3f} (u{emin[0]})    max: {emax[1]:+.3f} (u{emax[0]})")

    print("\nTRAINING OUTCOME BREAKDOWN  (all episodes this block, exploration-inflated)")
    tot_counts = Counter(episode_outcomes)
    total = sum(tot_counts.values())
    for k in OUTCOME_KEYS:
        c = tot_counts.get(k, 0)
        print(f"   {k:>14}: {c:5d}  ({(c/total if total else 0):5.1%})")
    print(f"   {'TOTAL':>14}: {total:5d}")

    print("\n" + line)
    if reached_90_at is not None:
        print(f"STATUS: reached {SUCCESS_THRESHOLD:.0%} on a 100-ep eval at update {reached_90_at}; "
              f"held at {final_cumulative} without advancing (awaiting confirmation).")
    else:
        print(f"STATUS: did NOT reach {SUCCESS_THRESHOLD:.0%} within {final_cumulative} cumulative updates "
              f"(safety cap {MAX_UPDATES_PER_LEVEL}). Current 100-ep decision eval: {sr:.1%}.")
    print(f"wall time: {elapsed:.1f}s   latest checkpoint: {args.name}_update{final_cumulative:04d}.pt")
    print(line)


if __name__ == "__main__":
    main()
