"""
Level 2 (1-2 wide-gap obstacles), warm-started from the new Level 1 checkpoint
(phase4_level1_v2_final.pt) under the 7-term reward. This is the real test of
whether clearance shaping (term 6) + goal-proximity bonus (term 7) break the
~65% Level 2 plateau we diagnosed (collisions into obstacles + stalling just
short of the goal).

Two-tier evaluation, per the project standard:
  - 100-episode deterministic eval = the PROGRESSION DECISION (>= 90% solves).
  - 20-episode deterministic eval = a cheap mid-training PULSE only.
Held-out seeds: decision uses SEED+1000, pulse uses SEED+2000 (both distinct
from the training seed).

This run does exactly 50 updates and STOPS (does not auto-advance to Level 3),
per the user's request to check in first. entropy_coef=0.003.
Per CLAUDE.md: plt.show() is never called; CPU training only.
"""

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
WARM_START = os.path.join(CHECKPOINT_DIR, "phase4_level1_v2_final.pt")

NUM_UPDATES = 50
ENTROPY_COEF = 0.003
DECISION_EVAL_EPISODES = 100    # drives the >= 90% progression decision
PULSE_EVAL_EPISODES = 20        # cheap mid-training check only
NAME = "phase4_level2_v2"       # fresh lineage (7-term reward, warm-started from level1_v2)
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    level2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")
    print("=" * 76)
    print("LEVEL 2 -- warm-started from phase4_level1_v2_final.pt (7-term reward)")
    print(f"  entropy_coef={ENTROPY_COEF}; running {NUM_UPDATES} updates then STOP")
    print(f"  decision eval = {DECISION_EVAL_EPISODES} ep (seed {SEED+1000}); "
          f"pulse = {PULSE_EVAL_EPISODES} ep (seed {SEED+2000})")
    print(f"  reward: W_DIST={QuadrotorEnv.W_DIST}, W_CLEARANCE={QuadrotorEnv.W_CLEARANCE}, "
          f"W_GOAL_PROX={QuadrotorEnv.W_GOAL_PROX}")
    print("=" * 76)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=ENTROPY_COEF),
    )
    agent.load(WARM_START)
    print(f"warm-started from {os.path.basename(WARM_START)}\n")

    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED)
    decision_env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED + 1000)
    pulse_env = QuadrotorEnv(obstacles=level2["obstacles"], seed=SEED + 2000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards, episode_outcomes = [], []
    entropy_trend = []      # (update, entropy)
    eval_history = []       # (update, kind, success_rate, avg_reward)
    reached_90_at = None
    current = 0.0
    obs = env.reset()
    start_time = time.time()

    for update in range(1, NUM_UPDATES + 1):
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
        print(f"  update {update:3d}/{NUM_UPDATES}  episodes:{len(episode_rewards):5d}  "
              f"moving_avg_reward:{np.mean(recent_r):8.2f}  train_pulse:{train_sr:5.1%}  "
              f"entropy:{diagnostics['entropy']:+.3f}")

        # every 10 -> 100-ep DECISION eval; else every 5 -> 20-ep PULSE
        if update % 10 == 0:
            esr, ear = run_evaluation(agent, decision_env, DECISION_EVAL_EPISODES)
            eval_history.append((update, "decision-100", esr, ear))
            print(f"      [DECISION 100-ep @ update {update}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")
            if esr >= SUCCESS_THRESHOLD and reached_90_at is None:
                reached_90_at = update
                print(f"      *** reached progression threshold {SUCCESS_THRESHOLD:.0%} at update {update} "
                      f"(not advancing -- holding at 50 as instructed) ***")
        elif update % 5 == 0:
            esr, ear = run_evaluation(agent, pulse_env, PULSE_EVAL_EPISODES)
            eval_history.append((update, "pulse-20", esr, ear))
            print(f"      [pulse 20-ep @ update {update}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")

        if update % 10 == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_update{update:04d}.pt"))

    elapsed = time.time() - start_time
    agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_update{NUM_UPDATES:04d}.pt"))

    # ---------------------------- report ----------------------------
    line = "=" * 76
    print("\n" + line)
    print(f"LEVEL 2 (v2, warm-started) REPORT -- updates 1..{NUM_UPDATES}")
    print(line)

    print("\n1) EVAL SUCCESS RATE  (100-ep = decision, 20-ep = pulse only)")
    print(f"   {'update':>6}  {'eval type':>13}  {'success':>8}  {'avg_reward':>10}")
    for u, kind, sr, ar in eval_history:
        print(f"   {u:>6}  {kind:>13}  {sr:>7.1%}  {ar:>10.2f}")

    print("\n2) ENTROPY TREND")
    if entropy_trend:
        u0, e0 = entropy_trend[0]
        u1, e1 = entropy_trend[-1]
        emin = min(entropy_trend, key=lambda t: t[1])
        emax = max(entropy_trend, key=lambda t: t[1])
        print(f"   start (u{u0}): {e0:+.3f}    end (u{u1}): {e1:+.3f}    delta: {e1 - e0:+.3f}")
        print(f"   min: {emin[1]:+.3f} (u{emin[0]})    max: {emax[1]:+.3f} (u{emax[0]})")
        step = max(1, len(entropy_trend) // 10)
        print("   sampled: " + " ".join(f"u{u}:{e:+.2f}" for u, e in entropy_trend[::step]))

    print("\n3) TRAINING OUTCOME BREAKDOWN  (all episodes this block)")
    counts = Counter(episode_outcomes)
    total = sum(counts.values())
    for k in OUTCOME_KEYS:
        c = counts.get(k, 0)
        print(f"   {k:>14}: {c:5d}  ({(c/total if total else 0):5.1%})")
    print(f"   {'TOTAL':>14}: {total:5d}")

    print("\n" + line)
    decisions = [(u, sr) for (u, kind, sr, _) in eval_history if kind == "decision-100"]
    last_decision = decisions[-1] if decisions else None
    if reached_90_at is not None:
        print(f"STATUS: reached {SUCCESS_THRESHOLD:.0%} on a 100-ep eval at update {reached_90_at}; "
              f"held at {NUM_UPDATES} without advancing (awaiting confirmation).")
    else:
        best = max(decisions, key=lambda t: t[1]) if decisions else (None, 0.0)
        print(f"STATUS: did NOT reach {SUCCESS_THRESHOLD:.0%} on a 100-ep eval within {NUM_UPDATES} updates. "
              f"best 100-ep decision eval: {best[1]:.0%} (update {best[0]}).")
    if last_decision:
        print(f"final 100-ep decision eval (update {last_decision[0]}): {last_decision[1]:.1%}")
    print(f"wall time: {elapsed:.1f}s   latest checkpoint: {NAME}_update{NUM_UPDATES:04d}.pt")
    print(line)


if __name__ == "__main__":
    main()
