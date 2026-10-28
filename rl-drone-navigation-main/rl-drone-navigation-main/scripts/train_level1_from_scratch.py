"""
Fresh Level 1 (no obstacles) training under the NEW 7-term reward
(goal-proximity bonus added on top of clearance shaping). No warm start:
the reward function changed, so any prior checkpoint is off-distribution and
would only mislead the value function -- we start from random init.

Progression decision uses 100-EPISODE deterministic evals (the project
standard for any decision -- a 20-ep eval has ~11-point sampling noise). The
per-update stochastic-rollout success rate is the free mid-training pulse.

Settings pinned by the user for this run:
  entropy_coef = 0.003   (not the curriculum default 0.001)
  W_DIST = 0.002, W_CLEARANCE = 0.15   (already the env defaults)

Reuses the curriculum's Level 1 config and shared components unchanged.
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
from training.utils import plot_learning_curve, record_demo_episode, run_evaluation
from scripts.train_drone_curriculum import (
    CURRICULUM, ROLLOUT_STEPS, SEED, SUCCESS_THRESHOLD, MOVING_AVG_WINDOW,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")

MAX_UPDATES = 250
EVAL_EVERY = 5
DECISION_EVAL_EPISODES = 100     # 100-ep evals drive the progression decision
ENTROPY_COEF = 0.003
NAME = "phase4_level1_v2"        # "v2" = trained under the 7-term reward; keeps the old-reward Level 1 files intact
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    level1 = next(c for c in CURRICULUM if c["name"] == "level1_no_obstacles")
    print("=" * 74)
    print("FRESH LEVEL 1 TRAINING -- 7-term reward (goal-proximity bonus included)")
    print(f"  no warm start (random init); entropy_coef={ENTROPY_COEF}")
    print(f"  progression: eval every {EVAL_EVERY} updates over "
          f"{DECISION_EVAL_EPISODES} deterministic episodes; advance at "
          f">= {SUCCESS_THRESHOLD:.0%}")
    print(f"  reward weights: W_DIST={QuadrotorEnv.W_DIST}, W_CLEARANCE={QuadrotorEnv.W_CLEARANCE}, "
          f"W_GOAL_PROX={QuadrotorEnv.W_GOAL_PROX} (range {QuadrotorEnv.GOAL_PROX_RANGE} m)")
    print("=" * 74)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=ENTROPY_COEF),
    )
    env = QuadrotorEnv(obstacles=level1["obstacles"], seed=SEED)
    eval_env = QuadrotorEnv(obstacles=level1["obstacles"], seed=SEED + 1000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards, episode_outcomes, eval_history = [], [], []
    solved_at, solved_rate = None, None
    current = 0.0
    obs = env.reset()
    start_time = time.time()

    for update in range(1, MAX_UPDATES + 1):
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
        recent = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_sr = np.mean([o == "success" for o in recent]) if recent else 0.0
        recent_r = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        print(f"  update {update:3d}/{MAX_UPDATES}  episodes:{len(episode_rewards):5d}  "
              f"moving_avg_reward:{np.mean(recent_r):8.2f}  train_pulse:{train_sr:5.1%}  "
              f"entropy:{diagnostics['entropy']:+.3f}")

        if update % EVAL_EVERY == 0:
            esr, ear = run_evaluation(agent, eval_env, DECISION_EVAL_EPISODES)
            eval_history.append((update, esr, ear))
            print(f"      [100-ep eval @ update {update}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")
            if esr >= SUCCESS_THRESHOLD:
                solved_at, solved_rate = update, esr
                print(f"  *** LEVEL 1 SOLVED: {esr:.0%} (>= {SUCCESS_THRESHOLD:.0%}) at update {update} ***")
                break

        if update % 10 == 0:
            agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_update{update:04d}.pt"))

    elapsed = time.time() - start_time
    agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_final.pt"))

    # Final 100-ep success rate: reuse the solving eval, else measure now.
    if solved_rate is not None:
        final_rate = solved_rate
    else:
        final_rate, _ = run_evaluation(agent, eval_env, DECISION_EVAL_EPISODES)

    title = "Phase 4 Level 1 (7-term reward)"
    if solved_at is not None:
        title += f" -- solved at update {solved_at}"
    plot_learning_curve(
        episode_rewards, eval_history, SUCCESS_THRESHOLD, MOVING_AVG_WINDOW,
        out_path=os.path.join(RESULTS_DIR, f"{NAME}_learning_curve.png"), title=title,
    )
    demo_outcome, _ = record_demo_episode(
        agent, eval_env, os.path.join(RESULTS_DIR, f"{NAME}_demo.gif"), title=title
    )

    counts = Counter(episode_outcomes)
    total = sum(counts.values())
    print("\n" + "=" * 74)
    print("LEVEL 1 REPORT (7-term reward, from scratch)")
    print("=" * 74)
    if solved_at is not None:
        print(f"  updates to reach 90% (100-ep eval): {solved_at}")
    else:
        print(f"  updates to reach 90%: NOT REACHED within {MAX_UPDATES} updates")
    print(f"  final 100-episode eval success rate: {final_rate:.1%}")
    print(f"  training episodes: {total}   wall time: {elapsed:.1f}s")
    print("  100-ep eval history (update -> success):")
    for u, sr, ar in eval_history:
        mark = "  <- solved" if solved_at == u else ""
        print(f"    update {u:3d}: {sr:5.1%}  (avg_reward {ar:7.2f}){mark}")
    print("  training outcome mix (all episodes):")
    for k in OUTCOME_KEYS:
        c = counts.get(k, 0)
        print(f"    {k:14s}: {c:5d}  ({(c/total if total else 0):5.1%})")
    print(f"  final checkpoint: {NAME}_final.pt   demo outcome: {demo_outcome}")
    print("=" * 74)


if __name__ == "__main__":
    main()
