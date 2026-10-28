"""
Level 1.5 curriculum step: one small (r=0.8) obstacle placed randomly each
episode but guaranteed clear of the straight start->goal line (see
make_level15_obstacle_fn). The stepping stone between Level 1 (no obstacles)
and Level 2 (must route around): the agent learns "obstacles exist and hurt"
while a straight path to the goal still works.

Warm-started from the Level 1 checkpoint (reward unchanged: Level 1 has no
obstacles so the clearance/LOS terms were inert there). 100-episode
deterministic decision evals drive progression (>=90%); 250-update cap.
entropy_coef=0.003. Per CLAUDE.md: no plt.show(); CPU only.
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
WARM_START = os.path.join(CHECKPOINT_DIR, "phase4_level1_v2_final.pt")
NAME = "phase4_level1_5"
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def deterministic_eval(agent, env, n):
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
    p.add_argument("--updates", type=int, default=50, help="updates to run this block (cap is 250)")
    p.add_argument("--entropy-coef", type=float, default=0.003)
    p.add_argument("--resume-checkpoint", default=WARM_START)
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    level = next(c for c in CURRICULUM if c["name"] == "level1_5_single_clear_obstacle")
    ofn = level["obstacle_fn"]

    print("=" * 76)
    print(f"LEVEL 1.5 -- one r=0.8 obstacle, clear line of sight; warm-start from "
          f"{os.path.basename(args.resume_checkpoint)}")
    print(f"  updates {args.offset + 1}..{args.offset + args.updates} (cap 250); "
          f"entropy_coef={args.entropy_coef}; 100-ep decision evals, advance at "
          f"{SUCCESS_THRESHOLD:.0%}")
    print("=" * 76)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=args.entropy_coef),
    )
    agent.load(args.resume_checkpoint)

    env = QuadrotorEnv(obstacle_fn=ofn, seed=SEED)
    eval_env = QuadrotorEnv(obstacle_fn=ofn, seed=SEED + 1000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_outcomes, entropy_trend, eval_history = [], [], []
    solved_at = None
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
                episode_outcomes.append(info["outcome"])
                current = 0.0
                obs = env.reset()

        diag = agent.update(buffer, agent.get_value(obs))
        entropy_trend.append((cumulative, diag["entropy"]))
        recent = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_sr = np.mean([o == "success" for o in recent]) if recent else 0.0
        print(f"  update {cumulative:3d}  train_pulse:{train_sr:5.1%}  entropy:{diag['entropy']:+.3f}")

        if u % 10 == 0:
            sr, ar, counts = deterministic_eval(agent, eval_env, 100)
            eval_history.append((cumulative, sr, counts))
            print(f"      [100-ep decision @ {cumulative}] success:{sr:5.1%}  "
                  f"collision:{counts.get('collision', 0)/100:.0%}  avg_reward:{ar:.2f}")
            if sr >= SUCCESS_THRESHOLD and solved_at is None:
                solved_at = cumulative
                print(f"  *** LEVEL 1.5 SOLVED: {sr:.0%} at update {cumulative} ***")
                break

    agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_update{args.offset + args.updates:04d}.pt"))
    if solved_at is not None:
        agent.save(os.path.join(CHECKPOINT_DIR, f"{NAME}_final.pt"))
    elapsed = time.time() - t0

    line = "=" * 76
    print("\n" + line)
    print(f"LEVEL 1.5 REPORT -- through update {args.offset + args.updates}")
    print(line)
    if solved_at is not None:
        print(f"  SOLVED at update {solved_at} (100-ep eval >= {SUCCESS_THRESHOLD:.0%})")
    else:
        best = max((sr for _, sr, _ in eval_history), default=0.0)
        print(f"  not yet solved; best 100-ep decision eval: {best:.0%}")
    print("\n  100-ep decision eval trend (success / collision):")
    for cu, sr, counts in eval_history:
        mark = "  <- solved" if solved_at == cu else ""
        print(f"    update {cu:3d}: success {sr:5.1%}   collision {counts.get('collision', 0)/100:5.1%}   "
              f"timeout {counts.get('timeout', 0)/100:5.1%}{mark}")
    if eval_history:
        _, _, last = eval_history[-1]
        print("\n  final deterministic breakdown (100 ep):")
        for k in OUTCOME_KEYS:
            print(f"    {k:14s}: {last.get(k, 0):3d}  ({last.get(k, 0)/100:5.1%})")
    e0, e1 = entropy_trend[0][1], entropy_trend[-1][1]
    print(f"\n  entropy: {e0:+.3f} -> {e1:+.3f}")
    print(f"  wall time: {elapsed:.1f}s   checkpoint: {NAME}_update{args.offset + args.updates:04d}.pt")
    print(line)


if __name__ == "__main__":
    main()
