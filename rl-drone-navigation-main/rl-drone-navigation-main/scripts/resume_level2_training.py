"""
Phase 4 helper: RESUME Level 2 training from an existing checkpoint and run
a fixed number of additional PPO updates, then print a compact progress
report (eval success-rate table, entropy trend, training outcome breakdown).

Why a separate script instead of re-running train_drone_curriculum.py?
The curriculum runner restarts its per-run update counter at 1 and names
checkpoints update0010, update0020, ... -- resuming Level 2 with it would
overwrite the existing 0->50 checkpoints and the final.pt anchor, and the
filenames would no longer reflect the true cumulative update count. This
wrapper keeps cumulative numbering (via --offset) under a distinct "cont"
name so nothing is clobbered, and it prints the success/collision/timeout
breakdown that the base runner does not.

It reuses the SAME components as the curriculum (PPOAgent, QuadrotorEnv,
RolloutBuffer, run_evaluation) and the SAME Level 2 config and lowered
ENTROPY_COEF=0.001, so behavior is identical to the curriculum run apart
from bookkeeping.

Note: checkpoints store only network weights (no optimizer state), so
resuming restarts Adam's momentum fresh -- this is exactly how the
curriculum's own --resume-checkpoint path behaves, not new here.

Per CLAUDE.md: plt.show() is never called; CPU training only.
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer
from training.utils import run_evaluation

# Import the exact Level 2 definition and hyperparameters from the curriculum
# runner so this stays a faithful continuation rather than a divergent copy.
from scripts.train_drone_curriculum import (
    CURRICULUM,
    ENTROPY_COEF,
    EVAL_EPISODES,
    EVAL_EVERY,
    ROLLOUT_STEPS,
    SEED,
    SUCCESS_THRESHOLD,
    MOVING_AVG_WINDOW,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")

LEVEL2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")

# Outcome categories the env can emit (env/quadrotor_env.py). "out_of_bounds"
# is a distinct terminal from "collision" but is also a crash; reported
# separately so the breakdown is honest.
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def resume_and_train(resume_checkpoint, num_updates, offset, entropy_coef=ENTROPY_COEF):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=entropy_coef),
    )
    agent.load(resume_checkpoint)
    print(f"resumed agent weights from {resume_checkpoint}")
    print(f"entropy_coef={entropy_coef} (curriculum default is {ENTROPY_COEF})")
    print(f"running {num_updates} updates on {LEVEL2['name']} "
          f"(cumulative updates {offset + 1}..{offset + num_updates})\n")

    env = QuadrotorEnv(obstacles=LEVEL2["obstacles"], seed=SEED)
    eval_env = QuadrotorEnv(obstacles=LEVEL2["obstacles"], seed=SEED + 1000)
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards = []
    episode_outcomes = []
    entropy_trend = []      # (cumulative_update, entropy)
    eval_history = []       # (cumulative_update, eval_success_rate, eval_avg_reward)
    solved_at = None

    current_episode_reward = 0.0
    obs = env.reset()

    for i in range(1, num_updates + 1):
        cumulative = offset + i
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=done, value=value)
            current_episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(current_episode_reward)
                episode_outcomes.append(info["outcome"])
                current_episode_reward = 0.0
                obs = env.reset()

        last_value = agent.get_value(obs)
        diagnostics = agent.update(buffer, last_value)
        entropy_trend.append((cumulative, diagnostics["entropy"]))

        recent_outcomes = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_success_rate = (
            np.mean([o == "success" for o in recent_outcomes]) if recent_outcomes else 0.0
        )
        recent_rewards = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        print(
            f"  update {cumulative:3d}  episodes:{len(episode_rewards):5d}  "
            f"moving_avg_reward:{np.mean(recent_rewards):8.2f}  "
            f"train_success:{train_success_rate:5.1%}  "
            f"entropy:{diagnostics['entropy']:+.3f}  "
            f"policy_loss:{diagnostics['policy_loss']:+.4f}"
        )

        if i % EVAL_EVERY == 0:
            esr, ear = run_evaluation(agent, eval_env, EVAL_EPISODES)
            eval_history.append((cumulative, esr, ear))
            print(f"      [eval @ update {cumulative}] success_rate:{esr:5.1%}  avg_reward:{ear:8.2f}")
            if esr >= SUCCESS_THRESHOLD and solved_at is None:
                solved_at = cumulative
                print(f"  *** {LEVEL2['name']} reached {SUCCESS_THRESHOLD:.0%} eval success "
                      f"at cumulative update {cumulative} ***")
                break

        if i % 10 == 0:
            ckpt = os.path.join(CHECKPOINT_DIR, f"phase4_level2_wide_gaps_cont_update{cumulative:04d}.pt")
            agent.save(ckpt)

    # Always leave a resumable anchor at the end of this block.
    final_cumulative = offset + num_updates if solved_at is None else solved_at
    latest = os.path.join(CHECKPOINT_DIR, f"phase4_level2_wide_gaps_cont_update{final_cumulative:04d}.pt")
    agent.save(latest)

    return {
        "episode_rewards": episode_rewards,
        "episode_outcomes": episode_outcomes,
        "entropy_trend": entropy_trend,
        "eval_history": eval_history,
        "solved_at": solved_at,
        "final_cumulative": final_cumulative,
        "latest_checkpoint": latest,
    }


def print_report(r, offset, num_updates):
    line = "=" * 72
    print("\n" + line)
    print(f"PROGRESS REPORT  (cumulative updates {offset + 1}..{r['final_cumulative']})")
    print(line)

    print("\n1) EVAL SUCCESS RATE  (20 deterministic episodes, every 5 updates)")
    print(f"   {'update':>8}  {'eval success':>12}  {'eval avg reward':>15}")
    for upd, esr, ear in r["eval_history"]:
        print(f"   {upd:>8}  {esr:>11.1%}  {ear:>15.2f}")

    print("\n2) ENTROPY TREND  (policy stochasticity per update; lower = more decisive)")
    et = r["entropy_trend"]
    if et:
        first_u, first_e = et[0]
        last_u, last_e = et[-1]
        emin = min(et, key=lambda t: t[1])
        emax = max(et, key=lambda t: t[1])
        print(f"   start (u{first_u}): {first_e:+.3f}    end (u{last_u}): {last_e:+.3f}    "
              f"delta: {last_e - first_e:+.3f}")
        print(f"   min: {emin[1]:+.3f} (u{emin[0]})    max: {emax[1]:+.3f} (u{emax[0]})")
        step = max(1, len(et) // 10)
        sampled = " ".join(f"u{u}:{e:+.2f}" for u, e in et[::step])
        print(f"   sampled: {sampled}")

    print("\n3) TRAINING OUTCOME BREAKDOWN  (all episodes across these updates)")
    counts = Counter(r["episode_outcomes"])
    total = sum(counts.values())
    for k in OUTCOME_KEYS:
        c = counts.get(k, 0)
        frac = c / total if total else 0.0
        print(f"   {k:>14}: {c:5d}  ({frac:5.1%})")
    other = {k: v for k, v in counts.items() if k not in OUTCOME_KEYS}
    for k, v in other.items():
        print(f"   {k:>14}: {v:5d}  ({v / total:5.1%})  [unexpected]")
    print(f"   {'TOTAL':>14}: {total:5d}")

    print("\n" + line)
    if r["solved_at"] is not None:
        print(f"STATUS: reached {SUCCESS_THRESHOLD:.0%} eval success at update {r['solved_at']} "
              f"(early stop).")
    else:
        print(f"STATUS: ran full {num_updates}-update block; did NOT reach "
              f"{SUCCESS_THRESHOLD:.0%} eval success. Now at cumulative update {r['final_cumulative']}.")
    print(f"latest checkpoint: {os.path.basename(r['latest_checkpoint'])}")
    print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", type=str, required=True)
    parser.add_argument("--updates", type=int, required=True,
                        help="number of additional PPO updates to run in this block")
    parser.add_argument("--offset", type=int, required=True,
                        help="cumulative update count the resume checkpoint represents")
    parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF,
                        help=f"override the entropy bonus coefficient (curriculum default {ENTROPY_COEF})")
    args = parser.parse_args()

    result = resume_and_train(args.resume_checkpoint, args.updates, args.offset, args.entropy_coef)
    print_report(result, args.offset, args.updates)
