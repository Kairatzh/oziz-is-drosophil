"""
Phase 4 failure analysis: before changing any more hyperparameters,
characterize HOW the current Level 2 policy is failing. "Success rate is
stuck at ~50%" could mean several very different things -- the agent
might be flying straight through obstacles as if they don't exist (a
perception/observation problem), reacting late and clipping them (a
timing/precision problem), or successfully avoiding them but failing to
actually reach the goal afterward (a re-navigation problem after evasive
action). Each of those points to a different fix; guessing without this
breakdown risks tuning the wrong knob a third time.

This script only runs deterministic evaluation episodes -- no training,
no hyperparameter changes.
"""

import os

import numpy as np
import torch

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM
from viz.render_drone import render_drone_gif

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "phase4_level2_wide_gaps_final.pt")
N_EPISODES = 20
EVAL_SEED = 1000  # matches the eval_env seed used during training


def classify_timing(step, max_steps):
    frac = step / max_steps
    if frac < 0.2:
        return "early"
    elif frac < 0.8:
        return "mid"
    else:
        return "late"


def run_episode(agent, env):
    obs = env.reset()
    goal = env.goal.copy()
    states = [env.state.copy()]
    actions = [tuple(agent.act_deterministic(obs))]
    done = False
    info = {"outcome": "running"}
    while not done:
        action = agent.act_deterministic(obs)
        obs, reward, done, info = env.step(action)
        states.append(env.state.copy())
        actions.append(tuple(action))
    return np.array(states), np.array(actions), info["outcome"], env.steps, goal


def main():
    level2 = next(l for l in CURRICULUM if l["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)

    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT_PATH)
    print(f"loaded {CHECKPOINT_PATH}")

    episodes = []
    for i in range(N_EPISODES):
        states, actions, outcome, steps, goal = run_episode(agent, env)
        timing = classify_timing(steps, env.MAX_STEPS) if outcome in ("collision", "out_of_bounds") else None
        episodes.append({
            "index": i, "states": states, "actions": actions, "goal": goal,
            "outcome": outcome, "steps": steps, "timing": timing,
        })

    # --- (1) outcome breakdown ---
    outcomes = [ep["outcome"] for ep in episodes]
    print("\n--- outcome breakdown (20 episodes) ---")
    for outcome_type in ("success", "collision", "out_of_bounds", "timeout"):
        count = outcomes.count(outcome_type)
        print(f"  {outcome_type:14s}: {count:2d}/20  ({count/N_EPISODES:.0%})")

    # --- (2) timing of collisions/out-of-bounds ---
    crash_episodes = [ep for ep in episodes if ep["outcome"] in ("collision", "out_of_bounds")]
    print(f"\n--- timing of {len(crash_episodes)} collision/out-of-bounds failures "
          f"(early <20% of max_steps, mid 20-80%, late >80%) ---")
    for timing_bucket in ("early", "mid", "late"):
        count = sum(1 for ep in crash_episodes if ep["timing"] == timing_bucket)
        print(f"  {timing_bucket:6s}: {count}/{len(crash_episodes)}")
    print("\n  per-episode detail:")
    for ep in crash_episodes:
        frac = ep["steps"] / env.MAX_STEPS
        print(f"    ep{ep['index']:2d}: outcome={ep['outcome']:13s} step={ep['steps']:3d}/{env.MAX_STEPS} "
              f"({frac:.0%} through episode, bucket={ep['timing']})")

    # --- (3) save representative failure GIFs for visual inspection ---
    # Prefer one of each timing bucket if available, so the visual sample
    # spans "reacts late" vs "never reacts" cases rather than 3 near-
    # identical clips.
    representative = []
    for timing_bucket in ("early", "mid", "late"):
        match = next((ep for ep in crash_episodes if ep["timing"] == timing_bucket), None)
        if match is not None:
            representative.append(match)
    if not representative:
        representative = crash_episodes[:3]
    representative = representative[:3]

    print(f"\n--- saving {len(representative)} representative failure GIFs ---")
    for ep in representative:
        env.goal = ep["goal"]  # render_drone_gif reads env.goal -- restore the actual goal from this episode
        out_path = os.path.join(RESULTS_DIR, f"phase4_level2_failure_ep{ep['index']}_{ep['outcome']}_{ep['timing']}.gif")
        n_frames = render_drone_gif(
            env, ep["states"], ep["actions"], out_path,
            title=f"Level 2 failure analysis: ep{ep['index']} ({ep['outcome']}, {ep['timing']})",
            frame_stride=2,
        )
        print(f"  saved {out_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
