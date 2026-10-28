"""
Phase 4 follow-up: diagnose_level2_timeouts.py found that timeout
episodes aren't wandering or crashing -- the agent reaches some point in
open space, then holds still there for the rest of the episode. This
script replays those exact episodes and decomposes the reward into its
three per-step components (see QuadrotorEnv._compute_reward_and_done):
  1. progress shaping:  W_PROGRESS * (prev_dist - dist)
  2. time penalty:       -W_TIME
  3. control effort:     -W_EFFORT * (left^2 + right^2)
for a window of steps while the agent is frozen, to see whether the
shaping term is actually pushing toward the goal at that state or
whether something cancels it out.

Deterministic evaluation only -- no training, no hyperparameter changes.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "phase4_level2_wide_gaps_update0100.pt")
# NOTE: intentionally NOT "..._final.pt" -- that file was overwritten by the
# from-scratch Level 1 + 5-term-reward Level 2 warm-start retrain. This
# specific update0100 file is the checkpoint from the OLD 3-term-reward run
# at cumulative update 150, which is the one that actually produced the ep0
# frozen-state trajectory being re-examined here.
EVAL_SEED = 1000
TARGET_EPISODES = {0}  # ep0 only, per the velocity-term sanity check
WINDOW = 5


def run_episode(agent, env):
    obs = env.reset()
    goal = env.goal.copy()
    states = [env.state.copy()]
    actions = []
    done = False
    while not done:
        action = agent.act_deterministic(obs)
        actions.append(tuple(action))
        obs, reward, done, info = env.step(action)
        states.append(env.state.copy())
    return np.array(states), np.array(actions), info["outcome"], goal


def decompose_step(env, prev_dist, state, action):
    x, z, vx, vz = state[0], state[1], state[2], state[3]
    dist = float(np.hypot(x - env.goal[0], z - env.goal[1]))
    left, right = action
    progress = env.W_PROGRESS * (prev_dist - dist)
    time_pen = -env.W_TIME
    effort_pen = -env.W_EFFORT * (left ** 2 + right ** 2)
    if dist > 1e-6:
        goal_dir_x = (env.goal[0] - x) / dist
        goal_dir_z = (env.goal[1] - z) / dist
        vel_toward_goal = vx * goal_dir_x + vz * goal_dir_z
    else:
        vel_toward_goal = 0.0
    velocity_term = env.W_VELOCITY * vel_toward_goal
    dist_pen = -env.W_DIST * dist
    total = progress + time_pen + effort_pen + velocity_term + dist_pen
    return dist, progress, time_pen, effort_pen, velocity_term, dist_pen, total


def find_frozen_window(states, window):
    """First index i such that positions barely move over [i, i+window]."""
    pos = states[:, 0:2]
    for i in range(len(pos) - window):
        span = np.linalg.norm(pos[i:i + window + 1] - pos[i], axis=1).max()
        if span < 0.02:
            return i
    return None


def main():
    level2 = next(l for l in CURRICULUM if l["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)

    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT_PATH)
    print(f"loaded {CHECKPOINT_PATH}\n")

    for ep_idx in range(max(TARGET_EPISODES) + 1):
        states, actions, outcome, goal = run_episode(agent, env)
        if ep_idx not in TARGET_EPISODES:
            continue

        print("=" * 78)
        print(f"episode {ep_idx}: outcome={outcome}  goal={goal}  steps={len(actions)}")

        frozen_start = find_frozen_window(states, WINDOW * 3)
        if frozen_start is None:
            print("  (no clearly frozen window found -- skipping)")
            continue

        # prev_dist going into frozen_start = distance at frozen_start (state before the window)
        prev_dist = float(np.hypot(
            states[frozen_start, 0] - goal[0], states[frozen_start, 1] - goal[1]
        ))
        print(f"  frozen window starts at step {frozen_start}/{len(actions)} "
              f"({frozen_start/len(actions):.0%} through), position={states[frozen_start, 0:2].round(3)}")
        print(f"  goal={goal.round(3)}  dist_to_goal_at_window_start={prev_dist:.4f}m\n")

        print(f"  {'step':>5s} {'x':>7s} {'z':>7s} {'vx':>7s} {'vz':>7s} "
              f"{'dist':>7s} {'progress':>10s} {'time_pen':>10s} {'effort_pen':>10s} "
              f"{'vel_term':>10s} {'dist_pen':>10s} {'total':>10s}")
        for k in range(WINDOW):
            i = frozen_start + k
            state = states[i + 1]
            action = actions[i]
            dist, progress, time_pen, effort_pen, velocity_term, dist_pen, total = decompose_step(
                env, prev_dist, state, action
            )
            print(
                f"  {i:5d} {state[0]:7.3f} {state[1]:7.3f} {state[2]:7.4f} {state[3]:7.4f} "
                f"{dist:7.4f} {progress:10.6f} {time_pen:10.6f} {effort_pen:10.6f} "
                f"{velocity_term:10.6f} {dist_pen:10.6f} {total:10.6f}"
            )
            prev_dist = dist
        print()


if __name__ == "__main__":
    main()
