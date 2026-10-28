"""
Phase 4 follow-up: diagnose_reward_composition.py showed the per-step
reward at the frozen states in ep0/ep2 is small and nearly constant
regardless of action, dominated by the flat time penalty rather than the
(vanishing, since velocity ~0) progress shaping term. That only rules out
the *immediate* reward as the cause. Whether stalling looks attractive to
the policy also depends on the critic's learned V(s) at those states --
if V(s) is high there, the critic itself has learned that hovering in
this spot is "good" (e.g. because it's discounted away the risk of the
obstacle field ahead), which would point at the value function /
training dynamics rather than the reward shaping. If V(s) is low, the
critic already knows this is a bad place to be and the actor is failing
to act on that signal.

Replays ep0 and ep2 exactly (same seed as prior diagnostics), reconstructs
the true observation (including lidar) at each frozen step via
env._get_obs(), and prints the critic's raw value estimate there.

Deterministic evaluation only -- no training, no hyperparameter changes.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "phase4_level2_wide_gaps_final.pt")
EVAL_SEED = 1000
TARGET_EPISODES = {0, 2}
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


def find_frozen_window(states, window):
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
        print(f"episode {ep_idx}: outcome={outcome}  goal={goal}")

        frozen_start = find_frozen_window(states, WINDOW * 3)
        if frozen_start is None:
            print("  (no frozen window found -- skipping)")
            continue

        # Reference values for scale: V at the very first state of the
        # episode (agent hasn't moved yet) and V right before freezing
        # began (agent still approaching).
        env.state = states[0].copy()
        env.goal = goal.copy()
        v_start = agent.get_value(env._get_obs())

        approach_idx = max(0, frozen_start - 10)
        env.state = states[approach_idx].copy()
        v_pre_freeze = agent.get_value(env._get_obs())

        print(f"  V(s) at episode start (step 0):            {v_start:9.4f}")
        print(f"  V(s) at step {approach_idx:3d} (still approaching, 10 steps before freeze): {v_pre_freeze:9.4f}\n")

        print(f"  {'step':>5s} {'x':>7s} {'z':>7s} {'dist_to_goal':>13s} {'V(s)':>10s}")
        for k in range(WINDOW):
            i = frozen_start + k
            state = states[i]
            env.state = state.copy()
            obs = env._get_obs()
            value = agent.get_value(obs)
            dist = float(np.hypot(state[0] - goal[0], state[1] - goal[1]))
            print(f"  {i:5d} {state[0]:7.3f} {state[1]:7.3f} {dist:13.4f} {value:10.4f}")
        print()


if __name__ == "__main__":
    main()
