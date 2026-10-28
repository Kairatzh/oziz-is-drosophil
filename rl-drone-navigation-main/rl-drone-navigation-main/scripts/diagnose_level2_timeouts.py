"""
Phase 4 follow-up failure analysis: after the lidar fix + longer training
(150/250 updates), diagnose_level2_failures.py found ZERO collisions or
out-of-bounds across 20 deterministic eval episodes -- all 8 non-successes
were timeouts. That is a different failure mode than the original
"agent doesn't perceive obstacles" problem, so it needs a different
diagnostic: for each timeout episode, track the closest the agent ever
got to the goal and its speed at that closest approach. This
distinguishes "never found the goal" (closest approach is large) from
"reached the goal but flew through too fast to satisfy GOAL_SPEED_LIMIT"
(closest approach is within GOAL_RADIUS but speed there exceeds the
limit).

Deterministic evaluation only -- no training, no hyperparameter changes.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM
from viz.render_drone import render_drone_gif

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "phase4_level2_wide_gaps_final.pt")
N_EPISODES = 20
EVAL_SEED = 1000  # matches the eval_env seed used during training


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


def closest_approach(states, goal):
    dists = np.linalg.norm(states[:, 0:2] - goal[None, :], axis=1)
    speeds = np.linalg.norm(states[:, 2:4], axis=1)
    i = int(np.argmin(dists))
    return i, dists[i], speeds[i]


def main():
    level2 = next(l for l in CURRICULUM if l["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)

    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT_PATH)
    print(f"loaded {CHECKPOINT_PATH}")

    episodes = []
    for i in range(N_EPISODES):
        states, actions, outcome, steps, goal = run_episode(agent, env)
        episodes.append({
            "index": i, "states": states, "actions": actions, "goal": goal,
            "outcome": outcome, "steps": steps,
        })

    outcomes = [ep["outcome"] for ep in episodes]
    print("\n--- outcome breakdown (20 episodes) ---")
    for outcome_type in ("success", "collision", "out_of_bounds", "timeout"):
        count = outcomes.count(outcome_type)
        print(f"  {outcome_type:14s}: {count:2d}/20  ({count/N_EPISODES:.0%})")

    timeouts = [ep for ep in episodes if ep["outcome"] == "timeout"]
    print(f"\n--- closest-approach analysis for {len(timeouts)} timeout episodes ---")
    print(f"  (GOAL_RADIUS={env.GOAL_RADIUS}m, GOAL_SPEED_LIMIT={env.GOAL_SPEED_LIMIT}m/s)")
    for ep in timeouts:
        i, dist, speed = closest_approach(ep["states"], ep["goal"])
        frac = i / ep["steps"]
        in_radius = dist <= env.GOAL_RADIUS
        reason = "within goal radius but too fast" if in_radius and speed > env.GOAL_SPEED_LIMIT else \
                 "never entered goal radius"
        ep["closest_step"], ep["closest_dist"], ep["closest_speed"] = i, dist, speed
        ep["reason"] = reason
        print(
            f"    ep{ep['index']:2d}: closest_dist={dist:5.2f}m at step {i:3d}/{ep['steps']} "
            f"({frac:.0%} through) speed_there={speed:5.2f}m/s  -> {reason}"
        )

    n_overshoot = sum(1 for ep in timeouts if ep["reason"].startswith("within"))
    n_never_near = len(timeouts) - n_overshoot
    print(f"\n  summary: {n_overshoot}/{len(timeouts)} reached the goal radius but too fast (overshoot), "
          f"{n_never_near}/{len(timeouts)} never got within goal radius")

    # representative GIFs: prefer variety -- one overshoot case and one
    # never-near case if both exist, else fill from what's available.
    representative = []
    overshoot_ep = next((ep for ep in timeouts if ep["reason"].startswith("within")), None)
    never_near_ep = next((ep for ep in timeouts if ep["reason"].startswith("never")), None)
    for ep in (overshoot_ep, never_near_ep):
        if ep is not None:
            representative.append(ep)
    remaining = [ep for ep in timeouts if ep not in representative]
    remaining.sort(key=lambda ep: ep["closest_dist"])
    representative += remaining[: max(0, 3 - len(representative))]
    representative = representative[:3]

    print(f"\n--- saving {len(representative)} representative timeout GIFs ---")
    for ep in representative:
        env.goal = ep["goal"]
        tag = "overshoot" if ep["reason"].startswith("within") else "far"
        out_path = os.path.join(RESULTS_DIR, f"phase4_level2_timeout_ep{ep['index']}_{tag}.gif")
        n_frames = render_drone_gif(
            env, ep["states"], ep["actions"], out_path,
            title=f"Level 2 timeout analysis: ep{ep['index']} (closest={ep['closest_dist']:.2f}m, {tag})",
            frame_stride=2,
        )
        print(f"  saved {out_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
