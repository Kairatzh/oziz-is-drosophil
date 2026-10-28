"""
Phase 5: generalization testing.

Tests the Phase 4 primary agent (model_checkpoints/final_model.pt, a copy of
phase4_level1_5_final.pt -- 99% success / 0% collision in training) against
obstacle courses it never saw during training, to quantify the train/test
gap. Training's ONLY obstacle generator (Level 1.5) always placed exactly
one obstacle with its surface kept clear of the straight start->goal line by
QuadrotorEnv.GOAL_LOS_MARGIN -- so the agent never once had to route around
an obstacle that blocked the direct path, never saw more than one obstacle,
and never saw a fixed (repeated) layout. Every held-out condition below
varies at least one of those three axes.

Read-only evaluation + visualization. Does not train.
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.quadrotor_env import CircleObstacle, QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM, make_level15_obstacle_fn
from viz.render_drone import render_drone_gif

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "final_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_EPISODES = 100  # per eval-batch-size convention: 100-ep for decision-grade numbers
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def make_random_obstacles_fn(n, radius, min_endpoint_clear=0.6, min_obstacle_gap=0.1, max_tries=300):
    """Places n circular obstacles at fully random positions each episode,
    with NO guarantee the straight start->goal line stays clear -- unlike
    make_level15_obstacle_fn, this is the axis training never varied.
    Obstacles are still kept clear of the start and goal points themselves
    (min_endpoint_clear) so episodes aren't degenerate instant-collisions."""
    def obstacle_fn(rng, start, goal):
        obstacles = []
        tries = 0
        while len(obstacles) < n and tries < max_tries:
            tries += 1
            x = float(rng.uniform(radius, QuadrotorEnv.WORLD_WIDTH - radius))
            z = float(rng.uniform(radius, QuadrotorEnv.WORLD_HEIGHT - radius))
            if np.hypot(x - start[0], z - start[1]) - radius < min_endpoint_clear:
                continue
            if np.hypot(x - goal[0], z - goal[1]) - radius < min_endpoint_clear:
                continue
            if any(np.hypot(x - o.x, z - o.z) < radius + o.radius + min_obstacle_gap for o in obstacles):
                continue
            obstacles.append(CircleObstacle(x=x, z=z, radius=radius))
        return obstacles
    return obstacle_fn


# "Unusual formation": a 3-obstacle vertical wall spanning most of the
# world's height, leaving ~0.6-0.7 m gaps above, below, and between the
# obstacles. This forces genuine gap-threading -- something Level 1.5
# training never required, since its single obstacle always left the
# straight line fully clear (no gap ever needed threading).
WALL_OBSTACLES = [
    CircleObstacle(x=5.0, z=1.5, radius=0.9),
    CircleObstacle(x=5.0, z=4.0, radius=0.9),
    CircleObstacle(x=5.0, z=6.5, radius=0.9),
]

LEVEL2_OBSTACLES = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")["obstacles"]
LEVEL3_OBSTACLES = next(c for c in CURRICULUM if c["name"] == "level3_moderate_gaps")["obstacles"]

TEST_CONDITIONS = [
    {
        "name": "train_distribution_control",
        "title": "In-distribution control (Level 1.5 generator, unseen seed)",
        "obstacles": [],
        "obstacle_fn": make_level15_obstacle_fn(),
        "seed": 4242,  # deliberately different from the seed=1000 used during training's own evals
    },
    {
        "name": "single_no_los_guarantee",
        "title": "1 obstacle r=0.8, NO clear-line-of-sight guarantee",
        "obstacles": [],
        "obstacle_fn": make_random_obstacles_fn(1, radius=0.8),
        "seed": 1000,
    },
    {
        "name": "level2_fixed_wide_gaps",
        "title": "2 fixed obstacles r=1.2 (curriculum Level 2 layout)",
        "obstacles": LEVEL2_OBSTACLES,
        "obstacle_fn": None,
        "seed": 1000,
    },
    {
        "name": "random_three_obstacles",
        "title": "3 random obstacles r=0.8, no LOS guarantee",
        "obstacles": [],
        "obstacle_fn": make_random_obstacles_fn(3, radius=0.8),
        "seed": 1000,
    },
    {
        "name": "level3_fixed_moderate_gaps",
        "title": "5 fixed obstacles r=0.8 (curriculum Level 3 layout)",
        "obstacles": LEVEL3_OBSTACLES,
        "obstacle_fn": None,
        "seed": 1000,
    },
    {
        "name": "wall_narrow_gaps",
        "title": "3-obstacle vertical wall, ~0.6-0.7m gaps (unusual formation)",
        "obstacles": WALL_OBSTACLES,
        "obstacle_fn": None,
        "seed": 1000,
    },
]


def evaluate(agent, env, n_episodes):
    outcomes = []
    total_reward = 0.0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        info = {"outcome": "running"}
        ep_reward = 0.0
        while not done:
            obs, reward, done, info = env.step(agent.act_deterministic(obs))
            ep_reward += reward
        outcomes.append(info["outcome"])
        total_reward += ep_reward
    return Counter(outcomes), total_reward / n_episodes


def save_success_and_failure_demo(agent, env, prefix, title, max_attempts=40):
    """Runs episodes until a success AND a failure (any non-success
    outcome) are both found, and renders each -- so the report can show
    what generalization looks like when it works and when it breaks."""
    found = {}
    for _ in range(max_attempts):
        if "success" in found and "failure" in found:
            break
        obs = env.reset()
        done = False
        states = [env.state.copy()]
        actions = [tuple(agent.act_deterministic(obs))]
        info = {"outcome": "running"}
        while not done:
            action = agent.act_deterministic(obs)
            obs, _, done, info = env.step(action)
            states.append(env.state.copy())
            actions.append(tuple(action))
        bucket = "success" if info["outcome"] == "success" else "failure"
        if bucket not in found:
            found[bucket] = (np.array(states), np.array(actions), info["outcome"])

    for bucket, (states, actions, outcome) in found.items():
        out_path = os.path.join(RESULTS_DIR, f"{prefix}_{bucket}.gif")
        render_drone_gif(env, states, actions, out_path, title=f"{title} ({outcome})")
        print(f"    saved {out_path}")


def main():
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)
    print(f"loaded {CHECKPOINT}\n")

    results = []
    for cond in TEST_CONDITIONS:
        env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
        counts, avg_reward = evaluate(agent, env, N_EPISODES)
        success_rate = counts.get("success", 0) / N_EPISODES
        results.append({
            "name": cond["name"], "title": cond["title"], "counts": counts,
            "success_rate": success_rate, "avg_reward": avg_reward,
        })
        print(f"[{cond['name']}] {cond['title']}")
        for k in OUTCOME_KEYS:
            c = counts.get(k, 0)
            print(f"    {k:14s}: {c:3d}  ({c / N_EPISODES:5.1%})")
        print(f"    avg_reward: {avg_reward:8.2f}\n")

    for cond in TEST_CONDITIONS:
        if cond["name"] in ("random_three_obstacles", "wall_narrow_gaps"):
            env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
            print(f"recording demo gifs for {cond['name']}...")
            save_success_and_failure_demo(agent, env, f"phase5_{cond['name']}", cond["title"])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r["name"] for r in results]
    rates = [r["success_rate"] for r in results]
    colors = ["tab:green"] + ["tab:red"] * (len(results) - 1)
    ax.bar(names, rates, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (100 deterministic episodes)")
    ax.set_title("Phase 5: generalization -- in-distribution control (green) vs held-out (red)")
    ax.axhline(results[0]["success_rate"], color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase5_generalization_success_rates.png"), dpi=120)
    plt.close("all")

    control_rate = results[0]["success_rate"]
    held_out = results[1:]
    avg_held_out = float(np.mean([r["success_rate"] for r in held_out]))
    print("=" * 78)
    print("PHASE 5 SUMMARY -- train vs held-out generalization gap")
    print("=" * 78)
    print(f"  in-distribution control : {control_rate:5.1%}")
    print(f"  held-out average        : {avg_held_out:5.1%}   (gap: {control_rate - avg_held_out:+.1%})")
    print("  per-condition gap vs control:")
    for r in held_out:
        print(f"    {r['name']:28s} {r['success_rate']:5.1%}   (gap: {control_rate - r['success_rate']:+.1%})")
    print("=" * 78)


if __name__ == "__main__":
    main()
