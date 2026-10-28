"""
Phase 4 failure-geometry diagnosis at cumulative update 200 (entropy 0.003
+ clearance shaping). Level 2 success has plateaued ~65% while the failure
mode shifted collision -> timeout. This asks WHERE the remaining failures
live, geometrically:

  (1) Are failures clustered on specific start->goal layouts (e.g. goal
      occluded by an obstacle) or spread randomly? -> we tag every episode
      by whether the STRAIGHT start->goal line is blocked by an obstacle,
      then compare failure rates across tags.
  (2) For timeouts, where does the agent end up? -> classify the final
      resting place: overshot the goal (too fast), parked near an obstacle
      it routed around, hovering near-but-not-in the goal, or stuck
      elsewhere.
  (3) What fraction of failures have an occluded straight path vs a clear
      straight path that still fails?

Deterministic eval only, on the same distribution training used
(EVAL_SEED=1000). Runs more episodes than a single eval (N_EPISODES below)
purely for a bigger failure sample -- same distribution, more draws. Does
NOT train and does NOT modify anything. Saves 2-3 representative GIFs.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM
from viz.render_drone import render_drone_gif

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model_checkpoints",
    "phase4_level2_wide_gaps_cont_update0200.pt",
)
N_EPISODES = 100         # more than one 20-episode eval, for a bigger failure sample
EVAL_SEED = 1000         # matches the eval_env seed used during training
CLEAR_BAND = QuadrotorEnv.LIDAR_CLEAR_THRESHOLD * QuadrotorEnv.LIDAR_MAX_RANGE  # 0.9 m


def run_episode(agent, env):
    obs = env.reset()
    start = env.state[:2].copy()
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
    return np.array(states), np.array(actions), info["outcome"], env.steps, start, goal


def point_segment_distance(c, a, b):
    """Shortest distance from point c to segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(c - a))
    t = np.clip(float((c - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(c - proj))


def straight_path_geometry(start, goal, obstacles):
    """For the straight start->goal segment, find the tightest obstacle.
    Returns (corridor_clearance, obstacle). corridor_clearance = distance
    from the segment to the obstacle SURFACE (seg-to-center minus radius):
      < 0  -> the straight line passes THROUGH the disk (goal occluded)
      >=0  -> straight line clears the disk by that many metres.
    """
    tightest = None
    best = float("inf")
    for o in obstacles:
        seg_d = point_segment_distance(np.array([o.x, o.z]), start, goal)
        clearance = seg_d - o.radius
        if clearance < best:
            best, tightest = clearance, o
    return best, tightest


def min_obstacle_surface_dist(p, obstacles):
    if not obstacles:
        return float("inf")
    return min(o.distance_to_point(p[0], p[1]) for o in obstacles)


def classify_timeout(ep, env):
    """Where did a timeout end up?"""
    states, goal = ep["states"], ep["goal"]
    dists = np.linalg.norm(states[:, 0:2] - goal[None, :], axis=1)
    speeds = np.linalg.norm(states[:, 2:4], axis=1)
    ci = int(np.argmin(dists))
    closest_dist, closest_speed = float(dists[ci]), float(speeds[ci])
    final_pos = states[-1, 0:2]
    final_goal_dist = float(dists[-1])
    final_obs_dist = min_obstacle_surface_dist(final_pos, env.obstacles)

    if closest_dist <= env.GOAL_RADIUS and closest_speed > env.GOAL_SPEED_LIMIT:
        cat = "overshoot (hit goal radius too fast)"
    elif final_obs_dist <= CLEAR_BAND:
        cat = "parked near obstacle"
    elif final_goal_dist <= 3 * env.GOAL_RADIUS:
        cat = "hovering near goal (never settled in)"
    else:
        cat = "stuck mid-field"
    ep.update(closest_dist=closest_dist, closest_speed=closest_speed,
              final_goal_dist=final_goal_dist, final_obs_dist=final_obs_dist, timeout_cat=cat)
    return cat


def main():
    level2 = next(l for l in CURRICULUM if l["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT_PATH)
    print(f"loaded {os.path.basename(CHECKPOINT_PATH)}   ({N_EPISODES} eval-distribution episodes)")
    print(f"obstacles: " + ", ".join(f"(x={o.x}, z={o.z}, r={o.radius})" for o in env.obstacles))
    print(f"occluded := straight start->goal line passes through an obstacle "
          f"(corridor clearance < 0); clear band = {CLEAR_BAND:.1f} m\n")

    episodes = []
    for i in range(N_EPISODES):
        states, actions, outcome, steps, start, goal = run_episode(agent, env)
        clearance, obstacle = straight_path_geometry(start, goal, env.obstacles)
        episodes.append({
            "index": i, "states": states, "actions": actions, "start": start, "goal": goal,
            "outcome": outcome, "steps": steps, "corridor_clearance": clearance,
            "occluded": clearance < 0.0,
        })

    successes = [e for e in episodes if e["outcome"] == "success"]
    failures = [e for e in episodes if e["outcome"] != "success"]
    n = len(episodes)
    print("=" * 74)
    print(f"OVERALL: {len(successes)}/{n} success ({len(successes)/n:.0%}), "
          f"{len(failures)}/{n} failure ({len(failures)/n:.0%})")
    for k in ("collision", "out_of_bounds", "timeout"):
        c = sum(1 for e in episodes if e["outcome"] == k)
        print(f"    {k:14s}: {c:3d} ({c/n:.0%})")

    # (1) + (3) clustering by straight-line occlusion --------------------
    occ = [e for e in episodes if e["occluded"]]
    clr = [e for e in episodes if not e["occluded"]]
    print("\n" + "=" * 74)
    print("(1)/(3) FAILURE RATE BY STRAIGHT-LINE GEOMETRY")
    print("=" * 74)

    def fail_rate(group):
        if not group:
            return 0, 0, 0.0
        f = sum(1 for e in group if e["outcome"] != "success")
        return f, len(group), f / len(group)

    fo, no, ro = fail_rate(occ)
    fc, nc, rc = fail_rate(clr)
    print(f"  occluded straight path (goal behind an obstacle): "
          f"{fo}/{no} fail ({ro:.0%})")
    print(f"  clear straight path (unobstructed line to goal) : "
          f"{fc}/{nc} fail ({rc:.0%})")
    if failures:
        occ_fail = sum(1 for e in failures if e["occluded"])
        print(f"\n  of the {len(failures)} FAILURES: {occ_fail} ({occ_fail/len(failures):.0%}) have an "
              f"occluded straight path, {len(failures)-occ_fail} "
              f"({(len(failures)-occ_fail)/len(failures):.0%}) have a CLEAR straight path but still fail")
    # breakdown of failure TYPE within each geometry
    print("\n  failure type by geometry:")
    for label, grp in (("occluded", occ), ("clear", clr)):
        row = {k: sum(1 for e in grp if e["outcome"] == k) for k in ("collision", "out_of_bounds", "timeout")}
        print(f"    {label:9s}: collision={row['collision']:2d}  out_of_bounds={row['out_of_bounds']:2d}  "
              f"timeout={row['timeout']:2d}")

    # corridor-clearance distribution: successes vs failures
    def stats(group):
        vals = np.array([e["corridor_clearance"] for e in group]) if group else np.array([0.0])
        return vals.mean(), np.median(vals), (vals < 0).mean()
    sm, smd, socc = stats(successes)
    fm, fmd, focc = stats(failures)
    print("\n  corridor clearance (m; negative = straight line blocked):")
    print(f"    successes: mean={sm:+.2f} median={smd:+.2f}  ({socc:.0%} occluded)")
    print(f"    failures : mean={fm:+.2f} median={fmd:+.2f}  ({focc:.0%} occluded)")

    # (2) where timeouts end up -----------------------------------------
    timeouts = [e for e in failures if e["outcome"] == "timeout"]
    print("\n" + "=" * 74)
    print(f"(2) WHERE {len(timeouts)} TIMEOUTS END UP")
    print("=" * 74)
    cats = {}
    for ep in timeouts:
        cat = classify_timeout(ep, env)
        cats[cat] = cats.get(cat, 0) + 1
    for cat, c in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:3d}  {cat}")
    if timeouts:
        print("\n  per-episode detail:")
        for ep in sorted(timeouts, key=lambda e: e["timeout_cat"]):
            print(f"    ep{ep['index']:3d}: occluded={str(ep['occluded']):5s}  "
                  f"closest_to_goal={ep['closest_dist']:4.2f}m@{ep['closest_speed']:.2f}m/s  "
                  f"final: goal={ep['final_goal_dist']:4.2f}m obstacle={ep['final_obs_dist']:4.2f}m  "
                  f"-> {ep['timeout_cat']}")

    # ---- representative GIFs: pick for variety -------------------------
    picks, seen = [], set()

    def add(ep, tag):
        if ep is not None and ep["index"] not in seen:
            seen.add(ep["index"])
            ep["_tag"] = tag
            picks.append(ep)

    # a) occluded-path failure (goal behind an obstacle)
    add(next((e for e in failures if e["occluded"]), None), "occluded_goal")
    # b) a timeout parked near an obstacle it routed around
    add(next((e for e in timeouts if e.get("timeout_cat") == "parked near obstacle"), None), "timeout_parked")
    # c) a CLEAR-path failure (surprising: unobstructed but still fails)
    add(next((e for e in failures if not e["occluded"]), None), "clear_path_fail")
    # fill up to 3 from remaining failures if needed
    for e in failures:
        if len(picks) >= 3:
            break
        add(e, e["outcome"])

    print("\n" + "=" * 74)
    print(f"saving {len(picks[:3])} representative GIFs")
    print("=" * 74)
    for ep in picks[:3]:
        env.goal = ep["goal"]
        out_path = os.path.join(RESULTS_DIR, f"phase4_level2_failgeom_ep{ep['index']}_{ep['_tag']}.gif")
        title = (f"L2 fail-geom ep{ep['index']}: {ep['outcome']}, {ep['_tag']} "
                 f"(corridor={ep['corridor_clearance']:+.2f}m)")
        n_frames = render_drone_gif(env, ep["states"], ep["actions"], out_path,
                                    title=title, frame_stride=2)
        print(f"  saved {os.path.basename(out_path)}  ({n_frames} frames)  "
              f"start=({ep['start'][0]:.1f},{ep['start'][1]:.1f}) goal=({ep['goal'][0]:.1f},{ep['goal'][1]:.1f})")


if __name__ == "__main__":
    main()
