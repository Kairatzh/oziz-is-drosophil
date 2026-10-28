"""
Level 3 failure-geometry diagnosis at cumulative update 100 (5-obstacle
moderate-gap field, warm-started from the accepted Level 2 checkpoint).
Success has plateaued ~37-38% with collision ~44% -- see memory
level3-progress. This asks WHERE the failures live, geometrically, same
method as diagnose_level2_failure_geometry.py generalized from 1-2 obstacles
to Level 3's fixed 5-obstacle field:

  (1) Are failures clustered on episodes whose straight start->goal line is
      blocked by an obstacle (occluded) vs a clear line that still fails?
  (2) For timeouts, where does the agent end up (overshoot / parked near an
      obstacle / hovering near goal / stuck mid-field)?
  (3) With 5 FIXED obstacles (unlike Level 2's 1-2), do failures cluster on
      specific obstacles/gaps in the field, or are they spread evenly across
      all five? -> for every episode we identify the "tightest" obstacle on
      the straight path (the one that would block it, or the closest call if
      clear) and, separately for collisions, the obstacle nearest the final
      (crash) position. We compare each obstacle's share of episodes vs its
      share of failures to see if any one obstacle/gap is disproportionately
      responsible.

Deterministic eval only, on the same distribution training used
(EVAL_SEED=1000). Does NOT train and does NOT modify anything. Saves 2-3
representative GIFs.
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
    "phase4_level3_from_level2_update0100.pt",
)
N_EPISODES = 100         # matches the 100-ep decision-eval sample size
EVAL_SEED = 1000         # matches the decision_env seed used during training
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
    """For the straight start->goal segment, find the TIGHTEST obstacle
    (smallest corridor clearance) among all of them. Returns
    (corridor_clearance, obstacle_index). corridor_clearance = distance from
    the segment to the obstacle SURFACE (seg-to-center minus radius):
      < 0  -> the straight line passes THROUGH that disk (goal occluded)
      >=0  -> straight line clears every disk by at least that many metres.
    """
    tightest_idx = None
    best = float("inf")
    for i, o in enumerate(obstacles):
        seg_d = point_segment_distance(np.array([o.x, o.z]), start, goal)
        clearance = seg_d - o.radius
        if clearance < best:
            best, tightest_idx = clearance, i
    return best, tightest_idx


def nearest_obstacle_index(p, obstacles):
    if not obstacles:
        return None, float("inf")
    dists = [o.distance_to_point(p[0], p[1]) for o in obstacles]
    i = int(np.argmin(dists))
    return i, dists[i]


def obstacle_label(i, obstacles):
    o = obstacles[i]
    return f"obs{i}({o.x:.1f},{o.z:.1f},r={o.radius:.1f})"


def classify_timeout(ep, env):
    """Where did a timeout end up?"""
    states, goal = ep["states"], ep["goal"]
    dists = np.linalg.norm(states[:, 0:2] - goal[None, :], axis=1)
    speeds = np.linalg.norm(states[:, 2:4], axis=1)
    ci = int(np.argmin(dists))
    closest_dist, closest_speed = float(dists[ci]), float(speeds[ci])
    final_pos = states[-1, 0:2]
    final_goal_dist = float(dists[-1])
    final_obs_idx, final_obs_dist = nearest_obstacle_index(final_pos, env.obstacles)

    if closest_dist <= env.GOAL_RADIUS and closest_speed > env.GOAL_SPEED_LIMIT:
        cat = "overshoot (hit goal radius too fast)"
    elif final_obs_dist <= CLEAR_BAND:
        cat = "parked near obstacle"
    elif final_goal_dist <= 3 * env.GOAL_RADIUS:
        cat = "hovering near goal (never settled in)"
    else:
        cat = "stuck mid-field"
    ep.update(closest_dist=closest_dist, closest_speed=closest_speed,
              final_goal_dist=final_goal_dist, final_obs_dist=final_obs_dist,
              final_obs_idx=final_obs_idx, timeout_cat=cat)
    return cat


def main():
    level3 = next(l for l in CURRICULUM if l["name"] == "level3_moderate_gaps")
    env = QuadrotorEnv(obstacles=level3["obstacles"], seed=EVAL_SEED)
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT_PATH)
    print(f"loaded {os.path.basename(CHECKPOINT_PATH)}   ({N_EPISODES} eval-distribution episodes)")
    print(f"obstacles: " + ", ".join(obstacle_label(i, env.obstacles) for i in range(len(env.obstacles))))
    print(f"occluded := straight start->goal line passes through an obstacle "
          f"(corridor clearance < 0); clear band = {CLEAR_BAND:.1f} m\n")

    episodes = []
    for i in range(N_EPISODES):
        states, actions, outcome, steps, start, goal = run_episode(agent, env)
        clearance, tightest_idx = straight_path_geometry(start, goal, env.obstacles)
        crash_idx, crash_dist = nearest_obstacle_index(states[-1, 0:2], env.obstacles)
        episodes.append({
            "index": i, "states": states, "actions": actions, "start": start, "goal": goal,
            "outcome": outcome, "steps": steps, "corridor_clearance": clearance,
            "occluded": clearance < 0.0, "tightest_idx": tightest_idx,
            "crash_idx": crash_idx, "crash_dist": crash_dist,
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

    # (1) clustering by straight-line occlusion --------------------------
    occ = [e for e in episodes if e["occluded"]]
    clr = [e for e in episodes if not e["occluded"]]
    print("\n" + "=" * 74)
    print("(1) FAILURE RATE BY STRAIGHT-LINE GEOMETRY")
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
    print("\n  failure type by geometry:")
    for label, grp in (("occluded", occ), ("clear", clr)):
        row = {k: sum(1 for e in grp if e["outcome"] == k) for k in ("collision", "out_of_bounds", "timeout")}
        print(f"    {label:9s}: collision={row['collision']:2d}  out_of_bounds={row['out_of_bounds']:2d}  "
              f"timeout={row['timeout']:2d}")

    def stats(group):
        vals = np.array([e["corridor_clearance"] for e in group]) if group else np.array([0.0])
        return vals.mean(), np.median(vals), (vals < 0).mean()
    sm, smd, socc = stats(successes)
    fm, fmd, focc = stats(failures)
    print("\n  corridor clearance (m; negative = straight line blocked):")
    print(f"    successes: mean={sm:+.2f} median={smd:+.2f}  ({socc:.0%} occluded)")
    print(f"    failures : mean={fm:+.2f} median={fmd:+.2f}  ({focc:.0%} occluded)")

    # (2) where timeouts end up -------------------------------------------
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
            near = obstacle_label(ep["final_obs_idx"], env.obstacles) if ep["final_obs_idx"] is not None else "n/a"
            print(f"    ep{ep['index']:3d}: occluded={str(ep['occluded']):5s}  "
                  f"closest_to_goal={ep['closest_dist']:4.2f}m@{ep['closest_speed']:.2f}m/s  "
                  f"final: goal={ep['final_goal_dist']:4.2f}m nearest_obs={ep['final_obs_dist']:4.2f}m ({near})  "
                  f"-> {ep['timeout_cat']}")

    # (3) clustering by WHICH obstacle / gap -------------------------------
    print("\n" + "=" * 74)
    print("(3) CLUSTERING BY OBSTACLE IDENTITY (5-obstacle field)")
    print("=" * 74)
    print("  share of ALL episodes vs share of FAILURES for which each obstacle is the\n"
          "  TIGHTEST one on the straight start->goal path (uniform ~20% each => no clustering):")
    n_obs = len(env.obstacles)
    for i in range(n_obs):
        all_share = sum(1 for e in episodes if e["tightest_idx"] == i) / n
        fail_count = sum(1 for e in failures if e["tightest_idx"] == i)
        fail_share = fail_count / len(failures) if failures else 0.0
        print(f"    {obstacle_label(i, env.obstacles):24s}: all_eps={all_share:5.0%}   "
              f"failures={fail_count:3d} ({fail_share:5.0%})")

    print("\n  for COLLISIONS specifically, obstacle nearest the final (crash) position:")
    collisions = [e for e in failures if e["outcome"] == "collision"]
    for i in range(n_obs):
        c = sum(1 for e in collisions if e["crash_idx"] == i)
        share = c / len(collisions) if collisions else 0.0
        print(f"    {obstacle_label(i, env.obstacles):24s}: {c:3d} collisions ({share:5.0%})")
    if collisions:
        mean_crash_dist = np.mean([e["crash_dist"] for e in collisions])
        print(f"    mean distance from crash position to nearest obstacle surface: {mean_crash_dist:.2f} m "
              f"(~0 confirms these are genuine surface hits)")

    # ---- representative GIFs: pick for variety ---------------------------
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
        out_path = os.path.join(RESULTS_DIR, f"phase4_level3_failgeom_ep{ep['index']}_{ep['_tag']}.gif")
        title = (f"L3 fail-geom ep{ep['index']}: {ep['outcome']}, {ep['_tag']} "
                 f"(corridor={ep['corridor_clearance']:+.2f}m)")
        n_frames = render_drone_gif(env, ep["states"], ep["actions"], out_path,
                                    title=title, frame_stride=2)
        print(f"  saved {os.path.basename(out_path)}  ({n_frames} frames)  "
              f"start=({ep['start'][0]:.1f},{ep['start'][1]:.1f}) goal=({ep['goal'][0]:.1f},{ep['goal'][1]:.1f})")


if __name__ == "__main__":
    main()
