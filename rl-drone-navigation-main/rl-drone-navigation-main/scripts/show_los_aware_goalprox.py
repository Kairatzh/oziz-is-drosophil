"""
Read-only: show the line-of-sight-aware goal-proximity bonus.
  (1) clear line of sight near the goal -> bonus at full strength.
  (2) goal occluded, drone still at MODERATE range from the obstacle (min_ray
      >= threshold, so proximity alone would NOT suppress) -> the LOS factor
      already suppresses the bonus, so clearance/avoidance wins.
Plus an approach sweep toward an occluded goal, showing the LOS factor is ~0
throughout while the proximity factor only reacts late -- i.e. LOS damps the
pull from the start. Uses the env's own reward + shared decompose(); asserts
they match. Does NOT train or modify anything.
"""

import numpy as np

from env.quadrotor_env import QuadrotorEnv, CircleObstacle
from scripts.train_drone_curriculum import CURRICULUM
from scripts.show_reward_decomposition import decompose

E = QuadrotorEnv


def factors(env, x, z, dist, min_ray):
    los_clear = env._goal_los_clearance(x, z, env.goal)
    prox_near = min(1.0, min_ray / E.LIDAR_CLEAR_THRESHOLD) ** 2
    prox_los = min(1.0, max(0.0, los_clear) / E.GOAL_LOS_MARGIN)
    combined = min(prox_near, prox_los) if dist < E.GOAL_PROX_RANGE else 0.0
    return los_clear, prox_near, prox_los, combined


def show(env, state, goal, label):
    env.state = np.array(state, dtype=float)
    env.goal = np.array(goal, dtype=float)
    env._prev_dist = env._distance_to_goal()
    dist = env._distance_to_goal()
    min_ray = float(env._lidar_scan(state[0], state[1]).min())
    los_clear, prox_near, prox_los, combined = factors(env, state[0], state[1], dist, min_ray)
    reward, _, _ = env._compute_reward_and_done(E.HOVER_ACTION, E.HOVER_ACTION)
    terms, total = decompose(env._prev_dist, env.state, env.goal, E.HOVER_ACTION, E.HOVER_ACTION,
                             min_ray, los_clear)
    assert abs(total - reward) < 1e-6

    print("=" * 70)
    print(label)
    print("=" * 70)
    print(f"  drone=({state[0]:.2f},{state[1]:.2f})  goal=({goal[0]:.2f},{goal[1]:.2f})  dist={dist:.3f} m")
    print(f"  min_ray={min_ray:.3f} ({'outside' if min_ray >= E.LIDAR_CLEAR_THRESHOLD else 'inside'} band)   "
          f"los_clearance={los_clear:+.3f} m ({'CLEAR' if los_clear >= 0 else 'OCCLUDED'})")
    print(f"  prox_near(min_ray)={prox_near:.3f}   prox_los={prox_los:.3f}   "
          f"=> prox_factor=min(...)={combined:.3f}")
    print("  " + "-" * 66)
    for name, val in terms.items():
        tag = "  <-- goal-prox" if name.startswith("7") else ("  <-- clearance" if name.startswith("6") else "")
        print(f"  {name:<32} {val:+10.5f}{tag}")
    print("  " + "-" * 66)
    print(f"  {'TOTAL per-step reward':<32} {total:+10.5f}   (env: {reward:+.5f})")
    full = E.W_GOAL_PROX * max(0.0, E.GOAL_PROX_RANGE - dist)
    prox_only = full * prox_near   # what the proximity-only version would have given
    print(f"\n  goal-prox: full={full:+.5f}  proximity-only={prox_only:+.5f}  "
          f"LOS-aware={terms['7. goal-proximity bonus (NEW)']:+.5f}")
    print()


def main():
    level2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")
    print(f"W_GOAL_PROX={E.W_GOAL_PROX}  GOAL_PROX_RANGE={E.GOAL_PROX_RANGE}  "
          f"GOAL_LOS_MARGIN={E.GOAL_LOS_MARGIN}  LIDAR_CLEAR_THRESHOLD={E.LIDAR_CLEAR_THRESHOLD}\n")

    # (1) clear line of sight near the goal (open corner of the Level 2 map).
    env2 = QuadrotorEnv(obstacles=level2["obstacles"], seed=0)
    show(env2, [2.0, 7.0, 0.0, 0.0, 0.0, 0.0], [2.4, 7.0],
         "(1) CLEAR LINE OF SIGHT near goal -- bonus should be FULL")

    # (2) occluded goal, drone at moderate range (small obstacle on the line,
    # ~1.0 m ahead so min_ray stays above the clearance threshold).
    env_occ = QuadrotorEnv(obstacles=[CircleObstacle(x=6.3, z=4.0, radius=0.3)], seed=0)
    show(env_occ, [5.0, 4.0, 0.0, 0.0, 0.0, 0.0], [6.45, 4.0],
         "(2) OCCLUDED goal, drone at MODERATE range -- LOS should SUPPRESS "
         "where proximity would not")

    # Approach sweep toward the occluded goal: LOS factor stays ~0 the whole
    # way in, while the proximity factor only drops late.
    print("=" * 70)
    print("APPROACH TO AN OCCLUDED GOAL  (obstacle (6.3,4.0) r0.3, goal (6.45,4.0))")
    print("=" * 70)
    print(f"  {'drone_x':>7} {'dist':>5} {'min_ray':>7} | {'prox_near':>9} {'prox_los':>8} {'combined':>8}")
    goal = np.array([6.45, 4.0])
    env_occ.goal = goal
    for dx in np.arange(5.0, 5.85, 0.15):
        min_ray = float(env_occ._lidar_scan(dx, 4.0).min())
        dist = float(np.hypot(dx - goal[0], 4.0 - goal[1]))
        _, prox_near, prox_los, combined = factors(env_occ, dx, 4.0, dist, min_ray)
        print(f"  {dx:7.2f} {dist:5.2f} {min_ray:7.3f} | {prox_near:9.3f} {prox_los:8.3f} {combined:8.3f}")
    print("\n  -> prox_los (and therefore the combined factor) is ~0 across the whole")
    print("     approach; the proximity factor alone would stay high until the last")
    print("     ~0.3 m. LOS damps the pull toward the occluded goal from the start.")


if __name__ == "__main__":
    main()
