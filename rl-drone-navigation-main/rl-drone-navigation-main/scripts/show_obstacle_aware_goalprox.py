"""
Read-only: show the obstacle-aware goal-proximity bonus at two hand-set
states, to confirm the intended behaviour before training:

  (A) clean approach near the goal (nearest obstacle outside the clearance
      band) -> prox_factor ~= 1, bonus at (near) full strength.
  (B) goal near an obstacle with the drone close to that obstacle (inside the
      band) -> prox_factor strongly suppressed, so the clearance penalty
      dominates and the bonus no longer rewards diving through the obstacle.

Uses the env's own _compute_reward_and_done for the ground-truth reward and
the shared decompose() for the term-by-term split, asserting they match.
Does NOT train or modify anything.
"""

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from scripts.train_drone_curriculum import CURRICULUM
from scripts.show_reward_decomposition import decompose

E = QuadrotorEnv


def show(env, state, goal, label):
    env.state = np.array(state, dtype=float)
    env.goal = np.array(goal, dtype=float)
    env._prev_dist = env._distance_to_goal()
    dist = env._distance_to_goal()
    min_ray = float(env._lidar_scan(state[0], state[1]).min())
    los_clear = env._goal_los_clearance(state[0], state[1], env.goal)
    reward, done, info = env._compute_reward_and_done(E.HOVER_ACTION, E.HOVER_ACTION)
    terms, total = decompose(env._prev_dist, env.state, env.goal, E.HOVER_ACTION, E.HOVER_ACTION,
                             min_ray, los_clear)
    assert abs(total - reward) < 1e-6, "decomposition != env reward"

    prox_near = min(1.0, min_ray / E.LIDAR_CLEAR_THRESHOLD) ** 2
    prox_los = min(1.0, max(0.0, los_clear) / E.GOAL_LOS_MARGIN)
    prox_factor = min(prox_near, prox_los) if dist < E.GOAL_PROX_RANGE else 0.0
    full_bonus = E.W_GOAL_PROX * max(0.0, E.GOAL_PROX_RANGE - dist)
    clearance = terms["6. clearance penalty (NEW)"]
    goal_prox = terms["7. goal-proximity bonus (NEW)"]

    print("=" * 66)
    print(label)
    print("=" * 66)
    print(f"  drone=({state[0]:.2f},{state[1]:.2f})  goal=({goal[0]:.2f},{goal[1]:.2f})  "
          f"dist_to_goal={dist:.3f} m")
    print(f"  min_ray={min_ray:.3f}  ({'CLEAN, outside band' if min_ray >= E.LIDAR_CLEAR_THRESHOLD else 'inside clearance band'})")
    print(f"  prox_factor = min(1, {min_ray:.3f}/{E.LIDAR_CLEAR_THRESHOLD})^2 = {prox_factor:.3f}")
    print("  " + "-" * 62)
    for name, val in terms.items():
        tag = ""
        if name.startswith("6"):
            tag = "  <-- clearance"
        elif name.startswith("7"):
            tag = "  <-- goal-prox (obstacle-aware)"
        print(f"  {name:<32} {val:+10.5f}{tag}")
    print("  " + "-" * 62)
    print(f"  {'TOTAL per-step reward':<32} {total:+10.5f}   (env: {reward:+.5f})")
    print(f"\n  goal-prox: full={full_bonus:+.5f} -> after suppression={goal_prox:+.5f} "
          f"({prox_factor*100:.0f}% of full)")
    if min_ray < E.LIDAR_CLEAR_THRESHOLD:
        winner = "clearance DOMINATES" if abs(clearance) > abs(goal_prox) else "goal-prox still wins (!)"
        print(f"  |clearance|={abs(clearance):.5f}  vs  |goal-prox|={abs(goal_prox):.5f}  ->  {winner}")
    print()


def main():
    level2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")
    print(f"W_GOAL_PROX={E.W_GOAL_PROX}  GOAL_PROX_RANGE={E.GOAL_PROX_RANGE}  "
          f"W_CLEARANCE={E.W_CLEARANCE}  LIDAR_CLEAR_THRESHOLD={E.LIDAR_CLEAR_THRESHOLD}\n")

    # (A) clean approach near goal: open corner, both obstacles far away.
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=0)
    show(env, [1.0, 7.0, 0.0, 0.0, 0.0, 0.0], [1.4, 7.0],
         "(A) CLEAN APPROACH near goal (obstacles far) -- bonus should be ~FULL")

    # (B) goal near an obstacle, drone hugging that obstacle (inside the band).
    show(env, [5.2, 6.0, 0.0, 0.0, 0.0, 0.0], [5.2, 6.4],
         "(B) GOAL BESIDE OBSTACLE, drone close to it -- bonus should be SUPPRESSED")


if __name__ == "__main__":
    main()
