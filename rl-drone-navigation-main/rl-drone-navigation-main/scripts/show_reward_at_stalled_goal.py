"""
Read-only: find an open-space "stalled just outside the goal" timeout under
the update-200 policy and print the per-step reward decomposition there, to
quantify how weak the pull-to-goal is in the last ~0.5 m. Open-space is
chosen deliberately (nearest obstacle > 1.5 m) so the clearance term is off
and we see the goal-seeking terms in isolation. Does NOT train or modify
anything.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.train_drone_curriculum import CURRICULUM
from scripts.show_reward_decomposition import decompose  # reuse the verified term-by-term math

E = QuadrotorEnv
CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints",
                          "phase4_level2_wide_gaps_cont_update0200.pt")
EVAL_SEED = 1000


def main():
    level2 = next(l for l in CURRICULUM if l["name"] == "level2_wide_gaps")
    env = QuadrotorEnv(obstacles=level2["obstacles"], seed=EVAL_SEED)
    agent = PPOAgent(obs_dim=E.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)

    # Find an open-space stall: a timeout whose final state is ~0.3-0.6 m
    # from goal, low speed, with the nearest obstacle far away.
    chosen = None
    for ep in range(100):
        obs = env.reset()
        goal = env.goal.copy()
        recs = []
        done = False
        outcome = "running"
        while not done:
            prev_dist = float(env._prev_dist)
            action = agent.act_deterministic(obs)
            left, right = np.clip(action, 0.0, 1.0)
            obs, reward, done, info = env.step(action)
            outcome = info["outcome"]
            s = env.state.copy()
            min_ray = float(env._lidar_scan(s[0], s[1]).min())
            obs_dist = min((o.distance_to_point(s[0], s[1]) for o in env.obstacles), default=9.9)
            recs.append({"prev_dist": prev_dist, "state": s, "left": float(left),
                         "right": float(right), "min_ray": min_ray, "reward": float(reward),
                         "obs_dist": obs_dist})
        if outcome != "timeout":
            continue
        final = recs[-1]
        d = float(np.hypot(final["state"][0] - goal[0], final["state"][1] - goal[1]))
        spd = float(np.hypot(final["state"][2], final["state"][3]))
        if 0.3 <= d <= 0.6 and spd < 0.3 and final["obs_dist"] > 1.5:
            chosen = {"ep": ep, "goal": goal, "recs": recs, "final_dist": d, "final_speed": spd}
            break

    if chosen is None:
        print("no matching open-space stall found")
        return

    goal = chosen["goal"]
    recs = chosen["recs"]
    print(f"open-space stall: eval episode #{chosen['ep']}  "
          f"final dist_to_goal={chosen['final_dist']:.3f} m  speed={chosen['final_speed']:.3f} m/s  "
          f"(GOAL_RADIUS={E.GOAL_RADIUS}, so {chosen['final_dist'] - E.GOAL_RADIUS:.3f} m short)\n")

    # ---- hover table: last 8 steps, showing position barely changes and the
    # per-step reward is essentially just the constant time penalty ----
    print("LAST 8 STEPS (drone hovering outside the goal):")
    print(f"  {'x':>5} {'z':>5} | {'dist':>5} {'speed':>5} | {'progress':>9} {'velocity':>9} "
          f"{'dist_pen':>9} | {'step_reward':>11}")
    for r in recs[-8:]:
        terms, _ = decompose(r["prev_dist"], r["state"], goal, r["left"], r["right"], r["min_ray"],
                             env._goal_los_clearance(r["state"][0], r["state"][1], goal))
        s = r["state"]
        d = float(np.hypot(s[0] - goal[0], s[1] - goal[1]))
        spd = float(np.hypot(s[2], s[3]))
        print(f"  {s[0]:5.2f} {s[1]:5.2f} | {d:5.2f} {spd:5.2f} | "
              f"{terms['1. progress (toward goal)']:+9.5f} {terms['4. velocity-toward-goal']:+9.5f} "
              f"{terms['5. distance penalty']:+9.5f} | {r['reward']:+11.5f}")

    # ---- full decomposition of the final stalled step ----
    r = recs[-1]
    terms, total = decompose(r["prev_dist"], r["state"], goal, r["left"], r["right"], r["min_ray"],
                             env._goal_los_clearance(r["state"][0], r["state"][1], goal))
    print("\n" + "=" * 60)
    print("REWARD DECOMPOSITION at the stalled state")
    print("=" * 60)
    d = float(np.hypot(r["state"][0] - goal[0], r["state"][1] - goal[1]))
    print(f"  dist_to_goal={d:.3f} m   speed={np.hypot(r['state'][2], r['state'][3]):.3f} m/s   "
          f"min_ray={r['min_ray']:.2f} (clearance term {'OFF' if r['min_ray'] >= E.LIDAR_CLEAR_THRESHOLD else 'ON'})")
    print("  " + "-" * 56)
    for name, val in terms.items():
        print(f"  {name:<32} {val:+10.5f}")
    print("  " + "-" * 56)
    print(f"  {'TOTAL per-step reward':<32} {total:+10.5f}")
    print(f"  {'(env-returned, verify)':<32} {r['reward']:+10.5f}")
    assert abs(total - r["reward"]) < 1e-5

    # ---- how weak is the position signal? ----
    print("\nHOW WEAK IS THE PULL-TO-GOAL HERE?")
    print(f"  distance penalty is LINEAR (-W_DIST*dist), so its gradient is constant = -W_DIST = {-E.W_DIST:+.4f}/m.")
    print(f"    -> moving the final {d:.2f} m fully into the goal improves per-step reward by only "
          f"{E.W_DIST * d:+.5f} (i.e. {E.W_DIST*d:.5f}).")
    print(f"  progress+velocity terms are ZERO at zero speed -- they only pay out while MOVING,")
    print(f"    so a stationary drone feels no local push to start closing the gap.")
    print(f"  the dominant per-step term is the constant time penalty ({-E.W_TIME:+.4f}), which is")
    print(f"    position-independent: hovering here vs 0.1 m closer costs essentially the same.")
    print(f"  the ONLY large signal (the +{E.R_GOAL:.0f} success bonus) sits behind the position")
    print(f"    gate dist<= {E.GOAL_RADIUS} AND speed<= {E.GOAL_SPEED_LIMIT}, and is invisible to the dense")
    print(f"    reward until that gate is crossed.")


if __name__ == "__main__":
    main()
