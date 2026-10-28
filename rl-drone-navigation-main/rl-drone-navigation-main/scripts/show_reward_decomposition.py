"""
Read-only: show the full reward decomposition (including the new
obstacle-clearance term) for a real near-obstacle state pulled from a
collision episode under the current policy. Does NOT train and does NOT
modify anything.

For each per-step reward term it recomputes the exact same arithmetic the
env uses, then asserts the six components sum to the reward the env
actually returned for that step -- so the decomposition is verified, not
just plausible.
"""

import os

import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from scripts.train_drone_curriculum import CURRICULUM, ENTROPY_COEF, SEED

CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "model_checkpoints",
    "phase4_level2_wide_gaps_cont_update0150.pt",
)
LEVEL2 = next(c for c in CURRICULUM if c["name"] == "level2_wide_gaps")
E = QuadrotorEnv  # constants shortcut


def decompose(prev_dist, state, goal, left, right, min_ray, los_clear=float("inf")):
    """Mirror QuadrotorEnv._compute_reward_and_done, term by term (no terminal).
    los_clear is the drone->goal corridor clearance (env._goal_los_clearance);
    default +inf means an unobstructed line of sight."""
    x, z, vx, vz, theta, omega = state
    dist = float(np.hypot(x - goal[0], z - goal[1]))

    progress = E.W_PROGRESS * (prev_dist - dist)
    time_pen = -E.W_TIME
    effort = -E.W_EFFORT * (left ** 2 + right ** 2)
    if dist > 1e-6:
        vel_toward = vx * (goal[0] - x) / dist + vz * (goal[1] - z) / dist
    else:
        vel_toward = 0.0
    velocity = E.W_VELOCITY * vel_toward
    distance = -E.W_DIST * dist
    clearance = -E.W_CLEARANCE * max(0.0, E.LIDAR_CLEAR_THRESHOLD - min_ray)
    if dist < E.GOAL_PROX_RANGE:
        prox_near = min(1.0, min_ray / E.LIDAR_CLEAR_THRESHOLD) ** 2
        prox_los = min(1.0, max(0.0, los_clear) / E.GOAL_LOS_MARGIN)
        prox_factor = min(prox_near, prox_los)
        goal_prox = E.W_GOAL_PROX * (E.GOAL_PROX_RANGE - dist) * prox_factor
    else:
        goal_prox = 0.0

    return {
        "1. progress (toward goal)": progress,
        "2. time penalty": time_pen,
        "3. control effort": effort,
        "4. velocity-toward-goal": velocity,
        "5. distance penalty": distance,
        "6. clearance penalty (NEW)": clearance,
        "7. goal-proximity bonus (NEW)": goal_prox,
    }, progress + time_pen + effort + velocity + distance + clearance + goal_prox


def main():
    agent = PPOAgent(
        obs_dim=E.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[E.HOVER_ACTION, E.HOVER_ACTION],
        config=PPOConfig(entropy_coef=ENTROPY_COEF),
    )
    agent.load(CHECKPOINT)
    env = QuadrotorEnv(obstacles=LEVEL2["obstacles"], seed=SEED + 1000)
    print(f"policy: {os.path.basename(CHECKPOINT)}   "
          f"W_CLEARANCE={E.W_CLEARANCE}  LIDAR_CLEAR_THRESHOLD={E.LIDAR_CLEAR_THRESHOLD} "
          f"(=> {E.LIDAR_CLEAR_THRESHOLD * E.LIDAR_MAX_RANGE:.1f} m)\n")

    # Find a collision episode and record every RUNNING step's inputs+reward.
    for ep in range(60):
        obs = env.reset()
        goal = env.goal.copy()
        records = []
        done = False
        outcome = "running"
        while not done:
            prev_dist = float(env._prev_dist)          # what reward will use
            action = agent.act_deterministic(obs)
            left, right = np.clip(action, 0.0, 1.0)
            obs, reward, done, info = env.step(action)
            outcome = info["outcome"]
            state = env.state.copy()
            min_ray = float(env._lidar_scan(state[0], state[1]).min())
            records.append({
                "prev_dist": prev_dist, "state": state, "left": float(left),
                "right": float(right), "min_ray": min_ray, "reward": float(reward),
                "outcome": outcome,
            })
        if outcome == "collision":
            break
    else:
        print("no collision episode found in 60 tries")
        return

    running = [r for r in records if r["outcome"] == "running"]
    target = running[-1]  # last running step before the terminal collision -- closest approach

    # ---- approach table: clearance term growing over the final steps ----
    print("APPROACH TO COLLISION (final running steps; clearance term switches on below "
          f"min_ray={E.LIDAR_CLEAR_THRESHOLD}):")
    print(f"  {'x':>5} {'z':>5} | {'min_ray':>7} | {'clearance_pen':>13} | {'progress':>9} | {'step_reward':>11}")
    for r in running[-7:]:
        terms, _tot = decompose(r["prev_dist"], r["state"], goal, r["left"], r["right"], r["min_ray"],
                                env._goal_los_clearance(r["state"][0], r["state"][1], goal))
        print(f"  {r['state'][0]:5.2f} {r['state'][1]:5.2f} | {r['min_ray']:7.3f} | "
              f"{terms['6. clearance penalty (NEW)']:13.5f} | {terms['1. progress (toward goal)']:9.4f} | "
              f"{r['reward']:11.4f}")

    # ---- full decomposition of the chosen near-obstacle state ----
    terms, total = decompose(
        target["prev_dist"], target["state"], goal, target["left"], target["right"], target["min_ray"],
        env._goal_los_clearance(target["state"][0], target["state"][1], goal),
    )
    x, z = target["state"][0], target["state"][1]
    print("\n" + "=" * 66)
    print("REWARD DECOMPOSITION -- near-obstacle state (last step before impact)")
    print("=" * 66)
    print(f"  drone pos: (x={x:.2f}, z={z:.2f})   goal: (x={goal[0]:.2f}, z={goal[1]:.2f})")
    print(f"  min lidar ray: {target['min_ray']:.3f}  "
          f"(=> nearest obstacle ~{target['min_ray'] * E.LIDAR_MAX_RANGE:.2f} m away, "
          f"{'BELOW' if target['min_ray'] < E.LIDAR_CLEAR_THRESHOLD else 'above'} threshold)")
    print("  " + "-" * 62)
    for name, val in terms.items():
        bar = "  <-- NEW" if name.startswith("6") else ""
        print(f"  {name:<32} {val:+10.5f}{bar}")
    print("  " + "-" * 62)
    print(f"  {'TOTAL per-step reward':<32} {total:+10.5f}")
    print(f"  {'(env-returned reward, verify)':<32} {target['reward']:+10.5f}")
    assert abs(total - target["reward"]) < 1e-5, "decomposition does NOT match env reward!"
    print("  [OK] decomposition matches env-returned reward exactly")

    # ---- magnitude sanity check vs the anchors the user cared about ----
    clr = terms["6. clearance penalty (NEW)"]
    print("\nMAGNITUDE CHECK (is it a nudge, not a wall?):")
    print(f"  clearance penalty at this near-collision : {clr:+.5f}")
    print(f"  per-step time penalty (target magnitude) : {-E.W_TIME:+.5f}   "
          f"ratio {abs(clr) / E.W_TIME:.2f}x")
    print(f"  this step's progress term                : {terms['1. progress (toward goal)']:+.5f}   "
          f"(shaping must not swamp this)")
    print(f"  terminal bonus / crash                   : {E.R_GOAL:+.1f} / {E.R_CRASH:+.1f}   "
          f"(clearance is ~{abs(clr) / E.R_GOAL * 100:.3f}% of a terminal)")
    max_pen = E.W_CLEARANCE * E.LIDAR_CLEAR_THRESHOLD
    print(f"  theoretical MAX clearance penalty (ray=0): {-max_pen:+.5f}   "
          f"({max_pen / E.W_TIME:.2f}x the time penalty)")


if __name__ == "__main__":
    main()
