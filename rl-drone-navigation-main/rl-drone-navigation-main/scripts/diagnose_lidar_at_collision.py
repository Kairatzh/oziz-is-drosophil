"""
Read-only diagnostic (changes nothing): confirm whether the lidar is
actually sensing the obstacle at collision time.

Runs deterministic eval episodes on Level 2 with the current trained
policy until it finds collisions, then prints the raw 8 lidar ray values
for the last 5 steps before the terminal collision -- alongside the TRUE
surface clearance to the nearest obstacle (ground truth) so we can tell
whether the sensor agrees with reality.

Interpretation:
  - lidar reading is normalized: 1.0 = clear / nothing within LIDAR_MAX_RANGE,
    0.0 = obstacle right at the drone. Obstacle-facing rays should trend
    toward ~0 as the drone closes in.
  - if the obstacle-facing rays read ~0 just before collision -> the agent
    HAS the information and ignores it (reward problem).
  - if they stay ~1.0 (clear) right up to collision -> the lidar wiring is
    broken (the real bug).
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
N_LAST_STEPS = 5
MAX_EPISODES = 60
COLLISION_EPISODES_TO_SHOW = 3


def true_surface_clearance(env, x, z):
    """Min over obstacles of surface distance (same metric the collision
    check uses: collision fires when this drops to <= DRONE_RADIUS)."""
    if not env.obstacles:
        return float("inf")
    return min(obs.distance_to_point(x, z) for obs in env.obstacles)


def main():
    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=ENTROPY_COEF),
    )
    agent.load(CHECKPOINT)
    print(f"loaded policy: {os.path.basename(CHECKPOINT)}")

    # Same eval distribution as run_evaluation (seed = SEED + 1000).
    env = QuadrotorEnv(obstacles=LEVEL2["obstacles"], seed=SEED + 1000)
    print(f"Level 2 obstacles: "
          + ", ".join(f"(x={o.x}, z={o.z}, r={o.radius})" for o in env.obstacles))
    print(f"DRONE_RADIUS={QuadrotorEnv.DRONE_RADIUS}  LIDAR_MAX_RANGE={QuadrotorEnv.LIDAR_MAX_RANGE}m  "
          f"(collision fires when true clearance <= {QuadrotorEnv.DRONE_RADIUS})\n")

    shown = 0
    for ep in range(MAX_EPISODES):
        obs = env.reset()
        done = False
        history = []  # rolling record of recent steps
        outcome = "running"
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, done, info = env.step(action)
            outcome = info["outcome"]
            x, z = env.state[0], env.state[1]
            lidar = obs[8:]  # last LIDAR_NUM_RAYS entries
            history.append({
                "step": env.steps,
                "x": x, "z": z,
                "lidar": np.array(lidar, dtype=float),
                "clear": true_surface_clearance(env, x, z),
            })

        if outcome != "collision":
            continue

        shown += 1
        print("=" * 96)
        print(f"COLLISION episode (env episode #{ep}) -- last {N_LAST_STEPS} steps before terminal")
        print("=" * 96)
        header = ("  step | " + " ".join(f"r{i}" + ("↑" if i == 2 else "→" if i == 0 else " ")
                                          for i in range(QuadrotorEnv.LIDAR_NUM_RAYS)))
        # Simpler, aligned header:
        ray_cols = " ".join(f"ray{i}" for i in range(QuadrotorEnv.LIDAR_NUM_RAYS))
        print(f"  {'step':>4} | {'pos (x,z)':>14} | {'true_clear':>10} | min_ray | {ray_cols}")
        print("  " + "-" * 92)
        for h in history[-N_LAST_STEPS:]:
            rays = " ".join(f"{v:4.2f}" for v in h["lidar"])
            print(f"  {h['step']:>4} | ({h['x']:5.2f},{h['z']:5.2f}) | {h['clear']:>10.3f} | "
                  f"{h['lidar'].min():6.2f}  | {rays}")
        # Which ray SHOULD be lit: direction from drone to nearest obstacle center.
        last = history[-1]
        nearest = min(env.obstacles, key=lambda o: o.distance_to_point(last["x"], last["z"]))
        ang = np.arctan2(nearest.z - last["z"], nearest.x - last["x"]) % (2 * np.pi)
        expected_ray = int(round(ang / (2 * np.pi / QuadrotorEnv.LIDAR_NUM_RAYS))) % QuadrotorEnv.LIDAR_NUM_RAYS
        print(f"  -> nearest obstacle at (x={nearest.x}, z={nearest.z}); direction ~{np.degrees(ang):.0f} deg "
              f"=> obstacle-facing ray is approx ray{expected_ray}")
        print(f"     ray{expected_ray} reading over those steps: "
              + " ".join(f"{h['lidar'][expected_ray]:.2f}" for h in history[-N_LAST_STEPS:]))
        print()

        if shown >= COLLISION_EPISODES_TO_SHOW:
            break

    if shown == 0:
        print(f"No collision episodes in {MAX_EPISODES} eval episodes -- try more episodes.")


if __name__ == "__main__":
    main()
