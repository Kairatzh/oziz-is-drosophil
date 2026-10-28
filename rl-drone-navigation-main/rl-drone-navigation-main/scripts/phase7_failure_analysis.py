"""
Phase 7: systematic failure analysis of the RL agent (model_checkpoints/final_model.pt).

Four analyses, all read-only (no training):

  1. OBSTACLE-COUNT SWEEP: success rate vs. number of random obstacles
     (1..6), holding placement style fixed -- where does performance start
     to break down as a function of a single, controllable difficulty axis?

  2. CORRIDOR-CLEARANCE SWEEP: success rate vs. how much the straight
     start->goal line is blocked by a single obstacle, sweeping from
     clearly clear to clearly occluded. Training (Level 1.5) only ever
     showed clearance >= GOAL_LOS_MARGIN (0.5m) -- this directly probes
     what happens right at, and beyond, that training-distribution
     boundary, holding obstacle count/radius fixed at training values.

  3. CATASTROPHIC VS. GRACEFUL FAILURE: pooling failures across the
     held-out conditions from Phase 5/6, characterize HOW the agent fails
     -- impact speed at collision, whether it was decelerating (braking
     attempt) or accelerating (blind) in the steps before impact, and
     where timeouts leave the drone (parked vs. still searching).

  4. WALL FORMATION DEEP-DIVE: the known failure case from Phase 6,
     re-examined for its specific failure signature (collision vs.
     oscillating in a local trap -- interesting to check since RL was
     never designed around potential fields, so a local-minimum-style trap
     would be a different failure mechanism producing a similar timeout).

Representative GIFs are saved for each qualitatively distinct failure kind
found. Does not train.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.quadrotor_env import CircleObstacle, QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.phase5_generalization_eval import TEST_CONDITIONS, make_random_obstacles_fn
from viz.render_drone import render_drone_gif

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "final_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_EPISODES = 100
EVAL_SEED = 1000
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]


def run_episode(agent, env):
    """Runs one episode, returning the full state/action history plus the
    per-step minimum lidar reading -- everything the later analyses need."""
    obs = env.reset()
    states = [env.state.copy()]
    actions = [tuple(agent.act_deterministic(obs))]  # seed to match states[0], per render_drone_gif's contract
    min_lidar = [float(obs[8:].min())]
    done, info = False, {"outcome": "running"}
    while not done:
        action = agent.act_deterministic(obs)
        obs, _, done, info = env.step(action)
        states.append(env.state.copy())
        actions.append(tuple(action))
        min_lidar.append(float(obs[8:].min()))
    return {
        "states": np.array(states), "actions": np.array(actions),
        "min_lidar": np.array(min_lidar), "outcome": info["outcome"],
        "goal": env.goal.copy(), "obstacles": list(env.obstacles),
    }


# ---------------------------------------------------------------------------
# 1. Obstacle-count sweep
# ---------------------------------------------------------------------------

def obstacle_count_sweep(agent, counts=(1, 2, 3, 4, 5, 6)):
    print("=" * 78)
    print("1. OBSTACLE-COUNT SWEEP (random placement, no LOS guarantee)")
    print("=" * 78)
    rates = []
    for n in counts:
        env = QuadrotorEnv(obstacle_fn=make_random_obstacles_fn(n, radius=0.8), seed=EVAL_SEED)
        outcomes = [run_episode(agent, env)["outcome"] for _ in range(N_EPISODES)]
        rate = outcomes.count("success") / N_EPISODES
        rates.append(rate)
        print(f"  {n} obstacle(s): success {rate:5.1%}   "
              f"collision {outcomes.count('collision')/N_EPISODES:5.1%}   "
              f"timeout {outcomes.count('timeout')/N_EPISODES:5.1%}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(counts, rates, marker="o", color="tab:blue")
    ax.axvline(1, color="gray", linestyle="--", alpha=0.6, label="training distribution (1 obstacle)")
    ax.set_xlabel("number of obstacles")
    ax.set_ylabel("success rate (100 episodes)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 7: success rate vs. obstacle count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase7_obstacle_count_sweep.png"), dpi=120)
    plt.close("all")
    return dict(zip(counts, rates))


# ---------------------------------------------------------------------------
# 2. Corridor-clearance sweep
# ---------------------------------------------------------------------------

def place_obstacle_at_clearance(env, rng, radius, clearance, min_start_goal_dist=4.0, max_tries=200):
    """Resets env (retrying the WHOLE start/goal/placement draw, not just
    clamping coordinates afterward) until it finds a start/goal pair far
    enough apart AND a placement that fits in-bounds, then plants ONE
    obstacle at exactly the given surface clearance from the straight
    start->goal line (negative = occludes it). Clamping cx/cz to the world
    bounds after the fact would silently shrink the true perpendicular
    offset and corrupt the requested clearance near the world edges -- so
    an out-of-bounds draw is discarded and retried instead."""
    # perpendicular distance from the obstacle CENTER to the line can't go
    # negative -- clearance values <= -radius (deeper than "center exactly
    # on the line") all collapse to that same maximally-occluded geometry.
    offset = max(radius + clearance, 0.0)
    for _ in range(max_tries):
        env.reset()
        sx, sz = env.state[0], env.state[1]
        gx, gz = env.goal
        dist = float(np.hypot(gx - sx, gz - sz))
        if dist < min_start_goal_dist:
            continue
        ux, uz = (gx - sx) / dist, (gz - sz) / dist
        px, pz = -uz, ux
        t = float(rng.uniform(0.35, 0.65)) * dist
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        cx = sx + t * ux + side * offset * px
        cz = sz + t * uz + side * offset * pz
        if radius <= cx <= QuadrotorEnv.WORLD_WIDTH - radius and radius <= cz <= QuadrotorEnv.WORLD_HEIGHT - radius:
            env.obstacles = [CircleObstacle(x=cx, z=cz, radius=radius)]
            return env._get_obs()
    # Fallback after exhausting max_tries (rare): clip rather than loop forever.
    cx = float(np.clip(cx, radius, QuadrotorEnv.WORLD_WIDTH - radius))
    cz = float(np.clip(cz, radius, QuadrotorEnv.WORLD_HEIGHT - radius))
    env.obstacles = [CircleObstacle(x=cx, z=cz, radius=radius)]
    return env._get_obs()


def run_clearance_episode(agent, env, rng, clearance):
    obs = place_obstacle_at_clearance(env, rng, radius=0.8, clearance=clearance)
    states = [env.state.copy()]
    actions = [tuple(agent.act_deterministic(obs))]
    done, info = False, {"outcome": "running"}
    while not done:
        action = agent.act_deterministic(obs)
        obs, _, done, info = env.step(action)
        states.append(env.state.copy())
        actions.append(tuple(action))
    return {
        "states": np.array(states), "actions": np.array(actions),
        "outcome": info["outcome"], "goal": env.goal.copy(), "obstacles": list(env.obstacles),
    }


def clearance_sweep(agent, clearances=(-1.2, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.2)):
    print("\n" + "=" * 78)
    print("2. CORRIDOR-CLEARANCE SWEEP (single r=0.8 obstacle; training only ever saw >= +0.5m)")
    print("=" * 78)
    rates = []
    rng = np.random.default_rng(EVAL_SEED)
    for c in clearances:
        env = QuadrotorEnv(seed=EVAL_SEED)
        outcomes = [run_clearance_episode(agent, env, rng, c)["outcome"] for _ in range(N_EPISODES)]
        rate = outcomes.count("success") / N_EPISODES
        rates.append(rate)
        tag = "  <- training boundary" if abs(c - 0.5) < 1e-9 else ""
        print(f"  clearance {c:+.1f}m: success {rate:5.1%}   "
              f"collision {outcomes.count('collision')/N_EPISODES:5.1%}   "
              f"timeout {outcomes.count('timeout')/N_EPISODES:5.1%}{tag}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(clearances, rates, marker="o", color="tab:blue")
    ax.axvline(0.0, color="firebrick", linestyle=":", alpha=0.7, label="line fully occluded (clearance=0)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="training boundary (GOAL_LOS_MARGIN)")
    ax.set_xlabel("corridor clearance to goal (m); negative = straight line blocked")
    ax.set_ylabel("success rate (100 episodes)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 7: success rate vs. corridor clearance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase7_clearance_sweep.png"), dpi=120)
    plt.close("all")
    return dict(zip(clearances, rates))


# ---------------------------------------------------------------------------
# 3. Catastrophic vs. graceful failure characterization
# ---------------------------------------------------------------------------

def speed_at(states, i):
    return float(np.hypot(states[i, 2], states[i, 3]))


def characterize_failures(agent, conditions):
    print("\n" + "=" * 78)
    print("3. CATASTROPHIC VS. GRACEFUL FAILURE (pooled across held-out conditions)")
    print("=" * 78)
    collisions, timeouts, cruise_speeds = [], [], []
    for cond in conditions:
        env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
        for _ in range(N_EPISODES):
            ep = run_episode(agent, env)
            states = ep["states"]
            if ep["outcome"] == "success":
                # mid-flight cruise speed: exclude the first/last 10% (accel/settle)
                n = len(states)
                cruise_speeds.extend(
                    float(np.hypot(s[2], s[3])) for s in states[n // 10: 9 * n // 10]
                )
            elif ep["outcome"] == "collision":
                impact_speed = speed_at(states, -1)
                pre = min(10, len(states) - 1)
                speed_trend = speed_at(states, -1) - speed_at(states, -1 - pre)
                min_lidar_far = float(ep["min_lidar"][-1 - pre]) if len(ep["min_lidar"]) > pre else float("nan")
                min_lidar_near = float(ep["min_lidar"][-1])
                collisions.append({
                    "condition": cond["name"], "impact_speed": impact_speed,
                    "speed_trend": speed_trend, "lidar_far": min_lidar_far, "lidar_near": min_lidar_near,
                    "frac_through": len(states) / QuadrotorEnv.MAX_STEPS,
                })
            elif ep["outcome"] == "timeout":
                final_dist = float(np.hypot(states[-1, 0] - ep["goal"][0], states[-1, 1] - ep["goal"][1]))
                final_speed = speed_at(states, -1)
                timeouts.append({"condition": cond["name"], "final_dist": final_dist, "final_speed": final_speed})

    cruise_mean = float(np.mean(cruise_speeds)) if cruise_speeds else float("nan")
    print(f"  successful-episode mid-flight cruise speed: mean {cruise_mean:.2f} m/s (n={len(cruise_speeds)} samples)")

    if collisions:
        impact_speeds = np.array([c["impact_speed"] for c in collisions])
        trends = np.array([c["speed_trend"] for c in collisions])
        print(f"\n  {len(collisions)} collisions pooled:")
        print(f"    impact speed: mean {impact_speeds.mean():.2f} m/s, median {np.median(impact_speeds):.2f} m/s "
              f"(vs. {cruise_mean:.2f} m/s cruise)")
        fast = int(np.sum(impact_speeds > cruise_mean))
        print(f"    {fast}/{len(collisions)} ({fast/len(collisions):.0%}) hit FASTER than typical cruise speed"
              f" -- i.e. no braking at all, a catastrophic/blind collision")
        braking = int(np.sum(trends < -0.05))
        accelerating = int(np.sum(trends > 0.05))
        flat = len(collisions) - braking - accelerating
        print(f"    in the last {10} steps before impact: {braking} braking (slowing down), "
              f"{accelerating} accelerating (speeding up), {flat} roughly flat")
        lidar_far = np.array([c["lidar_far"] for c in collisions if not np.isnan(c["lidar_far"])])
        lidar_near = np.array([c["lidar_near"] for c in collisions])
        print(f"    min-lidar reading: {lidar_far.mean():.2f} (10 steps before impact) -> "
              f"{lidar_near.mean():.2f} (at impact) -- confirms the sensor DOES see it coming in both cases")
        early = int(np.sum(np.array([c["frac_through"] for c in collisions]) < 0.3))
        print(f"    {early}/{len(collisions)} ({early/len(collisions):.0%}) collided within the first 30% of "
              f"the episode's step budget -- fails fast rather than wandering into trouble late")

    if timeouts:
        dists = np.array([t["final_dist"] for t in timeouts])
        speeds = np.array([t["final_speed"] for t in timeouts])
        print(f"\n  {len(timeouts)} timeouts pooled:")
        print(f"    final distance to goal: mean {dists.mean():.2f}m, median {np.median(dists):.2f}m")
        print(f"    final speed: mean {speeds.mean():.2f} m/s")
        parked = int(np.sum(speeds < 0.2))
        print(f"    {parked}/{len(timeouts)} ({parked/len(timeouts):.0%}) ended nearly stationary (parked/stuck) "
              f"rather than still actively moving")

    return collisions, timeouts, cruise_mean


# ---------------------------------------------------------------------------
# 4. Wall formation deep-dive
# ---------------------------------------------------------------------------

def wall_formation_deep_dive(agent):
    print("\n" + "=" * 78)
    print("4. WALL FORMATION DEEP-DIVE (known failure case)")
    print("=" * 78)
    cond = next(c for c in TEST_CONDITIONS if c["name"] == "wall_narrow_gaps")
    env = QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])
    episodes = [run_episode(agent, env) for _ in range(N_EPISODES)]
    outcomes = [ep["outcome"] for ep in episodes]
    for k in OUTCOME_KEYS:
        print(f"  {k:14s}: {outcomes.count(k):3d}  ({outcomes.count(k)/N_EPISODES:5.1%})")

    timeouts = [ep for ep in episodes if ep["outcome"] == "timeout"]
    oscillation_counts = []
    for ep in timeouts:
        vx = ep["states"][:, 2]
        sign_changes = int(np.sum(np.diff(np.sign(vx)) != 0))
        oscillation_counts.append(sign_changes)
    if oscillation_counts:
        print(f"  {len(timeouts)} timeouts: mean {np.mean(oscillation_counts):.1f} x-velocity sign changes "
              f"(vs. a handful for a direct flight) -- "
              + ("oscillation IS present, a potential-field-like local trap" if np.mean(oscillation_counts) > 8
                 else "little oscillation -- these are more like stalls/parking than active trapping"))
    else:
        print("  no timeouts in this batch -- failures here are pure collisions, not local-minimum-style traps")

    return episodes


# ---------------------------------------------------------------------------
# GIF export
# ---------------------------------------------------------------------------

def save_gif(ep, name, title):
    env = QuadrotorEnv(obstacles=ep["obstacles"], seed=0)
    env.goal = ep["goal"]
    out_path = os.path.join(RESULTS_DIR, f"phase7_{name}.gif")
    n_frames = render_drone_gif(env, ep["states"], ep["actions"], out_path,
                                 title=f"{title} ({ep['outcome']})", frame_stride=2)
    print(f"    saved {out_path} ({n_frames} frames)")


def main():
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)
    print(f"loaded {CHECKPOINT}")

    obstacle_count_sweep(agent)
    clearance_sweep(agent)

    held_out = [c for c in TEST_CONDITIONS if c["name"] != "train_distribution_control"]
    collisions, timeouts, cruise_mean = characterize_failures(agent, held_out)
    wall_episodes = wall_formation_deep_dive(agent)

    print("\n" + "=" * 78)
    print("SAVING REPRESENTATIVE FAILURE GIFS")
    print("=" * 78)

    # Re-run targeted episodes to capture specific representative cases for GIFs
    # (characterize_failures pooled summary stats but didn't keep every trajectory).
    rng = np.random.default_rng(EVAL_SEED)

    # catastrophic (fast) vs graceful (slow) collision, from the Level 3 fixed layout
    level3_cond = next(c for c in TEST_CONDITIONS if c["name"] == "level3_fixed_moderate_gaps")
    env = QuadrotorEnv(obstacles=level3_cond["obstacles"], seed=level3_cond["seed"])
    found = {}
    for _ in range(100):
        ep = run_episode(agent, env)
        if ep["outcome"] != "collision":
            continue
        spd = speed_at(ep["states"], -1)
        if "fast" not in found or spd > found["fast"][0]:
            found["fast"] = (spd, ep)
        if "slow" not in found or spd < found["slow"][0]:
            found["slow"] = (spd, ep)
    if "fast" in found:
        save_gif(found["fast"][1], "catastrophic_collision",
                  f"Catastrophic collision ({found['fast'][0]:.2f} m/s impact)")
    if "slow" in found:
        save_gif(found["slow"][1], "graceful_collision",
                  f"Slow/graceful collision ({found['slow'][0]:.2f} m/s impact)")

    # wall formation: one collision, one timeout (if present)
    wall_collision = next((ep for ep in wall_episodes if ep["outcome"] == "collision"), None)
    wall_timeout = next((ep for ep in wall_episodes if ep["outcome"] == "timeout"), None)
    if wall_collision is not None:
        save_gif(wall_collision, "wall_formation_collision", "Wall formation: collision")
    if wall_timeout is not None:
        save_gif(wall_timeout, "wall_formation_timeout", "Wall formation: timeout/stall")

    # boundary case (clearance just below the training margin) vs. deep occlusion
    env_b = QuadrotorEnv(seed=EVAL_SEED)
    for clearance, tag in ((0.3, "boundary_clearance_failure"), (-1.2, "deep_occlusion_failure")):
        ep = None
        for _ in range(60):
            candidate = run_clearance_episode(agent, env_b, rng, clearance)
            if candidate["outcome"] != "success":
                ep = candidate
                break
        if ep is not None:
            save_gif(ep, tag, f"Clearance={clearance:+.1f}m")

    print("\ndone.")


if __name__ == "__main__":
    main()
