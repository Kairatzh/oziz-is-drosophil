"""
Phase 6: RL agent vs. hand-coded potential-field baseline.

Runs the RL agent (model_checkpoints/final_model.pt) and the classical
PotentialFieldController (baseline/potential_field_controller.py) on
IDENTICAL episodes -- same test conditions as Phase 5
(scripts/phase5_generalization_eval.py), with each condition's two
environments seeded identically so both controllers face the exact same
sequence of start/goal/obstacle draws. This is a paired comparison, not two
independent samples: episode i is the same navigation problem for both.

Metrics reported per condition:
  - success / collision / out_of_bounds / timeout rate
  - time to goal (steps, successful episodes only)
  - path length (m, successful episodes only) -- trajectory smoothness proxy
  - control effort (mean left^2+right^2 per step, successful episodes only)
    -- same effort term the reward function penalizes; higher = jerkier

Read-only evaluation + visualization. Does not train.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from baseline.potential_field_controller import PotentialFieldController
from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.phase5_generalization_eval import TEST_CONDITIONS

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints", "final_model.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
N_EPISODES = 100
OUTCOME_KEYS = ["success", "collision", "out_of_bounds", "timeout"]
TRAJECTORY_PLOT_CONDITIONS = ["train_distribution_control", "level2_fixed_wide_gaps", "wall_narrow_gaps"]


def make_env(cond):
    return QuadrotorEnv(obstacles=cond["obstacles"], obstacle_fn=cond["obstacle_fn"], seed=cond["seed"])


def run_rl_episode(agent, env):
    obs = env.reset()
    states = [env.state.copy()]
    efforts = []
    done, info = False, {"outcome": "running"}
    while not done:
        action = agent.act_deterministic(obs)
        obs, _, done, info = env.step(action)
        states.append(env.state.copy())
        efforts.append(float(np.clip(action, 0.0, 1.0) ** 2 @ np.ones(2)))
    return np.array(states), info["outcome"], efforts


def run_baseline_episode(ctrl, env):
    obs = env.reset()
    states = [env.state.copy()]
    efforts = []
    done, info = False, {"outcome": "running"}
    while not done:
        x, z, vx, vz, theta, omega = env.state
        action = ctrl.act(x, z, vx, vz, theta, omega, env.goal, env.obstacles)
        obs, _, done, info = env.step(action)
        states.append(env.state.copy())
        efforts.append(float(np.clip(action, 0.0, 1.0) ** 2 @ np.ones(2)))
    return np.array(states), info["outcome"], efforts


def path_length(states):
    deltas = np.diff(states[:, :2], axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def summarize(records):
    """records: list of (outcome, n_steps, path_len, mean_effort)."""
    outcomes = [r[0] for r in records]
    counts = {k: outcomes.count(k) for k in OUTCOME_KEYS}
    success_rate = counts["success"] / len(records)
    succ = [r for r in records if r[0] == "success"]
    avg_steps = float(np.mean([r[1] for r in succ])) if succ else float("nan")
    avg_path = float(np.mean([r[2] for r in succ])) if succ else float("nan")
    avg_effort = float(np.mean([r[3] for r in succ])) if succ else float("nan")
    return {
        "success_rate": success_rate, "counts": counts,
        "avg_steps": avg_steps, "avg_path_length": avg_path, "avg_effort": avg_effort,
    }


def run_condition(agent, ctrl, cond, n_episodes=N_EPISODES):
    rl_env, bl_env = make_env(cond), make_env(cond)
    rl_records, bl_records = [], []
    for _ in range(n_episodes):
        states, outcome, efforts = run_rl_episode(agent, rl_env)
        rl_records.append((outcome, len(efforts), path_length(states), float(np.mean(efforts))))
        states, outcome, efforts = run_baseline_episode(ctrl, bl_env)
        bl_records.append((outcome, len(efforts), path_length(states), float(np.mean(efforts))))
    return summarize(rl_records), summarize(bl_records)


def plot_side_by_side(agent, ctrl, cond, episode_index, out_path):
    """Runs episode_index+1 resets on fresh, identically-seeded envs so both
    controllers face the exact same start/goal/obstacles, then plots the
    two resulting trajectories side by side."""
    rl_env, bl_env = make_env(cond), make_env(cond)
    for _ in range(episode_index + 1):
        rl_states, rl_outcome, rl_efforts = run_rl_episode(agent, rl_env)
        bl_states, bl_outcome, bl_efforts = run_baseline_episode(ctrl, bl_env)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, env, states, outcome, efforts, label in (
        (axes[0], rl_env, rl_states, rl_outcome, rl_efforts, "RL agent"),
        (axes[1], bl_env, bl_states, bl_outcome, bl_efforts, "Potential-field baseline"),
    ):
        ax.set_xlim(0, env.WORLD_WIDTH)
        ax.set_ylim(0, env.WORLD_HEIGHT)
        ax.set_aspect("equal")
        for obs in env.obstacles:
            ax.add_patch(Circle((obs.x, obs.z), obs.radius, color="firebrick", alpha=0.6))
        ax.add_patch(Circle(tuple(env.goal), env.GOAL_RADIUS, color="gold", alpha=0.6))
        ax.plot(states[:, 0], states[:, 1], color="steelblue", linewidth=1.5)
        ax.scatter([states[0, 0]], [states[0, 1]], color="black", marker="o", zorder=5, label="start")
        plen = path_length(states)
        ax.set_title(
            f"{label}: {outcome}\n{len(efforts)} steps, {plen:.1f}m path, "
            f"effort {np.mean(efforts):.3f}"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")

    fig.suptitle(f"{cond['title']} -- episode {episode_index}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close("all")


def print_condition_report(name, rl_summary, bl_summary):
    print(f"[{name}]")
    print(f"  {'':14s} {'RL agent':>12s} {'baseline':>12s}")
    print(f"  {'success':14s} {rl_summary['success_rate']:>11.1%} {bl_summary['success_rate']:>11.1%}")
    for k in ("collision", "out_of_bounds", "timeout"):
        print(f"  {k:14s} {rl_summary['counts'][k] / N_EPISODES:>11.1%} {bl_summary['counts'][k] / N_EPISODES:>11.1%}")
    print(f"  {'avg steps':14s} {rl_summary['avg_steps']:>12.1f} {bl_summary['avg_steps']:>12.1f}  (successful eps only)")
    print(f"  {'path length':14s} {rl_summary['avg_path_length']:>11.2f}m {bl_summary['avg_path_length']:>11.2f}m")
    print(f"  {'control effort':14s} {rl_summary['avg_effort']:>12.3f} {bl_summary['avg_effort']:>12.3f}")
    print()


def main():
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    agent.load(CHECKPOINT)
    ctrl = PotentialFieldController()
    print(f"loaded RL checkpoint {CHECKPOINT}\n")

    all_results = {}
    for cond in TEST_CONDITIONS:
        rl_summary, bl_summary = run_condition(agent, ctrl, cond)
        all_results[cond["name"]] = (rl_summary, bl_summary)
        print_condition_report(cond["name"], rl_summary, bl_summary)

    print("recording side-by-side trajectory visualizations...")
    for cond in TEST_CONDITIONS:
        if cond["name"] in TRAJECTORY_PLOT_CONDITIONS:
            out_path = os.path.join(RESULTS_DIR, f"phase6_{cond['name']}_side_by_side.png")
            plot_side_by_side(agent, ctrl, cond, episode_index=0, out_path=out_path)
            print(f"    saved {out_path}")

    fig, ax = plt.subplots(figsize=(11, 5))
    names = [c["name"] for c in TEST_CONDITIONS]
    rl_rates = [all_results[n][0]["success_rate"] for n in names]
    bl_rates = [all_results[n][1]["success_rate"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, rl_rates, width, label="RL agent", color="tab:blue")
    ax.bar(x + width / 2, bl_rates, width, label="potential-field baseline", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate (100 paired episodes)")
    ax.set_title("Phase 6: RL agent vs. potential-field baseline, by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase6_comparison_success_rates.png"), dpi=120)
    plt.close("all")

    print("=" * 78)
    print("PHASE 6 SUMMARY")
    print("=" * 78)
    rl_avg = float(np.mean(rl_rates))
    bl_avg = float(np.mean(bl_rates))
    print(f"  RL agent average success across all conditions      : {rl_avg:5.1%}")
    print(f"  baseline average success across all conditions       : {bl_avg:5.1%}")
    wins = sum(1 for n in names if all_results[n][1]["success_rate"] > all_results[n][0]["success_rate"])
    print(f"  conditions where baseline success rate is higher     : {wins}/{len(names)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
