"""
Phase 1 physics validation.

Runs two manual sanity checks and saves plots to results/:

1. Hover test: constant, symmetric thrust should hold the drone at a
   fixed altitude (net force = 0 when thrust exactly balances gravity).
2. Differential thrust test: an imbalance between the two rotors should
   produce rotation, and rotation should in turn produce lateral motion.

Nothing here is trained -- these are open-loop scripted inputs used only
to confirm the simulator's Newtonian mechanics are implemented correctly
before we build anything (a PPO agent, a demo) on top of it.

As required by CLAUDE.md: every figure is saved with plt.savefig() and
the figure is closed afterward. plt.show() is never called.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/Windows runs
import matplotlib.pyplot as plt
import numpy as np

from env.quadrotor_env import QuadrotorEnv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_hover_test():
    env = QuadrotorEnv()
    start_z = 4.0
    env.reset_to(state=[5.0, start_z, 0.0, 0.0, 0.0, 0.0], goal=[0.2, 0.2])

    hover_action = [env.HOVER_ACTION, env.HOVER_ACTION]
    n_steps = 250  # 5 seconds
    history = np.zeros((n_steps, 6))
    for i in range(n_steps):
        env.step(hover_action)
        history[i] = env.state

    t = np.arange(n_steps) * env.DT

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, history[:, 1], label="altitude z (m)")
    axes[0].axhline(start_z, color="gray", linestyle="--", label="target altitude")
    axes[0].set_ylabel("z (m)")
    axes[0].set_title(f"Hover test: HOVER_ACTION={env.HOVER_ACTION:.4f} per rotor")
    axes[0].legend()

    axes[1].plot(t, history[:, 0], label="x position (m)", color="tab:orange")
    axes[1].set_ylabel("x (m)")
    axes[1].legend()

    axes[2].plot(t, np.degrees(history[:, 4]), label="pitch theta (deg)", color="tab:green")
    axes[2].set_ylabel("theta (deg)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    fig.suptitle("Phase 1 sanity check 1: symmetric thrust should hold position")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "phase1_hover_test.png")
    fig.savefig(out_path, dpi=120)
    plt.close("all")

    final_z_error = abs(history[-1, 1] - start_z)
    final_x_error = abs(history[-1, 0] - 5.0)
    print(f"[hover test] saved {out_path}")
    print(f"[hover test] final altitude error: {final_z_error:.4f} m (want ~0)")
    print(f"[hover test] final horizontal drift: {final_x_error:.4f} m (want ~0)")
    print(f"[hover test] final theta: {np.degrees(history[-1, 4]):.4f} deg (want ~0)")


def run_differential_thrust_test():
    env = QuadrotorEnv()
    env.reset_to(state=[5.0, 4.0, 0.0, 0.0, 0.0, 0.0], goal=[0.2, 0.2])

    # Apply a fixed torque-producing imbalance (left > right) briefly,
    # then hold hover to see the drone coast on the velocity it built up.
    # NOTE: with this drone's torque/inertia, 1s of full imbalance would
    # spin it through several full rotations -- 12 steps (0.24s) keeps
    # this a realistic short "bank" maneuver, not a tumble.
    n_burst = 12    # 0.24 seconds of banking
    n_coast = 100   # 2 seconds of coasting at hover thrust
    n_steps = n_burst + n_coast
    history = np.zeros((n_steps, 6))

    burst_action = [env.HOVER_ACTION + 0.15, env.HOVER_ACTION - 0.15]  # left > right
    hover_action = [env.HOVER_ACTION, env.HOVER_ACTION]

    for i in range(n_steps):
        action = burst_action if i < n_burst else hover_action
        env.step(action)
        history[i] = env.state

    t = np.arange(n_steps) * env.DT

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, np.degrees(history[:, 4]), color="tab:green")
    axes[0].axvline(n_burst * env.DT, color="gray", linestyle="--", label="thrust burst ends")
    axes[0].set_ylabel("theta (deg)")
    axes[0].set_title("Differential thrust test: left > right rotor thrust")
    axes[0].legend()

    axes[1].plot(t, history[:, 5], color="tab:red")
    axes[1].axvline(n_burst * env.DT, color="gray", linestyle="--")
    axes[1].set_ylabel("omega (rad/s)")

    axes[2].plot(t, history[:, 0], color="tab:orange", label="x position")
    axes[2].axvline(n_burst * env.DT, color="gray", linestyle="--")
    axes[2].set_ylabel("x (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    fig.suptitle("Phase 1 sanity check 2: rotor imbalance -> rotation -> lateral motion")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "phase1_differential_thrust_test.png")
    fig.savefig(out_path, dpi=120)
    plt.close("all")

    print(f"[differential thrust test] saved {out_path}")
    print(f"[differential thrust test] theta at end of burst: {np.degrees(history[n_burst - 1, 4]):.2f} deg (want < 0)")
    print(f"[differential thrust test] final x displacement: {history[-1, 0] - 5.0:.4f} m (want > 0, i.e. moved right)")


if __name__ == "__main__":
    run_hover_test()
    run_differential_thrust_test()
