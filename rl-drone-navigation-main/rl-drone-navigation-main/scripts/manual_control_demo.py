"""
Phase 1 manual control demo.

Flies the drone through a hand-scripted (open-loop) sequence of rotor
commands -- no RL, no feedback controller navigating toward a goal -- and
renders the result as an animated GIF. The point is to show the physics
engine responding believably to control inputs in an environment that
contains obstacles, before any learning is introduced in later phases.

Each "maneuver" below is a pair of opposite thrust-imbalance pulses of
equal magnitude and duration. Because torque is proportional to the
thrust difference (see env/quadrotor_env.py), an imbalance pulse followed
by its exact negation always returns the angular velocity (omega) to
zero -- but leaves a net *residual* tilt, which is what lets the drone
hold a bank angle and cruise sideways. Repeating the negated pair again
later removes that residual tilt and returns the drone to level flight.
This doublet/relevel pattern is hand-tuned here (by simulating ahead of
time), not computed by any onboard controller -- it is the scripted
equivalent of stick movements a human pilot would make.

A small symmetric thrust "boost" (equal on both rotors, so it adds no
torque) is applied during sustained banks to counteract the real
aerodynamic fact that a tilted rotor thrust vector has a smaller vertical
component (lift ~ cos(theta)) -- without it the drone gently sinks while
banked, exactly as a real quadrotor would.

As required by CLAUDE.md: plt.show() is never called; the figure is
saved and closed.
"""

import os

import numpy as np

from env.quadrotor_env import CircleObstacle, QuadrotorEnv, RectObstacle
from viz.render_drone import render_drone_gif

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# (steps, differential d, symmetric boost b) -> action = [H + d + b, H - d + b]
FLIGHT_PLAN = [
    (25, 0.0, 0.0),                          # settle into hover
    (12, 0.035, 0.0), (12, -0.035, 0.0),     # bank cycle 1 -> residual rightward tilt
    (45, 0.0, 0.022),                        # cruise 1 (lift-compensated)
    (12, -0.035, 0.0), (12, 0.035, 0.0),     # relevel cycle 1
    (15, 0.0, 0.0),                          # brief level coast
    (12, 0.035, 0.0), (12, -0.035, 0.0),     # bank cycle 2
    (45, 0.0, 0.022),                        # cruise 2
    (12, -0.035, 0.0), (12, 0.035, 0.0),     # relevel cycle 2
    (15, 0.0, 0.0),                          # coast
    (16, -0.05, 0.0), (16, 0.05, 0.0),       # brake cycle -> opposite residual tilt
    (25, 0.0, 0.022),                        # hold brake tilt to bleed off speed
    (16, 0.05, 0.0), (16, -0.05, 0.0),       # relevel after braking (mirrors brake cycle)
    (30, 0.0, 0.0),                          # coast off remaining speed under drag
    (20, 0.0, 0.0),                          # final hover settle
]

START_STATE = [1.0, 4.5, 0.0, 0.0, 0.0, 0.0]
GOAL = [8.5, 5.5]   # not targeted by the scripted pilot -- just a scene marker

OBSTACLES = [
    CircleObstacle(x=3.5, z=2.5, radius=1.0),
    CircleObstacle(x=6.0, z=7.0, radius=0.9),
    RectObstacle(x=8.2, z=2.6, width=1.6, height=1.6),
]


def simulate():
    env = QuadrotorEnv(obstacles=OBSTACLES)
    env.reset_to(state=START_STATE, goal=GOAL)
    hover = env.HOVER_ACTION

    states = [env.state.copy()]
    actions = [(hover, hover)]
    for steps, d, boost in FLIGHT_PLAN:
        left, right = hover + d + boost, hover - d + boost
        for _ in range(steps):
            env.step([left, right])
            states.append(env.state.copy())
            actions.append((left, right))

    return env, np.array(states), np.array(actions)


if __name__ == "__main__":
    env, states, actions = simulate()
    print(f"Simulated {len(states)} steps ({len(states) * env.DT:.2f}s of flight)")
    print(f"x range: [{states[:,0].min():.2f}, {states[:,0].max():.2f}]  (world width {env.WORLD_WIDTH})")
    print(f"z range: [{states[:,1].min():.2f}, {states[:,1].max():.2f}]  (world height {env.WORLD_HEIGHT})")
    print(f"max |theta|: {np.degrees(np.abs(states[:,4])).max():.2f} deg")

    out_path = os.path.join(RESULTS_DIR, "phase1_manual_control_demo.gif")
    n_frames = render_drone_gif(
        env, states, actions, out_path,
        title="Phase 1: manual scripted control inputs in an obstacle environment",
    )
    print(f"[manual control demo] saved {out_path} ({n_frames} frames)")
