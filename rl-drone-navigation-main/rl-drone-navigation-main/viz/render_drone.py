"""
Shared drone episode rendering, used by every phase that needs to turn a
recorded (states, actions) trajectory into an animated GIF: the Phase 1
manual control demo, the Phase 3+ trained-agent rollouts, and beyond.

Factored out once it was needed a second time (Phase 3), rather than
speculatively -- Phase 1's manual_control_demo.py originally had this
logic inline.

As required by CLAUDE.md: plt.show() is never called; figures are saved
and closed.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from env.quadrotor_env import CircleObstacle


def render_drone_gif(env, states, actions, out_path, title, frame_stride=3, fps=25):
    """
    env: a QuadrotorEnv instance -- used for world bounds, obstacle list,
         goal position, and physical constants (ARM_LENGTH etc).
    states: array of shape (T, 6), the [x, z, vx, vz, theta, omega] history.
    actions: array of shape (T, 2), the [left, right] thrust command history.
    """
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set_xlim(0, env.WORLD_WIDTH)
    ax.set_ylim(0, env.WORLD_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)

    for obs in env.obstacles:
        if isinstance(obs, CircleObstacle):
            ax.add_patch(Circle((obs.x, obs.z), obs.radius, color="firebrick", alpha=0.6))
        else:
            ax.add_patch(Rectangle(
                (obs.x - obs.width / 2, obs.z - obs.height / 2),
                obs.width, obs.height, color="firebrick", alpha=0.6,
            ))

    ax.add_patch(Circle(tuple(env.goal), env.GOAL_RADIUS, color="gold", alpha=0.5, label="goal"))

    trail_line, = ax.plot([], [], color="steelblue", linewidth=1.0, alpha=0.6)
    body_line, = ax.plot([], [], color="black", linewidth=3.0)
    rotor_l = ax.scatter([], [], color="tab:blue", s=40, zorder=5)
    rotor_r = ax.scatter([], [], color="tab:orange", s=40, zorder=5)
    thrust_l, = ax.plot([], [], color="tab:blue", linewidth=2.0)
    thrust_r, = ax.plot([], [], color="tab:orange", linewidth=2.0)
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")

    frame_indices = list(range(0, len(states), frame_stride))
    thrust_arrow_scale = 1.2

    def rotor_positions(x, z, theta):
        c, s = np.cos(theta), np.sin(theta)
        right = np.array([x + env.ARM_LENGTH * c, z + env.ARM_LENGTH * s])
        left = np.array([x - env.ARM_LENGTH * c, z - env.ARM_LENGTH * s])
        return left, right

    def update(frame_num):
        i = frame_indices[frame_num]
        x, z, vx, vz, theta, omega = states[i]
        left_action, right_action = actions[i]
        left_pos, right_pos = rotor_positions(x, z, theta)

        body_line.set_data([left_pos[0], right_pos[0]], [left_pos[1], right_pos[1]])
        rotor_l.set_offsets([left_pos])
        rotor_r.set_offsets([right_pos])

        up = np.array([-np.sin(theta), np.cos(theta)])
        thrust_l.set_data(
            [left_pos[0], left_pos[0] + up[0] * left_action * thrust_arrow_scale],
            [left_pos[1], left_pos[1] + up[1] * left_action * thrust_arrow_scale],
        )
        thrust_r.set_data(
            [right_pos[0], right_pos[0] + up[0] * right_action * thrust_arrow_scale],
            [right_pos[1], right_pos[1] + up[1] * right_action * thrust_arrow_scale],
        )

        trail_line.set_data(states[:i + 1, 0], states[:i + 1, 1])
        time_text.set_text(f"t = {i * env.DT:.2f}s")
        return body_line, rotor_l, rotor_r, thrust_l, thrust_r, trail_line, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close("all")
    return len(frame_indices)
