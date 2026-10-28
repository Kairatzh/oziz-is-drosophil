"""
Renders several episodes back-to-back into ONE animated GIF -- the Phase 8
"progression" demo showing the trained agent's arc from random
initialization to smooth, competent flight. Reuses the same drawing
conventions as viz/render_drone.py (single-episode renderer) but redraws
the static scene (obstacles, goal, caption) at the start of each segment,
since each stage can have a different environment, and holds on the last
frame of each segment briefly so the outcome is readable before cutting to
the next stage.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from env.quadrotor_env import CircleObstacle


def render_progression_gif(segments, out_path, fps=25, frame_stride=3, hold_frames=15):
    """
    segments: list of dicts, each with keys:
      label: str -- caption shown for this stage
      env: the QuadrotorEnv the episode ran in (for world bounds/obstacles/goal)
      states: (T, 6) array
      actions: (T, 2) array
      outcome: str
    """
    fig, ax = plt.subplots(figsize=(9, 7.2))

    frame_map = []  # list of (segment_index, local_frame_index)
    for si, seg in enumerate(segments):
        idxs = list(range(0, len(seg["states"]), frame_stride))
        if idxs[-1] != len(seg["states"]) - 1:
            idxs.append(len(seg["states"]) - 1)
        frame_map.extend((si, idx) for idx in idxs)
        frame_map.extend((si, idxs[-1]) for _ in range(hold_frames))

    def rotor_positions(env, x, z, theta):
        c, s = np.cos(theta), np.sin(theta)
        right = np.array([x + env.ARM_LENGTH * c, z + env.ARM_LENGTH * s])
        left = np.array([x - env.ARM_LENGTH * c, z - env.ARM_LENGTH * s])
        return left, right

    def update(frame_num):
        si, idx = frame_map[frame_num]
        seg = segments[si]
        env, states, actions = seg["env"], seg["states"], seg["actions"]

        ax.clear()
        ax.set_xlim(0, env.WORLD_WIDTH)
        ax.set_ylim(0, env.WORLD_HEIGHT)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(f"{seg['label']}\n(outcome: {seg['outcome']})  t = {idx * env.DT:.2f}s")

        for obs in env.obstacles:
            if isinstance(obs, CircleObstacle):
                ax.add_patch(Circle((obs.x, obs.z), obs.radius, color="firebrick", alpha=0.6))
            else:
                ax.add_patch(Rectangle(
                    (obs.x - obs.width / 2, obs.z - obs.height / 2),
                    obs.width, obs.height, color="firebrick", alpha=0.6,
                ))
        ax.add_patch(Circle(tuple(env.goal), env.GOAL_RADIUS, color="gold", alpha=0.5))

        ax.plot(states[:idx + 1, 0], states[:idx + 1, 1], color="steelblue", linewidth=1.0, alpha=0.6)

        x, z, vx, vz, theta, omega = states[idx]
        left_action, right_action = actions[idx]
        left_pos, right_pos = rotor_positions(env, x, z, theta)
        ax.plot([left_pos[0], right_pos[0]], [left_pos[1], right_pos[1]], color="black", linewidth=3.0)
        ax.scatter([left_pos[0]], [left_pos[1]], color="tab:blue", s=40, zorder=5)
        ax.scatter([right_pos[0]], [right_pos[1]], color="tab:orange", s=40, zorder=5)
        up = np.array([-np.sin(theta), np.cos(theta)])
        thrust_scale = 1.2
        ax.plot([left_pos[0], left_pos[0] + up[0] * left_action * thrust_scale],
                [left_pos[1], left_pos[1] + up[1] * left_action * thrust_scale], color="tab:blue", linewidth=2.0)
        ax.plot([right_pos[0], right_pos[0] + up[0] * right_action * thrust_scale],
                [right_pos[1], right_pos[1] + up[1] * right_action * thrust_scale], color="tab:orange", linewidth=2.0)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_map), blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close("all")
    return len(frame_map)
