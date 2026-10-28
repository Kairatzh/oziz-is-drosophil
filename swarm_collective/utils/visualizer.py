"""
visualizer.py — Visualization tools for multi-agent swarm & communication graphs.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(metrics_history: dict, save_path: str = "results_training.png") -> None:
    """
    Plots training loss, reward, success rate, and communication volume over episodes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Success Rate
    ax = axes[0, 0]
    if "success_rate" in metrics_history:
        ax.plot(metrics_history["success_rate"], color="#2ca02c", lw=2)
        ax.set_title("Success Rate (Rolling)", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # 2. Episode Rewards
    ax = axes[0, 1]
    if "rewards" in metrics_history:
        ax.plot(metrics_history["rewards"], color="#1f77b4", alpha=0.6)
        ax.set_title("Total Team Reward", fontsize=12)
        ax.grid(True, alpha=0.3)

    # 3. Communication Norm (Chattiness)
    ax = axes[1, 0]
    if "msg_norm" in metrics_history:
        ax.plot(metrics_history["msg_norm"], color="#ff7f0e", lw=2)
        ax.set_title("Message Norm ||m|| (Comm Bandwidth)", fontsize=12)
        ax.grid(True, alpha=0.3)

    # 4. Training Loss
    ax = axes[1, 1]
    if "loss" in metrics_history:
        ax.plot(metrics_history["loss"], color="#d62728", alpha=0.8)
        ax.set_title("QMIX TD Loss", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close("all")
    print(f"[Visualizer] Saved training curves to: {save_path}")


def plot_communication_graph(
    positions:  List[List[int]],
    role_names: List[str],
    alive_mask: List[bool],
    attn_matrix: np.ndarray,
    target_pos: Optional[List[int]] = None,
    save_path:  str = "comm_graph.png"
) -> None:
    """
    Plots the spatial positions of drones, their roles, and TarMAC communication links.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    N = len(positions)

    # Colors per role
    role_colors = {
        "Scout": "#1f77b4",     # Blue
        "Carrier": "#2ca02c",   # Green
        "Analyzer": "#9467bd"   # Purple
    }

    # Plot Target if provided
    if target_pos is not None:
        ax.scatter([target_pos[1]], [target_pos[0]], marker="*", s=300, color="#d62728", zorder=5, label="Target")

    # Plot Drones
    for i in range(N):
        r, c = positions[i]
        role = role_names[i]
        alive = alive_mask[i]
        color = role_colors.get(role, "#7f7f7f") if alive else "#333333"
        alpha = 1.0 if alive else 0.3
        marker = "o" if alive else "X"

        ax.scatter([c], [r], s=160, color=color, alpha=alpha, marker=marker, zorder=4)
        ax.text(c + 0.3, r + 0.3, f"{role}_{i}", fontsize=9, fontweight="bold")

    # Draw attention communication directed arrows
    if attn_matrix is not None:
        for i in range(N):
            for j in range(N):
                if i != j and attn_matrix[i, j] > 0.05 and alive_mask[i] and alive_mask[j]:
                    ri, ci = positions[i]
                    rj, cj = positions[j]
                    weight = float(attn_matrix[i, j])
                    ax.annotate(
                        "",
                        xy=(cj, rj),
                        xytext=(ci, ri),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="#e377c2",
                            lw=1.0 + 3.0 * weight,
                            alpha=min(1.0, 0.2 + weight),
                            connectionstyle="arc3,rad=0.1"
                        ),
                        zorder=2
                    )

    ax.set_xlim(-1, 33)
    ax.set_ylim(33, -1)  # Grid image coords (0,0 top-left)
    ax.set_title("Swarm Directed Communication Graph (TarMAC Attention)", fontsize=11)
    ax.set_xlabel("Grid X (Col)")
    ax.set_ylabel("Grid Y (Row)")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close("all")
    print(f"[Visualizer] Saved communication graph to: {save_path}")
