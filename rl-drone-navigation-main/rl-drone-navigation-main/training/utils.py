"""
Shared training infrastructure used across every drone training script
(Phase 3's single-task run, Phase 4's curriculum). Factored out once it
was needed a second time, rather than speculatively.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from viz.render_drone import render_drone_gif


def run_evaluation(agent: PPOAgent, env: QuadrotorEnv, n_episodes: int):
    """Deterministic (no-exploration) evaluation episodes, kept entirely
    separate from the stochastic training rollout -- this is the number
    that should be trusted as "how good is the policy," since training
    rollouts include exploration noise that makes them look messier (or
    luckier) than the underlying policy actually is."""
    successes = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        outcome = "running"
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            outcome = info["outcome"]
        successes += int(outcome == "success")
        total_reward += episode_reward
    return successes / n_episodes, total_reward / n_episodes


def record_demo_episode(agent: PPOAgent, env: QuadrotorEnv, out_path: str, title: str, max_attempts: int = 10):
    """Runs the trained (deterministic) policy for one episode and
    renders it -- prefers a successful episode if one turns up within a
    few tries, but will render whatever the last attempt was otherwise,
    since a failure case is honest information too."""
    states, actions, outcome = None, None, None
    for _ in range(max_attempts):
        obs = env.reset()
        done = False
        ep_states = [env.state.copy()]
        ep_actions = [tuple(agent.act_deterministic(obs))]
        while not done:
            action = agent.act_deterministic(obs)
            obs, reward, done, info = env.step(action)
            ep_states.append(env.state.copy())
            ep_actions.append(tuple(action))
        states, actions, outcome = np.array(ep_states), np.array(ep_actions), info["outcome"]
        if outcome == "success":
            break

    n_frames = render_drone_gif(env, states, actions, out_path, title=f"{title} (outcome: {outcome})")
    return outcome, n_frames


def plot_learning_curve(
    episode_rewards, eval_history, success_threshold, moving_avg_window, out_path, title
):
    """eval_history: list of (update, success_rate, avg_reward) tuples."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    ax1.plot(episode_rewards, alpha=0.3, color="tab:blue", label="episode reward")
    if len(episode_rewards) >= moving_avg_window:
        moving_avg = np.convolve(
            episode_rewards, np.ones(moving_avg_window) / moving_avg_window, mode="valid"
        )
        ax1.plot(
            range(moving_avg_window - 1, len(episode_rewards)), moving_avg,
            color="tab:red", linewidth=2, label=f"{moving_avg_window}-episode moving average",
        )
    ax1.set_xlabel("episode")
    ax1.set_ylabel("total reward")
    ax1.set_title(title)
    ax1.legend()

    if eval_history:
        updates, success_rates, avg_rewards = zip(*eval_history)
        ax2.plot(updates, success_rates, marker="o", color="tab:green", label="eval success rate")
        ax2.axhline(success_threshold, color="gray", linestyle="--", label="progression threshold")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel("PPO update")
        ax2.set_ylabel("evaluation success rate")
        ax2.set_title("Deterministic evaluation success rate (no exploration noise)")
        ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close("all")
