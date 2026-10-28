"""
Phase 2 validation: train the from-scratch PPO implementation on the
hand-rolled CartPole environment and confirm the learning curve actually
climbs. This is the "does my algorithm even work" checkpoint before
Phase 3 connects PPO to the much harder drone environment -- if PPO
can't solve CartPole (a benchmark practically every RL algorithm solves
quickly), the bug is in PPO, not in some drone-specific complication.

Training loop, at a glance:
    repeat:
        collect a fixed number of environment steps using the CURRENT
        policy (a "rollout") -- this is on-policy: PPO can only learn
        from data collected by (something close to) its current policy,
        unlike off-policy methods (e.g. DQN) that can reuse old data.
        compute GAE advantages + value targets for that rollout
        run several epochs of minibatch gradient updates on that data
        discard the rollout (on-policy data can't be reused) and repeat

As required by CLAUDE.md: plt.show() is never called.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.cartpole_env import CartPoleEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ROLLOUT_STEPS = 2048     # environment steps collected per PPO update
MAX_UPDATES = 60         # hard cap regardless of whether it "solves" early
SOLVED_THRESHOLD = 475.0  # moving-average reward considered "solved" (max possible is 500)
MOVING_AVG_WINDOW = 20
SEED = 0


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = CartPoleEnv(seed=SEED)
    agent = PPOAgent(obs_dim=4, act_dim=2, config=PPOConfig())
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards = []
    current_episode_reward = 0.0
    obs = env.reset()

    solved_at_update = None

    for update in range(1, MAX_UPDATES + 1):
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=done, value=value)
            current_episode_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs = env.reset()

        last_value = agent.get_value(obs)
        diagnostics = agent.update(buffer, last_value)

        recent = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        moving_avg = float(np.mean(recent))
        print(
            f"update {update:3d}/{MAX_UPDATES}  episodes so far: {len(episode_rewards):4d}  "
            f"moving_avg_reward: {moving_avg:6.1f}  policy_loss: {diagnostics['policy_loss']:+.4f}  "
            f"value_loss: {diagnostics['value_loss']:.4f}  entropy: {diagnostics['entropy']:.3f}  "
            f"approx_kl: {diagnostics['approx_kl']:+.4f}"
        )

        if solved_at_update is None and moving_avg >= SOLVED_THRESHOLD and len(episode_rewards) >= MOVING_AVG_WINDOW:
            solved_at_update = update
            print(f"*** solved (moving avg >= {SOLVED_THRESHOLD}) at update {update} ***")
            break

    plot_learning_curve(episode_rewards, solved_at_update)
    return episode_rewards


def plot_learning_curve(episode_rewards, solved_at_update):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episode_rewards, alpha=0.3, color="tab:blue", label="episode reward")

    if len(episode_rewards) >= MOVING_AVG_WINDOW:
        moving_avg = np.convolve(
            episode_rewards, np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW, mode="valid"
        )
        ax.plot(
            range(MOVING_AVG_WINDOW - 1, len(episode_rewards)),
            moving_avg, color="tab:red", linewidth=2,
            label=f"{MOVING_AVG_WINDOW}-episode moving average",
        )

    ax.axhline(SOLVED_THRESHOLD, color="gray", linestyle="--", label="solved threshold")
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    title = "Phase 2: PPO on CartPole"
    if solved_at_update is not None:
        title += f" (solved at update {solved_at_update})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "phase2_cartpole_learning_curve.png")
    fig.savefig(out_path, dpi=120)
    plt.close("all")
    print(f"saved {out_path}")


if __name__ == "__main__":
    train()
