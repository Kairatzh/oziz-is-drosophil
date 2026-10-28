"""
Phase 3: connect the from-scratch PPO implementation to the drone
environment for the simplest possible task -- no obstacles, just hover
and reach a random target from a random start.

This is a much harder problem than CartPole in one specific way: CartPole
has a discrete, tiny action space (2 choices) and a policy that only has
to prevent one angle from drifting too far. The drone has a continuous,
2D action space, is underactuated (Phase 1's central theme), and starts
every episode from a different random position with a different random
goal. Success here means the agent has discovered, purely from reward
signal, that it needs to tilt to translate and level out to stop -- the
exact maneuver that was hand-scripted in Phase 1's manual demo.

Training infrastructure, as required by PROJECT_SPEC.md Phase 3:
  - episode logging: reward per episode, outcome (success/collision/
    out_of_bounds/timeout), and moving averages of both
  - periodic evaluation: run separate episodes with the DETERMINISTIC
    policy (no exploration sampling) so we measure what the agent has
    actually learned, not how lucky its exploration noise got
  - checkpointing: save weights periodically so a run can be inspected
    or resumed without retraining from scratch

As required by CLAUDE.md: plt.show() is never called; CPU training only.
"""

import os

import numpy as np
import torch

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer
from training.utils import plot_learning_curve, record_demo_episode, run_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ROLLOUT_STEPS = 4096
MAX_UPDATES = 300
EVAL_EVERY = 5
EVAL_EPISODES = 20
CHECKPOINT_EVERY = 10
SUCCESS_THRESHOLD = 0.9
MOVING_AVG_WINDOW = 20
SEED = 0


def train(max_updates: int = MAX_UPDATES):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = QuadrotorEnv(obstacles=[], seed=SEED)
    eval_env = QuadrotorEnv(obstacles=[], seed=SEED + 1000)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(),
    )
    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    episode_rewards = []
    episode_outcomes = []
    eval_history = []  # (update, success_rate, avg_reward)

    current_episode_reward = 0.0
    obs = env.reset()
    solved_at_update = None

    for update in range(1, max_updates + 1):
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=done, value=value)
            current_episode_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(current_episode_reward)
                episode_outcomes.append(info["outcome"])
                current_episode_reward = 0.0
                obs = env.reset()

        last_value = agent.get_value(obs)
        diagnostics = agent.update(buffer, last_value)

        recent_rewards = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        recent_outcomes = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_success_rate = (
            np.mean([o == "success" for o in recent_outcomes]) if recent_outcomes else 0.0
        )

        print(
            f"update {update:3d}/{max_updates}  episodes: {len(episode_rewards):5d}  "
            f"moving_avg_reward: {np.mean(recent_rewards):8.2f}  "
            f"train_success_rate: {train_success_rate:5.1%}  "
            f"policy_loss: {diagnostics['policy_loss']:+.4f}  value_loss: {diagnostics['value_loss']:8.3f}  "
            f"entropy: {diagnostics['entropy']:+.3f}"
        )

        if update % EVAL_EVERY == 0:
            eval_success_rate, eval_avg_reward = run_evaluation(agent, eval_env, EVAL_EPISODES)
            eval_history.append((update, eval_success_rate, eval_avg_reward))
            print(
                f"    [eval @ update {update}] success_rate: {eval_success_rate:5.1%}  "
                f"avg_reward: {eval_avg_reward:8.2f}"
            )
            if eval_success_rate >= SUCCESS_THRESHOLD:
                solved_at_update = update
                print(f"*** solved (eval success rate >= {SUCCESS_THRESHOLD:.0%}) at update {update} ***")
                break

        if update % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"phase3_hover_reach_update{update:04d}.pt")
            agent.save(ckpt_path)

    final_path = os.path.join(CHECKPOINT_DIR, "phase3_hover_reach_final.pt")
    agent.save(final_path)
    print(f"saved final checkpoint to {final_path}")

    title = "Phase 3: PPO on drone hover-and-reach (no obstacles)"
    if solved_at_update is not None:
        title += f" -- solved at update {solved_at_update}"
    plot_learning_curve(
        episode_rewards, eval_history, SUCCESS_THRESHOLD, MOVING_AVG_WINDOW,
        out_path=os.path.join(RESULTS_DIR, "phase3_learning_curve.png"), title=title,
    )

    demo_path = os.path.join(RESULTS_DIR, "phase3_trained_agent_demo.gif")
    outcome, n_frames = record_demo_episode(
        agent, eval_env, demo_path, title="Phase 3: trained agent, hover-and-reach"
    )
    print(f"saved {demo_path} ({n_frames} frames, outcome={outcome})")

    return agent, episode_rewards, episode_outcomes, eval_history


if __name__ == "__main__":
    train()
