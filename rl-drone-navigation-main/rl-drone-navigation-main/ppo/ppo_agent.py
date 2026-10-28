"""
The PPO agent: ties the actor, critic, and rollout buffer together into
the actual "Proximal Policy Optimization" update rule.

Why not simpler alternatives?
- REINFORCE (vanilla policy gradient) updates directly on raw returns
  with no critic baseline, so gradient estimates are extremely
  high-variance, and there is nothing stopping a single bad update from
  moving the policy so far that it never recovers -- because the *next*
  batch of data is collected with that same ruined policy.
- DQN (and other value-based methods) learn a Q-function and act
  greedily with respect to it. That's a natural fit for small discrete
  action spaces, but the drone's action space is continuous
  (two real-valued thrust commands) -- DQN has no natural way to take an
  argmax over a continuum of actions.
PPO keeps the actor-critic structure (a critic to reduce variance, like
REINFORCE-with-baseline) but adds a clipping mechanism that keeps every
policy update small and safe, which is what makes it stable enough to be
one of the most widely used general-purpose RL algorithms in practice.

The clipping trick, precisely:
  ratio_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
This ratio is 1.0 if the new policy assigns the same probability to the
action actually taken as the old (data-collecting) policy did; >1 if the
new policy makes that action more likely, <1 if less likely. The naive
surrogate objective ratio_t * A_t would, if you fully optimized it, let
the ratio grow without bound whenever A_t > 0 -- i.e., "if this action
was good, make it infinitely more likely right now," which overshoots
wildly. PPO instead optimizes:
  L^CLIP = min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t)
Once ratio_t moves outside [1-eps, 1+eps], the clipped term stops
providing gradient in the direction that would push it further outside
that range. This caps how much the policy is allowed to change from the
data-collecting policy in a single update -- hence "proximal."
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ppo.networks import Actor, Critic, GaussianActor
from ppo.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    n_epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: PPOConfig = PPOConfig(),
        continuous: bool = False,
        init_mean_bias: Optional[Sequence[float]] = None,
    ):
        self.config = config
        self.continuous = continuous
        if continuous:
            self.actor = GaussianActor(
                obs_dim, act_dim, hidden_size=config.hidden_size, init_mean_bias=init_mean_bias
            )
        else:
            self.actor = Actor(obs_dim, act_dim, hidden_size=config.hidden_size)
        self.critic = Critic(obs_dim, hidden_size=config.hidden_size)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=config.lr
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """Samples an action from the current policy for rollout collection.
        Returns (action, log_prob, value) as plain Python/numpy types."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = self.actor.get_distribution(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_t)
        if self.continuous:
            return action.squeeze(0).numpy(), log_prob.item(), value.item()
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray):
        """Returns the policy's best-guess action with no sampling noise
        -- the distribution's mean (continuous) or most likely action
        (discrete). Used for evaluation, where we want to measure what
        the agent has actually learned, not exploration behavior."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.continuous:
            mean, _ = self.actor.forward(obs_t)
            return mean.squeeze(0).numpy()
        logits = self.actor.forward(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.critic(obs_t).item()

    def save(self, path: str) -> None:
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "continuous": self.continuous,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict:
        cfg = self.config
        obs, actions, old_log_probs, advantages, returns = buffer.get_tensors(last_value)

        # Normalizing advantages per-batch is a standard PPO stabilization
        # trick: it keeps the scale of the policy gradient consistent
        # across batches regardless of the reward function's raw scale.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = obs.shape[0]
        diagnostics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}

        for _ in range(cfg.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, cfg.minibatch_size):
                mb_idx = indices[start:start + cfg.minibatch_size]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                dist = self.actor.get_distribution(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(mb_obs)
                value_loss = nn.functional.mse_loss(values_pred, mb_returns)

                # Entropy bonus: subtracting entropy from the loss (i.e.
                # rewarding high entropy) discourages the policy from
                # collapsing to a deterministic action too early, which
                # would stop exploration before the agent has found a
                # good strategy.
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm
                )
                self.optimizer.step()

                if self.continuous:
                    # Belt-and-suspenders: GaussianActor.forward() already
                    # clamps log_std when computing std, but Adam's momentum
                    # can still nudge the raw parameter beyond the clamp
                    # bounds (zero instantaneous gradient there doesn't mean
                    # zero momentum). Clamping the parameter itself keeps it
                    # from drifting arbitrarily far past the bound over many
                    # thousands of updates.
                    with torch.no_grad():
                        self.actor.log_std.clamp_(self.actor.MIN_LOG_STD, self.actor.MAX_LOG_STD)

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item()

                diagnostics["policy_loss"].append(policy_loss.item())
                diagnostics["value_loss"].append(value_loss.item())
                diagnostics["entropy"].append(entropy.item())
                diagnostics["approx_kl"].append(approx_kl)
                diagnostics["clip_fraction"].append(clip_fraction)

        return {k: float(np.mean(v)) for k, v in diagnostics.items()}
