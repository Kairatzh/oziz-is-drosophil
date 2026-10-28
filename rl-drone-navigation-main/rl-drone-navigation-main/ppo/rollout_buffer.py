"""
Rollout storage and Generalized Advantage Estimation (GAE).

"Advantage" answers a specific question: was this action better or worse
than what the critic expected from this state on average? A_t =
Q(s_t, a_t) - V(s_t). Positive advantage means "do more of this action
in this state"; negative means "do less." Training the policy on
advantages rather than raw returns is what lets PPO tell apart "this
state was already great" from "this action made it better."

The problem with estimating advantage directly is a variance/bias
tradeoff:
  - One-step TD residual delta_t = r_t + gamma*V(s_{t+1}) - V(s_t) is a
    low-variance but biased estimate of the advantage (it trusts the
    critic's V estimate, which is itself imperfect, especially early in
    training).
  - Full Monte Carlo return minus baseline is unbiased but high-variance
    (it depends on every reward for the rest of the episode, including a
    lot of action noise from exploration).

GAE(lambda) blends these: it's an exponentially-weighted average of
n-step TD residuals, controlled by lambda in [0, 1]. lambda=0 recovers
the low-variance/high-bias one-step estimate; lambda=1 recovers the
high-variance/low-bias Monte Carlo estimate. lambda=0.95 (used here) is
the standard middle ground from the original GAE paper (Schulman et al.,
2015).
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add(self, obs, action, log_prob, reward, done, value) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_advantages_and_returns(self, last_value: float):
        """Computes GAE advantages and value-function regression targets
        (returns = advantages + values) for every step currently stored.

        `last_value` is the critic's estimate of the state *after* the
        final stored step -- needed to bootstrap the recursion when the
        rollout is cut off mid-episode rather than at a true terminal
        state.
        """
        T = len(self)
        values = np.array(self.values + [last_value], dtype=np.float64)
        advantages = np.zeros(T, dtype=np.float64)

        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages, returns

    def get_tensors(self, last_value: float):
        advantages, returns = self.compute_advantages_and_returns(last_value)
        obs = torch.as_tensor(np.array(self.observations), dtype=torch.float32)
        actions = torch.as_tensor(np.array(self.actions))
        old_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        return obs, actions, old_log_probs, advantages_t, returns_t

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
