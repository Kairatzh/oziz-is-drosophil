"""
Actor and Critic networks for PPO.

These are deliberately kept as two separate, small feedforward networks
rather than a single network with two heads sharing a trunk. Sharing a
trunk is a common efficiency optimization, but it also couples the two
learning signals (policy gradient and value regression) through shared
weights, which is a subtlety worth avoiding while first learning how PPO
works. Two independent networks keeps the actor/critic distinction -- the
core conceptual split in actor-critic methods -- visible in the code
itself.

Actor:  state -> a *distribution* over actions (the policy, pi_theta).
        We never output a single action directly; we output the
        parameters of a distribution and sample from it. This is what
        makes exploration possible: early in training the distribution
        is close to uniform/wide, so sampled actions are essentially
        random, and it sharpens as the policy improves.

Critic: state -> a single scalar, the *value* V_phi(s): the expected
        total future reward from this state onward if the current policy
        is followed. The critic doesn't choose actions; it only judges
        how good a state is, which is what makes it possible to compute
        advantages (see ppo/rollout_buffer.py) with much lower variance
        than raw episode returns.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal


def _orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    # Orthogonal initialization (with a small gain on the final layer of
    # the actor) is a well-known PPO stability trick: it keeps the
    # initial policy close to uniform-random and the initial value
    # estimates close to zero, avoiding large, noisy gradients in the
    # very first updates before the networks have learned anything.
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Actor(nn.Module):
    """Discrete-action policy network: state -> categorical distribution over actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, act_dim), gain=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (unnormalized log-probabilities) over actions."""
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)


class GaussianActor(nn.Module):
    """Continuous-action policy network: state -> a diagonal Gaussian
    distribution over actions.

    Used for the drone, whose actions (two real-valued thrust commands)
    form a continuum rather than a small discrete set -- exactly why a
    DQN-style "pick the action with the highest Q-value" approach has no
    natural equivalent here; there's no way to enumerate every possible
    thrust value and take an argmax.

    The standard deviation is a single learned parameter per action
    dimension, shared across every state ("state-independent" std) --
    this is a common simplification in continuous-control PPO (it halves
    what the network has to learn: it only has to predict the mean, and
    a single global exploration scale adapts over training).

    Note this network's raw output is unbounded (a real-valued mean), and
    sampling from a Gaussian can land outside the environment's valid
    action range. Rather than adding a bias-correcting squashing function
    (e.g. tanh, as in SAC), this project relies on the environment
    itself clipping actions to a valid range -- a common, simpler
    convention for continuous PPO baselines.

    log_std is clamped to [MIN_LOG_STD, MAX_LOG_STD] (std in ~[0.05, 0.5]).
    Without this, the entropy bonus in the PPO loss (see ppo_agent.py)
    gives the optimizer a persistent, unopposed incentive to *increase*
    entropy by growing log_std, since it directly reduces the loss with
    no natural counterforce once the policy is roughly correct. On the
    obstacle-free Phase 3/4-Level-1 task this went unnoticed because the
    dense progress-shaping reward was forgiving enough to solve the task
    anyway -- but on Level 2 (obstacles), entropy climbed ~6x over 325
    updates (0.45 -> 2.6) with no sign of leveling off, and eval success
    rate plateaued at 25-75% instead of climbing toward the 90% target.
    Clamping log_std caps how noisy the policy can become, without
    removing its ability to still explore within a reasonable range.
    """

    MIN_LOG_STD = -3.0   # std ~= 0.05
    MAX_LOG_STD = -0.7   # std ~= 0.50

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_size: int = 64,
        init_mean_bias: Optional[Sequence[float]] = None,
        init_log_std: float = -1.2,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
        )
        self.mean_head = _orthogonal_init(nn.Linear(hidden_size, act_dim), gain=0.01)
        if init_mean_bias is not None:
            # Bias the initial policy toward a known-reasonable action
            # (e.g. hover thrust) instead of 0 -- a randomly-initialized
            # mean near 0 means near-zero thrust, so literally every
            # rollout free-falls and crashes immediately, which gives the
            # critic almost nothing informative to learn from.
            with torch.no_grad():
                self.mean_head.bias.copy_(torch.as_tensor(init_mean_bias, dtype=torch.float32))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor):
        mean = self.mean_head(self.trunk(obs))
        std = torch.exp(torch.clamp(self.log_std, self.MIN_LOG_STD, self.MAX_LOG_STD))
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Independent:
        mean, std = self.forward(obs)
        return Independent(Normal(mean, std), 1)


class Critic(nn.Module):
    """Value network: state -> scalar estimate of expected future return."""

    def __init__(self, obs_dim: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            _orthogonal_init(nn.Linear(obs_dim, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, hidden_size), gain=np.sqrt(2)),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
