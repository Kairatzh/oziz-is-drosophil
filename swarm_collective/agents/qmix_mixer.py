"""
qmix_mixer.py — QMIX monotonic value mixing network.

Reference: Rashid et al., "QMIX: Monotonic Value Function Factorisation for
Deep Multi-Agent Reinforcement Learning" (ICML 2018).

Architecture:
  Two hyper-networks conditioned on the global state s generate the weights
  and biases of a two-layer mixing MLP.  Absolute values of weights enforce
  monotonicity: Q_tot is non-decreasing in each Q_i.

  Layer 1:  hidden = ELU( |W1(s)| · Q + b1(s) )
  Layer 2:  Q_tot  = |W2(s)| · hidden + b2(s)

  b2(s) passes through an identity (no abs), allowing the total value to
  go negative.

Global state:
  Concatenation of all agents' global map observations (padded to max_agents).
  Shape: (B, max_agents * GLOBAL_OBS_DIM)  where GLOBAL_OBS_DIM = 6*32*32 = 6144.

  Using raw pixel maps is expensive.  We optionally accept pre-computed
  backbone features instead (max_agents × 1024).
"""

import torch
import torch.nn as nn


class QMixMixer(nn.Module):
    """
    QMIX mixing network.

    Args:
        n_agents          : maximum number of agents (padding target)
        state_dim         : dimensionality of the global state vector
        mixing_embed_dim  : hidden width of the mixing MLP  (default 32)
        hypernet_embed_dim: hidden width of the hyper-networks (default 64)
    """

    def __init__(
        self,
        n_agents:           int,
        state_dim:          int,
        mixing_embed_dim:   int = 32,
        hypernet_embed_dim: int = 64,
    ):
        super().__init__()
        self.n_agents  = n_agents
        self.state_dim = state_dim
        self.embed_dim = mixing_embed_dim

        # ── Hyper-network for W1: state → (n_agents × embed_dim) ─────────
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed_dim, n_agents * mixing_embed_dim),
        )

        # ── Hyper-network for b1: state → (embed_dim,) ───────────────────
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
        )

        # ── Hyper-network for W2: state → (embed_dim × 1) ────────────────
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim),
        )

        # ── Hyper-network for b2: state → scalar ─────────────────────────
        #   Two-layer MLP for more expressiveness in the output bias.
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mixing_embed_dim, 1),
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        q_vals: torch.Tensor,   # (B, n_agents)  — chosen Q-values per agent
        state:  torch.Tensor,   # (B, state_dim) — global state
    ) -> torch.Tensor:
        """
        Mix individual Q-values into a monotonic joint Q_tot.

        Returns:
            q_tot : (B, 1)
        """
        B = q_vals.size(0)
        qs = q_vals.unsqueeze(-1)  # (B, n_agents, 1)

        # ── Layer 1 ───────────────────────────────────────────────────
        w1 = self.hyper_w1(state)                           # (B, N*E)
        w1 = w1.abs().view(B, self.n_agents, self.embed_dim)  # (B, N, E)  monotone
        b1 = self.hyper_b1(state).unsqueeze(1)             # (B, 1, E)

        hidden = torch.bmm(qs.transpose(1, 2), w1)        # (B, 1, E)
        hidden = torch.nn.functional.elu(hidden + b1)     # (B, 1, E)

        # ── Layer 2 ───────────────────────────────────────────────────
        w2 = self.hyper_w2(state).abs()                    # (B, E)  monotone
        w2 = w2.unsqueeze(-1)                              # (B, E, 1)
        b2 = self.hyper_b2(state)                          # (B, 1)

        q_tot = torch.bmm(hidden, w2).squeeze(-1) + b2    # (B, 1)
        return q_tot


# ── State builder helper ──────────────────────────────────────────────────────

def build_global_state(
    obs_globals: list,          # list of (6144,) tensors, one per agent
    max_agents:  int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Concatenate per-agent global observations into a single state vector.
    Pads with zeros for missing/broken agents so the dimension is fixed.

    Args:
        obs_globals : list[Tensor(6144,)]  length = actual n_agents (alive+broken)
        max_agents  : int — padding target
        device      : torch device

    Returns:
        state : (state_dim,)  where state_dim = max_agents * 6144
    """
    obs_dim = obs_globals[0].numel()
    parts = []
    for obs in obs_globals[:max_agents]:
        parts.append(obs.to(device).float().flatten())
    # Pad missing agents with zeros
    for _ in range(max_agents - len(obs_globals)):
        parts.append(torch.zeros(obs_dim, device=device))
    return torch.cat(parts, dim=0)   # (max_agents * obs_dim,)


def build_global_state_batch(
    obs_globals_batch: torch.Tensor,   # (B, n_agents, 6144)
    max_agents:        int,
    device:            torch.device,
) -> torch.Tensor:
    """
    Batched version: (B, n_agents, obs_dim) → (B, max_agents * obs_dim).
    """
    B, N, obs_dim = obs_globals_batch.shape
    if N < max_agents:
        pad = torch.zeros(B, max_agents - N, obs_dim, device=device)
        obs_globals_batch = torch.cat([obs_globals_batch, pad], dim=1)
    return obs_globals_batch[:, :max_agents].reshape(B, -1)
