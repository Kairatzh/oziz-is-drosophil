"""
tarmac.py — TarMAC (Targeted Multi-Agent Communication) module.

Reference: Das et al., "TarMAC: Targeted Multi-Agent Communication" (ICML 2019).

Each agent i generates:
  - message  m_i  (dim_msg)   — what to broadcast
  - signature s_i  (dim_key)  — who should listen (key for attention)
  - query    q_i  (dim_key)  — what agent i is looking for

Aggregation for agent i (only over neighbours within comm_range):
  α_ij = softmax_j( dot(s_j, q_i) / sqrt(dim_key) )   [attention weight]
  c_i  = Σ_j  α_ij · m_j                               [received context]

Communication penalty (added to reward):
  r_comm_i = -comm_cost · ||m_i||_2

The fused representation for agent i's decision head:
  fused_i = FuseNet( cat[ feat_i,  c_i ] )   → (hidden_dim,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TarMACComm(nn.Module):
    """
    TarMAC communication module.

    Args:
        feat_dim    : dimension of the per-agent feature vector coming in
        hidden_dim  : output dimension after fusion (same as feat_dim by default)
        dim_msg     : size of message vector m_i
        dim_key     : size of signature/query vectors s_i, q_i
        comm_cost   : penalty coefficient λ for ||m_i||_2
    """

    def __init__(
        self,
        feat_dim:   int = 512,
        hidden_dim: int = 512,
        dim_msg:    int = 16,
        dim_key:    int = 8,
        comm_cost:  float = 0.01,
    ):
        super().__init__()
        self.dim_msg   = dim_msg
        self.dim_key   = dim_key
        self.comm_cost = comm_cost
        self.scale     = math.sqrt(dim_key)

        # Encoder: feat → message vector (what to say)
        self.msg_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_msg),
            nn.Tanh(),
        )

        # Encoder: feat → signature (how to be addressed)
        self.key_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_key),
        )

        # Encoder: feat → query (what to listen for)
        self.query_encoder = nn.Sequential(
            nn.Linear(feat_dim, dim_key),
        )

        # Fusion: [feat_i || received_context_i] → hidden
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim + dim_msg, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    # ------------------------------------------------------------------
    def encode(self, feat: torch.Tensor):
        """
        Encode per-agent features into messages / keys / queries.

        Args:
            feat : (N, feat_dim)  — features for N agents in one env

        Returns:
            msg  : (N, dim_msg)
            sig  : (N, dim_key)  — signature / key
            qry  : (N, dim_key)  — query
        """
        return (
            self.msg_encoder(feat),
            self.key_encoder(feat),
            self.query_encoder(feat),
        )

    # ------------------------------------------------------------------
    def aggregate(
        self,
        msg:       torch.Tensor,       # (N, dim_msg)
        sig:       torch.Tensor,       # (N, dim_key)
        qry:       torch.Tensor,       # (N, dim_key)
        alive_mask: torch.Tensor,      # (N,) bool — True if agent is active
        adj:        torch.Tensor,      # (N, N) bool — True if i can hear j
    ):
        """
        Compute attention-weighted message aggregation.

        Returns:
            context : (N, dim_msg)  — aggregated received messages per agent
            attn    : (N, N)        — attention weights (for logging)
        """
        N = msg.size(0)

        # Raw attention scores: (N_query, N_key) = qry @ sig^T / scale
        scores = torch.matmul(qry, sig.t()) / self.scale   # (N, N)

        # Mask: agent can only receive from alive neighbours within comm_range
        # comm_mask[i, j] = True  ⟹  agent i can hear agent j
        comm_mask = adj & alive_mask.unsqueeze(0)           # (N, N)
        scores = scores.masked_fill(~comm_mask, float("-inf"))

        # Softmax over senders (dim=1); if no valid sender → zero vector
        # Guard against all-inf rows (isolated agents)
        valid_row = comm_mask.any(dim=1)                    # (N,)
        attn = torch.zeros(N, N, device=msg.device)
        if valid_row.any():
            attn[valid_row] = F.softmax(scores[valid_row], dim=1)

        # Weighted sum of messages
        context = torch.matmul(attn, msg)                   # (N, dim_msg)
        return context, attn

    # ------------------------------------------------------------------
    def forward(
        self,
        feat:       torch.Tensor,      # (N, feat_dim)
        alive_mask: torch.Tensor,      # (N,) bool
        adj:        torch.Tensor,      # (N, N) bool — adjacency within comm_range
    ):
        """
        Full TarMAC forward pass.

        Returns:
            fused       : (N, hidden_dim)  — feature after comm fusion
            comm_penalty: scalar tensor    — to subtract from rewards
            attn        : (N, N)           — attention weights (for logging)
            msg_norms   : (N,)             — ||m_i||_2 per agent (for logging)
        """
        msg, sig, qry = self.encode(feat)                   # (N, *) each
        context, attn = self.aggregate(msg, sig, qry, alive_mask, adj)

        fused = self.fuse(torch.cat([feat, context], dim=-1))  # (N, hidden)

        # Penalty: proportional to message L2-norm for ACTIVE agents only
        msg_norms    = msg.norm(dim=-1)                     # (N,)
        active_norms = msg_norms * alive_mask.float()
        comm_penalty = self.comm_cost * active_norms.sum()

        return fused, comm_penalty, attn, msg_norms

    # ------------------------------------------------------------------
    @staticmethod
    def build_adjacency(
        positions:  torch.Tensor,  # (N, 2) integer grid positions (row, col)
        comm_range: int,
        alive_mask: torch.Tensor,  # (N,) bool
    ) -> torch.Tensor:
        """
        Build boolean adjacency matrix: adj[i,j]=True iff Manhattan dist ≤ comm_range
        and both agents are alive.

        Args:
            positions  : (N, 2) tensor of agent grid positions
            comm_range : int — maximum Manhattan distance for communication
            alive_mask : (N,) bool

        Returns:
            adj : (N, N) bool tensor
        """
        N = positions.size(0)
        # Pairwise Manhattan distance
        diff = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (N, N, 2)
        dist = diff.sum(-1)                                               # (N, N)
        in_range = dist <= comm_range
        both_alive = alive_mask.unsqueeze(0) & alive_mask.unsqueeze(1)
        adj = in_range & both_alive
        # No self-messages
        adj.fill_diagonal_(False)
        return adj
