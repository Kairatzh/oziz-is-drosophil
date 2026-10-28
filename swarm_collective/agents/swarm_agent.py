"""
swarm_agent.py — SwarmAgent: backbone + TarMAC + role embedding + Q-head.

One SwarmAgent instance is shared (parameter-sharing) across all agents of
the same TYPE (role).  Role identity is injected via a learnable embedding
looked up by role_id ∈ {0=Scout, 1=Carrier, 2=Analyzer}.

Full forward pipeline per agent i:
  1. obs  → BackboneExtractor  → combined (1413,)
  2. combined + role_emb       → FC(1413+64 → 512) + LN + GELU  → feat (512,)
  3. feat  → TarMACComm.encode → (msg_i, sig_i, qry_i)
     (aggregation happens at the SwarmSystem level after ALL agents encoded)
  4. feat + received context   → fused (512,)
  5. fused → Dueling Q-head    → Q(s,a) for N_ACTIONS=4 discrete actions
"""

import torch
import torch.nn as nn
from typing import Optional

from .backbone_loader import BackboneExtractor, FEAT_COMBINED
from .tarmac import TarMACComm

# Number of discrete movement actions (UP, DOWN, LEFT, RIGHT)
N_ACTIONS = 4

# Role IDs
SCOUT    = 0
CARRIER  = 1
ANALYZER = 2
N_ROLES  = 3


class SwarmAgent(nn.Module):
    """
    Parameter-shared multi-role agent with TarMAC communication.

    Args:
        backbone         : shared BackboneExtractor (may be None for unit tests)
        role_embedding_dim: dimension of the learnable role embedding (default 64)
        hidden_dim       : size of the shared feature representation (default 512)
        dim_msg          : TarMAC message vector size
        dim_key          : TarMAC signature/query size
        comm_cost        : L2 penalty coefficient per message norm
    """

    def __init__(
        self,
        backbone:          Optional[BackboneExtractor] = None,
        role_embedding_dim: int   = 64,
        hidden_dim:         int   = 512,
        dim_msg:            int   = 16,
        dim_key:            int   = 8,
        comm_cost:          float = 0.01,
    ):
        super().__init__()

        self.backbone   = backbone
        self.hidden_dim = hidden_dim

        # ── Role embedding ───────────────────────────────────────────────
        self.role_emb = nn.Embedding(N_ROLES, role_embedding_dim)

        # ── Input projection: [backbone_combined || role_emb] → hidden ───
        in_dim = FEAT_COMBINED + role_embedding_dim  # 1413 + 64 = 1477
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── TarMAC communication module ──────────────────────────────────
        self.tarmac = TarMACComm(
            feat_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dim_msg=dim_msg,
            dim_key=dim_key,
            comm_cost=comm_cost,
        )

        # ── Dueling Q-head ────────────────────────────────────────────────
        # Matches the original Multi-Agent DQN head style
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU())
        self.value_head     = nn.Linear(hidden_dim // 2, 1)
        self.advantage_head = nn.Linear(hidden_dim // 2, N_ACTIONS)

    # ------------------------------------------------------------------
    # Step 1+2: encode observation → hidden feature
    # ------------------------------------------------------------------
    def encode_obs(
        self,
        obs_global: torch.Tensor,   # (B, 6*32*32) or (B, 6, 32, 32)
        obs_local:  torch.Tensor,   # (B, 5*32*32) or (B, 5, 32, 32)
        ctx:        torch.Tensor,   # (B, 5)
        role_ids:   torch.Tensor,   # (B,) long — role per agent in batch
    ) -> torch.Tensor:
        """Returns hidden feature: (B, hidden_dim)."""
        if self.backbone is not None:
            _, _, combined = self.backbone(obs_global, obs_local, ctx)
        else:
            # Fallback (no backbone): just concat flattened obs
            g = obs_global.view(obs_global.size(0), -1).float()
            l = obs_local.view(obs_local.size(0), -1).float()
            combined = torch.cat([g, l, ctx], dim=-1)

        role_vec = self.role_emb(role_ids)              # (B, role_dim)
        x = torch.cat([combined, role_vec], dim=-1)    # (B, 1413+64)
        feat = self.input_proj(x)                      # (B, hidden_dim)
        return feat

    # ------------------------------------------------------------------
    # Step 3: encode message vectors (before aggregation)
    # ------------------------------------------------------------------
    def encode_messages(self, feat: torch.Tensor):
        """
        Args:
            feat : (N, hidden_dim) — features for N agents

        Returns:
            msg  : (N, dim_msg)
            sig  : (N, dim_key)
            qry  : (N, dim_key)
        """
        return self.tarmac.encode(feat)

    # ------------------------------------------------------------------
    # Step 4+5: fuse received context → Q-values
    # ------------------------------------------------------------------
    def compute_q(
        self,
        feat:    torch.Tensor,   # (B, hidden_dim)
        context: torch.Tensor,   # (B, dim_msg)
    ) -> torch.Tensor:
        """
        Returns Q-values: (B, N_ACTIONS).
        """
        fused = self.tarmac.fuse(torch.cat([feat, context], dim=-1))
        x = self.fc1(fused)
        x = self.fc2(x)
        v = self.value_head(x)          # (B, 1)
        a = self.advantage_head(x)      # (B, N_ACTIONS)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q

    # ------------------------------------------------------------------
    # Convenience: full single-agent forward (no comm — for debugging)
    # ------------------------------------------------------------------
    def forward_no_comm(
        self,
        obs_global: torch.Tensor,
        obs_local:  torch.Tensor,
        ctx:        torch.Tensor,
        role_ids:   torch.Tensor,
    ) -> torch.Tensor:
        """Returns Q-values (B, N_ACTIONS) without any communication."""
        feat    = self.encode_obs(obs_global, obs_local, ctx, role_ids)
        context = torch.zeros(feat.size(0), self.tarmac.dim_msg, device=feat.device)
        return self.compute_q(feat, context)


# ── SwarmSystem: manages N agents with shared network + TarMAC pass ───────────

class SwarmSystem(nn.Module):
    """
    Top-level wrapper managing a swarm of agents sharing one SwarmAgent network.

    Usage (one environment step):
        1. feats = system.encode_all(observations)
        2. q_vals, comm_penalty, attn = system.communicate_and_qvals(
               feats, positions, alive_mask, comm_range)
        3. actions = epsilon_greedy(q_vals, epsilon)
    """

    def __init__(self, agent: SwarmAgent):
        super().__init__()
        self.agent = agent

    # ------------------------------------------------------------------
    def encode_all(
        self,
        obs_globals: torch.Tensor,  # (N, 6*32*32)
        obs_locals:  torch.Tensor,  # (N, 5*32*32)
        ctxs:        torch.Tensor,  # (N, 5)
        role_ids:    torch.Tensor,  # (N,) long
    ) -> torch.Tensor:
        """Encode all N agents' observations → feats (N, hidden_dim)."""
        return self.agent.encode_obs(obs_globals, obs_locals, ctxs, role_ids)

    # ------------------------------------------------------------------
    def communicate_and_qvals(
        self,
        feats:      torch.Tensor,   # (N, hidden_dim)
        positions:  torch.Tensor,   # (N, 2) grid positions
        alive_mask: torch.Tensor,   # (N,) bool
        comm_range: int,
    ):
        """
        Run TarMAC message passing and compute Q-values for all agents.

        Returns:
            q_vals       : (N, N_ACTIONS)
            comm_penalty : scalar
            attn         : (N, N) attention weights
        """
        adj = TarMACComm.build_adjacency(positions, comm_range, alive_mask)
        fused, comm_penalty, attn, msg_norms = self.agent.tarmac(
            feats, alive_mask, adj
        )
        q_vals = self.agent.compute_q(feats, fused - feats)  # context = fused - feat

        # Zero out Q-values for broken agents (they take no-op)
        q_vals = q_vals * alive_mask.float().unsqueeze(-1)
        return q_vals, comm_penalty, attn
