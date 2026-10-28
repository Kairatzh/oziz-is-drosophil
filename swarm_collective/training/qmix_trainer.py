"""
qmix_trainer.py — Centralized QMIX Trainer with TarMAC and frozen backbone support.

Features:
- Parameter-sharing SwarmAgent across all roles (distinguished by role_ids)
- Monotonic QMixMixer hypernetworks
- Double-Q target estimation for stability
- Gradient clipping & learning rate separation (frozen backbone vs heads)
- Epsilon-greedy multi-agent exploration
"""

import copy
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.backbone_loader import get_optimizer_param_groups
from agents.swarm_agent import SwarmAgent, SwarmSystem, N_ACTIONS
from agents.qmix_mixer import QMixMixer
from .replay_buffer import MultiAgentReplayBuffer


class QMixTrainer:
    """
    Coordinates training of the SwarmSystem and QMixMixer.
    """

    def __init__(
        self,
        swarm_system:     SwarmSystem,
        max_agents:       int = 6,
        obs_dim_per_agent: int = 6144,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
        lr:               float = 3e-4,
        gamma:            float = 0.97,
        target_update_freq: int = 500,
        grad_clip:        float = 1.0,
        device:           torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.max_agents = max_agents
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        # Online networks
        self.swarm_system = swarm_system.to(device)
        state_dim = max_agents * obs_dim_per_agent
        self.mixer = QMixMixer(
            n_agents=max_agents,
            state_dim=state_dim,
            mixing_embed_dim=mixing_embed_dim,
            hypernet_embed_dim=hypernet_embed_dim,
        ).to(device)

        # Target networks (deep copies)
        self.target_swarm_system = copy.deepcopy(self.swarm_system).to(device)
        self.target_mixer        = copy.deepcopy(self.mixer).to(device)

        for p in self.target_swarm_system.parameters():
            p.requires_grad = False
        for p in self.target_mixer.parameters():
            p.requires_grad = False

        # Optimizer: parameter groups respecting backbone freeze
        agent_params = get_optimizer_param_groups(self.swarm_system, base_lr=lr)
        mixer_params = [{"params": list(self.mixer.parameters()), "lr": lr}]
        self.optimizer = torch.optim.Adam(agent_params + mixer_params)

        self.train_steps = 0

    # ------------------------------------------------------------------
    def select_actions(
        self,
        obs_list:   List[Dict[str, Any]],
        role_ids:   List[int],
        positions:  List[List[int]],
        alive_mask: List[bool],
        comm_range: int,
        epsilon:    float = 0.0,
    ) -> Tuple[List[int], float, Optional[np.ndarray]]:
        """
        Computes actions for all agents in the environment for the current timestep.

        Returns:
            actions      : list of int actions (length N)
            comm_penalty : scalar float to apply to env reward
            attn_weights : (N, N) numpy array for telemetry/logging
        """
        N = len(obs_list)
        if N == 0 or not any(alive_mask):
            return [0] * N, 0.0, None

        self.swarm_system.eval()
        with torch.no_grad():
            obs_g = torch.stack([
                torch.from_numpy(o["global"]).float() for o in obs_list
            ]).to(self.device)
            obs_l = torch.stack([
                torch.from_numpy(o["local"]).float() for o in obs_list
            ]).to(self.device)
            ctx = torch.stack([
                torch.from_numpy(o["ctx"]).float() for o in obs_list
            ]).to(self.device)
            roles = torch.tensor(role_ids, dtype=torch.long, device=self.device)
            pos   = torch.tensor(positions, dtype=torch.long, device=self.device)
            alive = torch.tensor(alive_mask, dtype=torch.bool, device=self.device)

            feats = self.swarm_system.encode_all(obs_g, obs_l, ctx, roles)
            q_vals, comm_penalty, attn = self.swarm_system.communicate_and_qvals(
                feats, pos, alive, comm_range
            )

        actions = []
        for i in range(N):
            if not alive_mask[i]:
                actions.append(0)  # No-op for broken/inactive drone
            elif np.random.rand() < epsilon:
                actions.append(int(np.random.randint(N_ACTIONS)))
            else:
                actions.append(int(q_vals[i].argmax().item()))

        return actions, float(comm_penalty.item()), attn.cpu().numpy()

    # ------------------------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Performs one QMIX training step over a sampled batch.
        """
        self.swarm_system.train()
        self.mixer.train()

        B, M = batch["actions"].shape

        # ── 1. Calculate online Q-values for chosen actions ─────────────
        # Flatten batch and agents: (B * M, ...)
        flat_obs_g = batch["obs_global"].view(B * M, -1)
        flat_obs_l = batch["obs_local"].view(B * M, -1)
        flat_ctx   = batch["ctx"].view(B * M, -1)
        flat_roles = batch["role_ids"].view(B * M)

        feats = self.swarm_system.agent.encode_obs(
            flat_obs_g, flat_obs_l, flat_ctx, flat_roles
        ).view(B, M, -1)

        # Communicate per episode in batch
        all_q_chosen = []
        for b in range(B):
            pos_b   = batch["positions"][b]
            alive_b = batch["alive_mask"][b]
            q_vals, _, _ = self.swarm_system.communicate_and_qvals(
                feats[b], pos_b, alive_b, comm_range=5
            )
            actions_b = batch["actions"][b].unsqueeze(-1)  # (M, 1)
            chosen_q = q_vals.gather(1, actions_b).squeeze(-1)  # (M,)
            all_q_chosen.append(chosen_q)

        q_chosen = torch.stack(all_q_chosen, dim=0)  # (B, M)

        # Mix Q-values into Q_tot
        q_tot = self.mixer(q_chosen, batch["global_state"]).squeeze(-1)  # (B,)

        # ── 2. Calculate target Q-values (Double Q-learning) ────────────
        with torch.no_grad():
            flat_next_g = batch["next_obs_global"].view(B * M, -1)
            flat_next_l = batch["next_obs_local"].view(B * M, -1)
            flat_next_ctx = batch["next_ctx"].view(B * M, -1)

            next_feats_online = self.swarm_system.agent.encode_obs(
                flat_next_g, flat_next_l, flat_next_ctx, flat_roles
            ).view(B, M, -1)
            next_feats_target = self.target_swarm_system.agent.encode_obs(
                flat_next_g, flat_next_l, flat_next_ctx, flat_roles
            ).view(B, M, -1)

            all_target_q_chosen = []
            for b in range(B):
                next_pos_b   = batch["next_positions"][b]
                next_alive_b = batch["next_alive_mask"][b]

                # Online network selects next action
                q_next_online, _, _ = self.swarm_system.communicate_and_qvals(
                    next_feats_online[b], next_pos_b, next_alive_b, comm_range=5
                )
                best_actions = q_next_online.argmax(dim=-1, keepdim=True)  # (M, 1)

                # Target network evaluates selected action
                q_next_target, _, _ = self.target_swarm_system.communicate_and_qvals(
                    next_feats_target[b], next_pos_b, next_alive_b, comm_range=5
                )
                target_q = q_next_target.gather(1, best_actions).squeeze(-1)  # (M,)
                all_target_q_chosen.append(target_q)

            target_q_chosen = torch.stack(all_target_q_chosen, dim=0)  # (B, M)
            target_q_tot = self.target_mixer(
                target_q_chosen, batch["next_state"]
            ).squeeze(-1)  # (B,)

            # Team reward = sum of alive agents' rewards
            team_reward = (batch["rewards"] * batch["alive_mask"].float()).sum(dim=-1)
            dones = batch["dones"].squeeze(-1)
            targets = team_reward + self.gamma * (1.0 - dones) * target_q_tot

        # ── 3. Compute loss and optimize ────────────────────────────────
        loss = F.smooth_l1_loss(q_tot, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.swarm_system.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.mixer.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_swarm_system.load_state_dict(self.swarm_system.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())

        return {
            "loss": float(loss.item()),
            "q_tot_mean": float(q_tot.mean().item()),
            "target_mean": float(targets.mean().item())
        }
