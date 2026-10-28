"""
replay_buffer.py — Multi-agent episode & transition replay buffer for QMIX.

Stores multi-agent transitions with support for:
- Variable team sizes (padded to max_agents)
- Agent positions & alive masks (for adjacency graph & dead agents masking)
- Role IDs
- Global joint state (for QMIX hyper-networks)
- n-step discounted returns
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch


@dataclass
class MultiAgentTransition:
    """Stores one joint step across all N agents."""
    obs_global:      np.ndarray  # (N, 6144)
    obs_local:       np.ndarray  # (N, 5120)
    ctx:             np.ndarray  # (N, 5)
    role_ids:        np.ndarray  # (N,) int64
    positions:       np.ndarray  # (N, 2) int64
    alive_mask:      np.ndarray  # (N,) bool
    actions:         np.ndarray  # (N,) int64
    rewards:         np.ndarray  # (N,) float32
    next_obs_global: np.ndarray  # (N, 6144)
    next_obs_local:  np.ndarray  # (N, 5120)
    next_ctx:        np.ndarray  # (N, 5)
    next_positions:  np.ndarray  # (N, 2) int64
    next_alive_mask: np.ndarray  # (N,) bool
    terminated:      bool
    truncated:       bool


class MultiAgentReplayBuffer:
    """
    Fixed-capacity cyclic replay buffer for multi-agent transitions.
    """

    def __init__(self, capacity: int = 100000, max_agents: int = 6):
        self.capacity   = capacity
        self.max_agents = max_agents
        self.buffer     = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: MultiAgentTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of multi-agent transitions and converts to padded PyTorch tensors.
        Returns:
            Dict containing batched tensors:
              - 'obs_global'     : (B, max_agents, 6144)
              - 'obs_local'      : (B, max_agents, 5120)
              - 'ctx'            : (B, max_agents, 5)
              - 'role_ids'       : (B, max_agents) long
              - 'positions'      : (B, max_agents, 2) long
              - 'alive_mask'     : (B, max_agents) bool
              - 'actions'        : (B, max_agents) long
              - 'rewards'        : (B, max_agents) float32
              - 'next_obs_global': (B, max_agents, 6144)
              - 'next_obs_local' : (B, max_agents, 5120)
              - 'next_ctx'       : (B, max_agents, 5)
              - 'next_positions' : (B, max_agents, 2) long
              - 'next_alive_mask': (B, max_agents) bool
              - 'dones'          : (B, 1) float32
              - 'global_state'   : (B, max_agents * 6144)
              - 'next_state'     : (B, max_agents * 6144)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        B = len(batch)
        M = self.max_agents

        obs_g      = np.zeros((B, M, 6144), dtype=np.float32)
        obs_l      = np.zeros((B, M, 5120), dtype=np.float32)
        ctx        = np.zeros((B, M, 5), dtype=np.float32)
        role_ids   = np.zeros((B, M), dtype=np.int64)
        pos        = np.zeros((B, M, 2), dtype=np.int64)
        alive      = np.zeros((B, M), dtype=bool)
        actions    = np.zeros((B, M), dtype=np.int64)
        rewards    = np.zeros((B, M), dtype=np.float32)

        next_obs_g = np.zeros((B, M, 6144), dtype=np.float32)
        next_obs_l = np.zeros((B, M, 5120), dtype=np.float32)
        next_ctx   = np.zeros((B, M, 5), dtype=np.float32)
        next_pos   = np.zeros((B, M, 2), dtype=np.int64)
        next_alive = np.zeros((B, M), dtype=bool)
        dones      = np.zeros((B, 1), dtype=np.float32)

        for b, item in enumerate(batch):
            n = min(len(item.actions), M)
            obs_g[b, :n]      = item.obs_global[:n]
            obs_l[b, :n]      = item.obs_local[:n]
            ctx[b, :n]        = item.ctx[:n]
            role_ids[b, :n]   = item.role_ids[:n]
            pos[b, :n]        = item.positions[:n]
            alive[b, :n]      = item.alive_mask[:n]
            actions[b, :n]    = item.actions[:n]
            rewards[b, :n]    = item.rewards[:n]

            next_obs_g[b, :n] = item.next_obs_global[:n]
            next_obs_l[b, :n] = item.next_obs_local[:n]
            next_ctx[b, :n]   = item.next_ctx[:n]
            next_pos[b, :n]   = item.next_positions[:n]
            next_alive[b, :n] = item.next_alive_mask[:n]
            dones[b, 0]       = float(item.terminated or item.truncated)

        # Joint global state vectors (concatenation of all agents' global obs)
        state_tensor      = torch.from_numpy(obs_g).reshape(B, -1).to(device)
        next_state_tensor = torch.from_numpy(next_obs_g).reshape(B, -1).to(device)

        return {
            "obs_global":      torch.from_numpy(obs_g).to(device),
            "obs_local":       torch.from_numpy(obs_l).to(device),
            "ctx":             torch.from_numpy(ctx).to(device),
            "role_ids":        torch.from_numpy(role_ids).to(device),
            "positions":       torch.from_numpy(pos).to(device),
            "alive_mask":      torch.from_numpy(alive).to(device),
            "actions":         torch.from_numpy(actions).to(device),
            "rewards":         torch.from_numpy(rewards).to(device),
            "next_obs_global": torch.from_numpy(next_obs_g).to(device),
            "next_obs_local":  torch.from_numpy(next_obs_l).to(device),
            "next_ctx":        torch.from_numpy(next_ctx).to(device),
            "next_positions":  torch.from_numpy(next_pos).to(device),
            "next_alive_mask": torch.from_numpy(next_alive).to(device),
            "dones":           torch.from_numpy(dones).to(device),
            "global_state":    state_tensor,
            "next_state":      next_state_tensor,
        }
