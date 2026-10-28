"""
swarm_env.py — SwarmEnv: extends DroneSearchEnv with roles, multi-task
pipeline, simultaneous multi-agent stepping, and explicit TarMAC communication rewards.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Allow importing Multi-Agent-Systems env unchanged ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAS_PATH  = _REPO_ROOT / "Multi-Agent-Systems-Project-main"
if str(_MAS_PATH) not in sys.path:
    sys.path.insert(0, str(_MAS_PATH))

from env.grid_env import DroneSearchEnv, GLOBAL_CHANNELS, LOCAL_CHANNELS
from .tasks import TaskManager, TaskPhase, TASK_SEARCH, TASK_ANALYZE, TASK_TRANSPORT, TASK_FULL

# Role constants
ROLE_SCOUT    = 0
ROLE_CARRIER  = 1
ROLE_ANALYZER = 2
ROLE_NAMES    = {ROLE_SCOUT: "Scout", ROLE_CARRIER: "Carrier", ROLE_ANALYZER: "Analyzer"}

# Context normalization denominators
DEFAULT_CTX_NORM = {
    "vision_radius": 6.0,
    "comm_range":    12.0,
    "n_agents":      6.0,
}


class SwarmEnv(DroneSearchEnv):
    """
    Multi-role multi-task drone swarm environment with simultaneous multi-agent stepping.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_scouts:          int   = 2,
        n_carriers:        int   = 1,
        n_analyzers:       int   = 1,
        task:              str   = "search",
        analyze_time:      int   = 10,
        carry_time:        int   = 15,
        grid_size:         int   = 32,
        obstacle_density:  float = 0.15,
        vision_radius:     int   = 3,
        comm_range:        int   = 5,
        comm_metric:       str   = "manhattan",
        max_steps:         int   = 300,
        step_penalty:      float = -0.3,
        wall_penalty:      float = -0.02,
        revisit_penalty:   float = -0.05,
        target_reward:     float = 200.0,
        exploration_bonus: float = 0.05,
        collision_penalty: float = -0.5,
        fault_prob:        float = 0.0,
        seed:              int   = 42,
        config:            Optional[dict] = None,
        render_mode:       Optional[str]  = None,
        **kwargs,
    ):
        n_agents = n_scouts + n_carriers + n_analyzers

        # Construct configuration expected by DroneSearchEnv
        if config is None:
            base_env_cfg = {
                "grid_size":         grid_size,
                "obstacle_density":  obstacle_density,
                "n_agents":          n_agents,
                "vision_radius":     vision_radius,
                "comm_range":        comm_range,
                "comm_metric":       comm_metric,
                "max_steps":         max_steps,
                "step_penalty":      step_penalty,
                "wall_penalty":      wall_penalty,
                "revisit_penalty":   revisit_penalty,
                "target_reward":     target_reward,
                "exploration_bonus": exploration_bonus,
                "collision_penalty": collision_penalty,
                "fault_prob":        fault_prob,
                "seed":              seed,
            }
            # Allow any kwargs to override
            base_env_cfg.update(kwargs)
            config = {"env": base_env_cfg}
        else:
            # Update n_agents in provided config
            config["env"]["n_agents"] = n_agents

        super().__init__(config=config, render_mode=render_mode)

        self.n_scouts    = n_scouts
        self.n_carriers  = n_carriers
        self.n_analyzers = n_analyzers
        self.task_name   = task

        # Assign role IDs: scouts first, then carriers, then analyzers
        self.role_ids = (
            [ROLE_SCOUT]    * n_scouts   +
            [ROLE_CARRIER]  * n_carriers +
            [ROLE_ANALYZER] * n_analyzers
        )

        # Task manager tracks pipeline phase
        self.task_manager = TaskManager(
            task=task,
            n_agents=n_agents,
            analyze_time=analyze_time,
            carry_time=carry_time,
        )

        self._comm_penalty: float = 0.0

    # ------------------------------------------------------------------
    def set_comm_penalty(self, penalty: float) -> None:
        """Called by the trainer each step to inject TarMAC comm penalty."""
        self._comm_penalty = float(penalty)

    # ------------------------------------------------------------------
    def _build_context(self, agent_idx: int) -> np.ndarray:
        """
        Builds the 5-dim context vector for agent i:
        [vision_radius_norm, comm_range_norm, n_agents_norm, agent_id_norm, n_alive_belief_norm]
        """
        max_vr = DEFAULT_CTX_NORM["vision_radius"]
        max_cr = DEFAULT_CTX_NORM["comm_range"]
        max_na = DEFAULT_CTX_NORM["n_agents"]

        return np.array([
            self.vision_radius / max_vr,
            self.comm_range    / max_cr,
            self.n_agents      / max_na,
            agent_idx          / max_na,
            self.n_alive_belief(agent_idx) / max_na,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def _inject_obs_fields(self, obs_list: list) -> list:
        """Add 'ctx' and 'role_id' to each agent's observation dictionary."""
        enriched = []
        for i, obs in enumerate(obs_list):
            item = dict(obs)
            item["ctx"]     = self._build_context(i)
            item["role_id"] = self.role_ids[i]
            enriched.append(item)
        return enriched

    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[list, dict]:
        raw_obs, info = super().reset(**kwargs)
        self.task_manager.reset()
        self._comm_penalty = 0.0

        enriched_obs = self._inject_obs_fields(raw_obs)
        info["role_ids"]   = list(self.role_ids)
        info["task_phase"] = self.task_manager.current_phase
        return enriched_obs, info

    # ------------------------------------------------------------------
    def step(self, actions: List[int]) -> Tuple[list, List[float], bool, bool, dict]:
        """
        Execute simultaneous joint actions for all N agents.
        actions: list[int] of length n_agents
        """
        rewards = []
        any_term = False
        any_trunc = False
        target_found_this_step = False

        for i, a in enumerate(actions):
            if not self.done:
                obs_i, rew_i, term_i, trunc_i, info_i = self.step_agent(i, a)
                rewards.append(rew_i)
                if term_i:
                    any_term = True
                    target_found_this_step = True
                if trunc_i:
                    any_trunc = True
            else:
                rewards.append(0.0)

        # ── Role-specific reward modifiers ────────────────────────────
        rewards = self._apply_role_rewards(rewards)

        # ── Task phase rewards / transitions ──────────────────────────
        task_rewards, task_done = self.task_manager.step(
            agent_positions=self.agent_pos,
            agent_alive=self.agent_alive,
            role_ids=self.role_ids,
            target_pos=self.target_pos,
            target_found=target_found_this_step,
        )
        for i, tr in enumerate(task_rewards):
            rewards[i] += tr

        # ── TarMAC comm penalty ───────────────────────────────────────
        n_alive = int(self.agent_alive.sum())
        if n_alive > 0 and self._comm_penalty > 0.0:
            per_agent_penalty = self._comm_penalty / n_alive
            rewards = [r - per_agent_penalty for r in rewards]

        terminated = task_done or any_term or self.done
        truncated  = any_trunc or (self.step_count >= self.max_steps * self.n_agents)

        raw_all_obs  = self.get_all_obs()
        enriched_obs = self._inject_obs_fields(raw_all_obs)

        info = {
            "step":                   self.step_count,
            "target_found":           any_term or (self.agent_target.max() > 0),
            "target_found_this_step": target_found_this_step,
            "agent_positions":        list(self.agent_pos),
            "n_alive":                n_alive,
            "task_phase":             self.task_manager.current_phase,
            "role_ids":               list(self.role_ids),
            "comm_penalty":           self._comm_penalty,
        }

        return enriched_obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    def _apply_role_rewards(self, rewards: list) -> list:
        """
        Role-specific incentives:
        - Carriers: bonus when closing distance to discovered target.
        - Analyzers: bonus when hovering at target.
        """
        new_rewards = list(rewards)
        target_known = self.task_manager.target_found

        for i, (role, alive, pos) in enumerate(
            zip(self.role_ids, self.agent_alive, self.agent_pos)
        ):
            if not alive:
                continue

            if role == ROLE_CARRIER and target_known:
                dist = abs(pos[0] - self.target_pos[0]) + abs(pos[1] - self.target_pos[1])
                if dist <= 2:
                    new_rewards[i] += 0.5

            elif role == ROLE_ANALYZER and target_known:
                dist = abs(pos[0] - self.target_pos[0]) + abs(pos[1] - self.target_pos[1])
                if dist == 0:
                    new_rewards[i] += 0.3

        return new_rewards
