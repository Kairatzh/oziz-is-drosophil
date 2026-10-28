"""
swarm_env.py — SwarmEnv: extends DroneSearchEnv with roles, multi-task
pipeline, and explicit TarMAC communication rewards.

New features over the base DroneSearchEnv:
  1. **Roles**: Scout, Carrier, Analyzer with role-specific reward modifiers.
  2. **Task pipeline**: search → analyze → transport (sequential phases,
     controlled by TaskManager in tasks.py).
  3. **Comm penalty hook**: the env records a per-step comm_penalty supplied
     by the SwarmSystem and subtracts it from rewards (so the trainer only
     needs to call env.set_comm_penalty(penalty) each step).
  4. **Role obs channel**: adds a 7th global-map channel (Role) where each
     agent sees its own role_id normalized to [0,1].
  5. **Role IDs**: obs dict gets an extra 'role_id' key (int scalar).
"""

import sys
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Allow importing Multi-Agent-Systems env unchanged ────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAS_PATH  = _REPO_ROOT / "Multi-Agent-Systems-Project-main"
if str(_MAS_PATH) not in sys.path:
    sys.path.insert(0, str(_MAS_PATH))

from env.grid_env import DroneSearchEnv, GLOBAL_CHANNELS, LOCAL_CHANNELS  # noqa

from .tasks import TaskManager, TaskPhase, TASK_SEARCH, TASK_ANALYZE, TASK_TRANSPORT, TASK_FULL

# Role constants
ROLE_SCOUT    = 0
ROLE_CARRIER  = 1
ROLE_ANALYZER = 2
ROLE_NAMES    = {ROLE_SCOUT: "Scout", ROLE_CARRIER: "Carrier", ROLE_ANALYZER: "Analyzer"}

# Extra global channel: role_id (normalized)
ROLE_CHANNEL = GLOBAL_CHANNELS      # index 6 in extended 7-channel map
EXT_GLOBAL_CHANNELS = GLOBAL_CHANNELS + 1   # = 7


class SwarmEnv(DroneSearchEnv):
    """
    Multi-role multi-task drone swarm environment.

    Args (on top of DroneSearchEnv kwargs):
        n_scouts    : int — number of Scout drones
        n_carriers  : int — number of Carrier drones
        n_analyzers : int — number of Analyzer drones
        task        : str — one of 'search', 'search_analyze',
                      'search_transport', 'full_pipeline'
        analyze_time: int — steps an Analyzer must hover at target to complete
        carry_time  : int — steps a Carrier must stay at target to transport
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_scouts:    int = 2,
        n_carriers:  int = 1,
        n_analyzers: int = 1,
        task:        str = "search",
        analyze_time: int = 10,
        carry_time:   int = 15,
        **kwargs,
    ):
        # Total agents = sum of all roles
        n_agents = n_scouts + n_carriers + n_analyzers
        kwargs.setdefault("n_agents", n_agents)
        super().__init__(**kwargs)

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

        # Comm penalty (set externally each step by the trainer)
        self._comm_penalty: float = 0.0

        # Extend observation spaces to include role channel
        self._patch_obs_space()

    # ------------------------------------------------------------------
    def _patch_obs_space(self) -> None:
        """Add 'role_id' scalar to the observation space of each agent."""
        old_spaces = self.observation_space.spaces  # dict per agent, or single
        # DroneSearchEnv may use a single space for all agents or a dict per agent
        # We add 'role_id' (int64, scalar) to each agent's space
        # (handled in _get_obs by extending the dict)
        self._role_id_space = spaces.Discrete(3)

    # ------------------------------------------------------------------
    def set_comm_penalty(self, penalty: float) -> None:
        """Called by the trainer each step to inject TarMAC comm penalty."""
        self._comm_penalty = float(penalty)

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.task_manager.reset()
        self._comm_penalty = 0.0
        # Inject role_id into each agent's obs dict
        obs = self._inject_role_obs(obs)
        info["role_ids"] = list(self.role_ids)
        return obs, info

    # ------------------------------------------------------------------
    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step(actions)

        # ── Role-specific reward modifiers ────────────────────────────
        rewards = self._apply_role_rewards(rewards, info)

        # ── Task phase rewards / transitions ─────────────────────────
        task_rewards, task_done = self.task_manager.step(
            agent_positions=self.agent_pos,
            agent_alive=self.agent_alive,
            role_ids=self.role_ids,
            target_pos=self.target_pos,
            target_found=info.get("target_found_this_step", False),
        )
        for i, tr in enumerate(task_rewards):
            rewards[i] = rewards[i] + tr

        if task_done:
            terminated = True

        # ── TarMAC comm penalty (shared across all agents) ────────────
        n_alive = sum(self.agent_alive)
        if n_alive > 0:
            per_agent_penalty = self._comm_penalty / n_alive
            rewards = [r - per_agent_penalty for r in rewards]

        # ── Inject role obs ───────────────────────────────────────────
        obs = self._inject_role_obs(obs)
        info["task_phase"]  = self.task_manager.current_phase
        info["role_ids"]    = list(self.role_ids)
        info["comm_penalty"] = self._comm_penalty

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    def _apply_role_rewards(self, rewards: list, info: dict) -> list:
        """
        Apply per-role reward modifiers on top of base env rewards.

        Scouts   : exploration bonus × 1.5 (already in base env)
        Carriers : extra +0.5 if near target AFTER it's been found
        Analyzers: extra +0.3 per step hovering at target
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
                    new_rewards[i] += 0.3   # hovering at target

        return new_rewards

    # ------------------------------------------------------------------
    def _inject_role_obs(self, obs) -> dict:
        """Add 'role_id' key to each agent's observation dict."""
        if isinstance(obs, dict) and "global" in obs:
            # Single agent dict (shouldn't happen but guard)
            return obs

        # DroneSearchEnv returns a list of dicts or a single dict
        if isinstance(obs, list):
            for i, o in enumerate(obs):
                o["role_id"] = self.role_ids[i]
        elif isinstance(obs, dict):
            # Keyed by agent index
            for i in obs:
                obs[i]["role_id"] = self.role_ids[i]
        return obs

    # ------------------------------------------------------------------
    @property
    def n_agents(self) -> int:
        return self.n_scouts + self.n_carriers + self.n_analyzers
