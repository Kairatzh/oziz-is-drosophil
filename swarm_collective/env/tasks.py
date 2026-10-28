"""
tasks.py — TaskManager: tracks multi-phase mission pipeline.

Phases (in order for 'full_pipeline' task):
  1. SEARCH      — at least one Scout/any drone must reach the target.
  2. ANALYZE     — an Analyzer must hover at the target for `analyze_time` steps.
  3. TRANSPORT   — a Carrier must reach the target and stay for `carry_time` steps.

Simpler tasks use a prefix of this pipeline:
  'search'           → only Phase 1
  'search_analyze'   → Phases 1 + 2
  'search_transport' → Phases 1 + 3 (skip analyze)
  'full_pipeline'    → Phases 1 + 2 + 3
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# Task mode constants
TASK_SEARCH     = "search"
TASK_ANALYZE    = "search_analyze"
TASK_TRANSPORT  = "search_transport"
TASK_FULL       = "full_pipeline"

# Phase names
class TaskPhase:
    SEARCH    = "search"
    ANALYZE   = "analyze"
    TRANSPORT = "transport"
    DONE      = "done"

ROLE_SCOUT    = 0
ROLE_CARRIER  = 1
ROLE_ANALYZER = 2


class TaskManager:
    """
    Tracks mission phase and emits per-agent reward bonuses.

    Args:
        task         : one of TASK_* constants
        n_agents     : total number of drones
        analyze_time : steps Analyzer must stay at target in ANALYZE phase
        carry_time   : steps Carrier must stay at target in TRANSPORT phase
    """

    def __init__(
        self,
        task:         str = TASK_FULL,
        n_agents:     int = 5,
        analyze_time: int = 10,
        carry_time:   int = 15,
    ):
        self.task         = task
        self.n_agents     = n_agents
        self.analyze_time = analyze_time
        self.carry_time   = carry_time

        # Build ordered phase list
        if task == TASK_SEARCH:
            self._phases = [TaskPhase.SEARCH]
        elif task == TASK_ANALYZE:
            self._phases = [TaskPhase.SEARCH, TaskPhase.ANALYZE]
        elif task == TASK_TRANSPORT:
            self._phases = [TaskPhase.SEARCH, TaskPhase.TRANSPORT]
        elif task == TASK_FULL:
            self._phases = [TaskPhase.SEARCH, TaskPhase.ANALYZE, TaskPhase.TRANSPORT]
        else:
            raise ValueError(f"Unknown task: {task!r}. Choose from "
                             f"{[TASK_SEARCH, TASK_ANALYZE, TASK_TRANSPORT, TASK_FULL]}")

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._phase_idx      = 0
        self.target_found    = False
        self._analyze_counter = 0
        self._carry_counter   = 0

    # ------------------------------------------------------------------
    @property
    def current_phase(self) -> str:
        if self._phase_idx >= len(self._phases):
            return TaskPhase.DONE
        return self._phases[self._phase_idx]

    # ------------------------------------------------------------------
    def step(
        self,
        agent_positions: list,      # list of [row, col] per agent
        agent_alive:     list,      # list of bool
        role_ids:        list,      # list of int (ROLE_*)
        target_pos:      tuple,     # (row, col) of the target
        target_found:    bool,      # True if any drone just found target this step
    ) -> Tuple[List[float], bool]:
        """
        Advance the task state machine.

        Returns:
            rewards   : list[float] — per-agent bonus rewards this step
            task_done : bool        — True if all phases completed
        """
        rewards   = [0.0] * self.n_agents
        task_done = False

        if self.current_phase == TaskPhase.DONE:
            return rewards, True

        # ── Phase 1: SEARCH ───────────────────────────────────────────
        if self.current_phase == TaskPhase.SEARCH:
            if target_found and not self.target_found:
                self.target_found = True
                # Big bonus to the discovering agent(s)
                tr, tc = target_pos
                for i, (pos, alive, role) in enumerate(
                    zip(agent_positions, agent_alive, role_ids)
                ):
                    if alive and list(pos) == [tr, tc]:
                        rewards[i] += 50.0   # discovery bonus
                self._advance_phase()

        # ── Phase 2: ANALYZE ──────────────────────────────────────────
        elif self.current_phase == TaskPhase.ANALYZE:
            tr, tc = target_pos
            analyzer_at_target = False
            for i, (pos, alive, role) in enumerate(
                zip(agent_positions, agent_alive, role_ids)
            ):
                if alive and role == ROLE_ANALYZER and list(pos) == [tr, tc]:
                    self._analyze_counter += 1
                    rewards[i] += 1.0     # progress reward
                    analyzer_at_target = True
                    break   # one analyzer is enough

            if not analyzer_at_target:
                self._analyze_counter = max(0, self._analyze_counter - 1)  # decay

            if self._analyze_counter >= self.analyze_time:
                # Analysis complete
                for i, (role, alive) in enumerate(zip(role_ids, agent_alive)):
                    if alive and role == ROLE_ANALYZER:
                        rewards[i] += 100.0   # completion bonus
                self._advance_phase()

        # ── Phase 3: TRANSPORT ────────────────────────────────────────
        elif self.current_phase == TaskPhase.TRANSPORT:
            tr, tc = target_pos
            carrier_at_target = False
            for i, (pos, alive, role) in enumerate(
                zip(agent_positions, agent_alive, role_ids)
            ):
                if alive and role == ROLE_CARRIER and list(pos) == [tr, tc]:
                    self._carry_counter += 1
                    rewards[i] += 1.0     # progress reward
                    carrier_at_target = True
                    break

            if not carrier_at_target:
                self._carry_counter = max(0, self._carry_counter - 1)

            if self._carry_counter >= self.carry_time:
                for i, (role, alive) in enumerate(zip(role_ids, agent_alive)):
                    if alive and role == ROLE_CARRIER:
                        rewards[i] += 150.0   # transport completion bonus
                self._advance_phase()

        # Check done
        if self.current_phase == TaskPhase.DONE:
            task_done = True

        return rewards, task_done

    # ------------------------------------------------------------------
    def _advance_phase(self) -> None:
        self._phase_idx += 1
        if self._phase_idx < len(self._phases):
            next_phase = self._phases[self._phase_idx]
            print(f"[TaskManager] → Phase: {next_phase}")

    # ------------------------------------------------------------------
    def progress_summary(self) -> dict:
        """Return a dict of phase progress for logging."""
        return {
            "phase":            self.current_phase,
            "target_found":     self.target_found,
            "analyze_progress": self._analyze_counter,
            "carry_progress":   self._carry_counter,
        }
