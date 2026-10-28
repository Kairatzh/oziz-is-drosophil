"""
curriculum.py — Automatic curriculum staging for the collective swarm.

Controls environment progression:
  Stage 1: Scout only (search)
  Stage 2: Scout + Carrier (search & transport)
  Stage 3: Scout + Analyzer (search & analyze)
  Stage 4: Full swarm (search -> analyze -> transport)
  Stage 5: Full swarm with faults & communication packet drops
"""

from collections import deque
from typing import Dict, Any, Optional
import numpy as np


class CurriculumManager:
    """
    Monitors rolling evaluation success rate and advances environment stages.
    """

    def __init__(self, stages_config: list, window_size: int = 100):
        self.stages = stages_config
        self.current_stage_idx = 0
        self.window_size = window_size
        self.success_history = deque(maxlen=window_size)

    @property
    def current_stage(self) -> Dict[str, Any]:
        return self.stages[self.current_stage_idx]

    @property
    def current_stage_name(self) -> str:
        return self.current_stage.get("name", f"stage_{self.current_stage_idx}")

    def record_episode(self, success: bool) -> None:
        self.success_history.append(float(success))

    @property
    def rolling_success_rate(self) -> float:
        if len(self.success_history) == 0:
            return 0.0
        return float(np.mean(self.success_history))

    def should_advance(self) -> bool:
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already in final stage

        if len(self.success_history) < self.window_size:
            return False

        threshold = self.current_stage.get("advance_threshold", 0.70)
        return self.rolling_success_rate >= threshold

    def advance_stage(self) -> Optional[Dict[str, Any]]:
        """Advances to next stage if threshold met. Returns new stage config or None."""
        if self.should_advance():
            self.current_stage_idx += 1
            self.success_history.clear()
            new_stage = self.current_stage
            print(f"\n[CurriculumManager] 🎉 Advanced to stage: '{new_stage.get('name')}'")
            print(f"  Task: {new_stage.get('task')} | Scouts: {new_stage.get('n_scouts')}, "
                  f"Carriers: {new_stage.get('n_carriers')}, Analyzers: {new_stage.get('n_analyzers')}, "
                  f"Fault Prob: {new_stage.get('fault_prob')}\n")
            return new_stage
        return None
