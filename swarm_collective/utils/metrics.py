"""
metrics.py — Telemetry & metrics for swarm collective intelligence.

Tracks:
- Task success rate & completion time
- Communication volume (bits/step, msg norm)
- "Chattiness" penalty vs silence ratio
- Fault resilience & recovery rate
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class SwarmMetricsTracker:
    def __init__(self):
        self.reset_all()

    def reset_all(self):
        self.episodes_completed = 0
        self.successes          = 0
        self.episode_lengths    = []
        self.episode_rewards    = []
        self.comm_penalties     = []
        self.message_norms      = []
        self.attention_entropies= []
        self.fault_recoveries   = []

    def record_step_telemetry(self, comm_penalty: float, attn_matrix: np.ndarray, msg_norms: np.ndarray):
        self.comm_penalties.append(comm_penalty)
        self.message_norms.extend(msg_norms.tolist())

        # Shannon entropy of attention distribution across senders (measures targeted vs broadcast comms)
        if attn_matrix is not None and attn_matrix.shape[0] > 1:
            # attn_matrix: (N, N), each row sums to 1 (or 0 if isolated)
            for row in attn_matrix:
                row_valid = row[row > 1e-6]
                if len(row_valid) > 1:
                    entropy = -np.sum(row_valid * np.log2(row_valid))
                    self.attention_entropies.append(entropy)

    def record_episode_end(self, success: bool, length: int, total_reward: float, had_fault: bool = False):
        self.episodes_completed += 1
        if success:
            self.successes += 1
        self.episode_lengths.append(length)
        self.episode_rewards.append(total_reward)
        if had_fault:
            self.fault_recoveries.append(float(success))

    def get_summary(self) -> Dict[str, float]:
        if self.episodes_completed == 0:
            return {}

        return {
            "success_rate":       self.successes / self.episodes_completed,
            "mean_episode_len":   float(np.mean(self.episode_lengths)),
            "mean_reward":        float(np.mean(self.episode_rewards)),
            "mean_msg_norm":      float(np.mean(self.message_norms)) if self.message_norms else 0.0,
            "mean_comm_penalty":  float(np.mean(self.comm_penalties)) if self.comm_penalties else 0.0,
            "mean_attn_entropy":  float(np.mean(self.attention_entropies)) if self.attention_entropies else 0.0,
            "fault_recovery_rate":float(np.mean(self.fault_recoveries)) if self.fault_recoveries else 1.0,
        }
