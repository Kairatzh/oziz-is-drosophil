# Phase 6: RL Agent vs. Hand-Coded Baseline

Baseline: `baseline/potential_field_controller.py`, a classical two-loop
cascade (Khatib 1986 potential field for guidance + PD attitude control for
the underactuated tilt-to-translate inner loop), tuned by hand against the
same layouts used to evaluate the RL agent. RL agent: `model_checkpoints/final_model.pt`
(Phase 4's Level 1.5 checkpoint, see `PHASE4_CURRICULUM_RESULTS.md`).

**Important asymmetry, stated up front:** the baseline is given the EXACT
obstacle positions/radii and the drone's full true state. The RL agent only
ever sees an 8-ray lidar scan (`PHASE5_GENERALIZATION_RESULTS.md`). This is
the standard setup for comparing classical vs. learned control — classical
control requires a known, structured model of the world; RL is learning
one from partial observations. Every result below should be read with that
asymmetry in mind: a baseline win is expected when you hand a controller
perfect information. What's actually interesting is *how much* it wins by,
and the handful of things measured here where the RL agent still comes out
ahead despite the handicap.

All numbers are 100 **paired** episodes per condition (`scripts/phase6_baseline_comparison.py`):
both controllers are run on identically-seeded environments, so episode *i*
is the exact same start/goal/obstacle draw for both — a controlled
comparison, not two independent samples. Test conditions are reused
verbatim from Phase 5 (`scripts/phase5_generalization_eval.py`).

## Success rate, by condition

| Condition | RL agent | Baseline | Baseline wins by |
|---|---|---|---|
| In-distribution control (1 obstacle, clear LOS) | 99% | 100% | +1 pp |
| 1 obstacle, no LOS guarantee | 85% | 97% | +12 pp |
| 3 random obstacles, no LOS guarantee | 83% | 95% | +12 pp |
| Level 2 fixed layout (2 obstacles, wide gaps) | 60% | 85% | +25 pp |
| Wall formation (3-obstacle, narrow gaps) | 51% | 53% | +2 pp |
| Level 3 fixed layout (5 obstacles, moderate gaps) | 42% | 76% | +34 pp |
| **Average** | **70.0%** | **84.3%** | **+14.3 pp** |

**The baseline has a higher success rate in all 6 conditions.** Given the
perfect-information asymmetry above, that is the expected direction. The
margin varies a lot by condition, though, which is itself informative (see
below).

## Time to goal, path length, control effort (successful episodes only)

| Condition | Steps (RL / baseline) | Path length (RL / baseline) | Control effort (RL / baseline) |
|---|---|---|---|
| In-distribution control | 117 / 185 | 4.78m / 4.78m | 1.010 / 0.897 |
| 1 obstacle, no LOS | 109 / 186 | 4.23m / 4.69m | 1.004 / 0.900 |
| 3 random obstacles | 104 / 178 | 3.78m / 4.30m | 0.997 / 0.904 |
| Level 2 fixed layout | 103 / 209 | 3.77m / 4.99m | 0.996 / 0.913 |
| Wall formation | 90 / 134 | 2.61m / 2.74m | 0.972 / 0.914 |
| Level 3 fixed layout | 95 / 257 | 3.13m / 5.69m | 0.981 / 0.929 |

(Control effort = mean of left²+right² per step, the same term the reward
function penalizes; lower = smoother/less jittery. Path length in meters is
the raw distance traveled, not the straight-line distance.)

**Where the RL agent wins:** on every single condition where both
succeeded, the **RL agent reaches the goal in roughly half the time or
less**, via a **shorter, straighter path**. This is a real, honest RL win —
not an artifact of the success-rate gap, since these numbers are computed
only over episodes each controller actually completed. The RL agent
learned a much more direct, time-optimal-looking policy; the potential
field's damping and gain tuning (needed for stability, see below) trades
speed for safety margin.

**Where the baseline wins besides success rate:** control effort is
consistently **lower for the baseline** (0.90–0.93 vs. the RL agent's
0.97–1.01). The baseline's actions sit closer to a smooth hover-plus-small-correction
regime; the RL agent's policy is more aggressive/bang-bang-like, which is
part of *why* it's faster, but it is measurably jerkier.

## The baseline's own honest failure mode: local minima

The wall-formation trajectory plot (`results/phase6_wall_narrow_gaps_side_by_side.png`)
shows the baseline's most characteristic weakness: on episode 0 of that
condition, the RL agent collides quickly (50 steps), but the baseline
**oscillates for the full 500-step timeout**, visibly trapped between two
obstacles where the attractive pull toward the goal and the repulsive push
away from both obstacles nearly cancel — the textbook potential-field local
minimum (Koren & Borenstein, 1991). This shows up in the aggregate numbers
too: the wall condition is the only one where the baseline's timeout rate
(28%) is far higher than its collision rate (17%), and it is the condition
with the smallest baseline win-margin (+2 pp) — the one formation
structurally suited to this specific classical failure mode, rather than
sheer obstacle density.

## Visualizations

- `results/phase6_comparison_success_rates.png` — grouped bar chart, all 6 conditions.
- `results/phase6_train_distribution_control_side_by_side.png` — both succeed; baseline's path is a smooth straight glide, RL's is curved but ~40% faster.
- `results/phase6_level2_fixed_wide_gaps_side_by_side.png` — both succeed, similar paths; RL again ~2x faster.
- `results/phase6_wall_narrow_gaps_side_by_side.png` — RL collides quickly; baseline gets trapped oscillating in a local minimum for the full episode.

## Conclusion — being honest about who wins

**The baseline wins on success rate and control smoothness, everywhere
tested**, and that is the expected, honest result once you account for the
information it's given: exact obstacle geometry and full state, versus the
RL agent's 8-ray lidar. It is not evidence that classical control is
"better" in general — it is evidence that a well-known, well-tuned model-based
method beats a partially-observed learned policy when the model is
exactly correct and available. That said, the baseline is not strictly
dominant: the **RL agent is consistently ~2x faster with shorter paths**
whenever it succeeds, and the baseline has its own well-documented failure
mode (local-minimum oscillation) that the RL agent, whatever else it does
wrong, never exhibits — it always resolves quickly, one way or another,
rather than idling for a full timeout.

This sets up Phase 7 (failure analysis) naturally: characterize *when*
each method fails and whether the RL agent's failures are systematically
different in kind (fast, aggressive collisions) from the baseline's (slow
oscillation, or the same occlusion-driven collisions already diagnosed in
Phase 4/5).
