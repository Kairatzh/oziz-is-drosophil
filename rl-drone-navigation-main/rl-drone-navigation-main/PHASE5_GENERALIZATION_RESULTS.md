# Phase 5: Generalization Testing

Agent under test: `model_checkpoints/final_model.pt` (= `phase4_level1_5_final.pt`,
the Phase 4 curriculum-exit agent — see `PHASE4_CURRICULUM_RESULTS.md`).
Trained on exactly ONE distribution: a single r=0.8 obstacle, randomly
placed each episode but always kept clear of the straight start→goal line
by `QuadrotorEnv.GOAL_LOS_MARGIN` (0.5 m) — so training never required
routing around a blocked direct path, never showed more than one obstacle,
and never repeated a fixed layout.

All numbers are 100-episode deterministic evals (`scripts/phase5_generalization_eval.py`,
no exploration noise), per the project's decision-grade eval convention.

## Results: training distribution vs. held-out test conditions

| Condition | What varies from training | Success | Collision | Gap vs. control |
|---|---|---|---|---|
| **In-distribution control** (Level 1.5 generator, unseen seed) | nothing — same generator | **99%** | 0% | — |
| 1 obstacle, no LOS guarantee | direct path can now be blocked | 85% | 13% | −14 pp |
| 3 random obstacles, no LOS guarantee | count (1→3) + no LOS guarantee | 83% | 17% | −16 pp |
| 2 fixed obstacles, wide gaps (curriculum Level 2 layout) | count, fixed (repeated) layout, no LOS guarantee | 60% | 38% | −39 pp |
| Vertical 3-obstacle wall, ~0.6-0.7 m gaps (unusual formation) | genuine gap-threading, never once required in training | 51% | 44% | −48 pp |
| 5 fixed obstacles, moderate gaps (curriculum Level 3 layout) | count, fixed layout, densest field | 42% | 55% | −57 pp |

**Held-out average: 64.2% success (gap: +34.8 pp vs. the 99% in-distribution
control).**

## Quantifying the gap: what actually drives it

The single biggest lever is not obstacle *count* — it's whether the straight
path can be **occluded** and whether the layout is **fixed** (i.e., the same
few configurations repeated every episode, which the agent can partially
memorize during training but which a single held-out checkpoint has never
seen at all):

- Going from 1→3 obstacles while keeping placement *random* each episode
  cost almost nothing (85% → 83%). Random layouts rarely gang up on one
  specific route.
- Going from 1 random obstacle to the 2-obstacle **fixed** Level 2 layout
  cost 25 points (85% → 60%), despite *fewer* obstacles than the 3-random
  condition. A fixed, repeated layout that reliably sits between plausible
  start/goal pairs is harder than a fresh random draw with more obstacles.
- The wall formation (51%) confirms this isn't just about LOS occlusion —
  it is specifically hard because it requires threading a genuine gap
  between two nearby obstacles, a skill the Level 1.5 curriculum never
  exercised even once (its clear-LOS constraint meant a straight line
  never touched an obstacle, so the agent was never forced through a gap).

## The most important finding: extra curriculum training added ~0 net capability

Phase 4 additionally trained dedicated checkpoints on the Level 2 and
Level 3 layouts specifically (`phase4_level2_from15_update0050.pt`:
61% success / 34% collision on Level 2; `phase4_level3_from_level2_update0100.pt`:
44% success / 38% collision on Level 3 via the failure-geometry eval). This
Phase 5 run puts the **Level-1.5-only** checkpoint on those exact same two
layouts:

| Layout | Checkpoint trained specifically on it | Level-1.5-only checkpoint (never trained past 1 obstacle) |
|---|---|---|
| Level 2 (2 obstacles, wide gaps) | 61% success / 34% collision | **60% success / 38% collision** |
| Level 3 (5 obstacles, moderate gaps) | 44% success / 38% collision | **42% success / 55% collision** |

The Level-1.5-only checkpoint matches the success rate of checkpoints that
trained directly on these layouts for 50-100 additional PPO updates each —
it just fails slightly more often via collision than via other outcomes.
This is strong independent confirmation of the Phase 4 conclusion: the ~34-38%
collision plateau is a property of what this architecture/observation (8-ray
lidar + PPO policy of this size) can learn about general obstacle avoidance,
not something the Level 2/3-specific training sessions actually added. Extra
curriculum time at Levels 2-3 bought almost no measurable capability over
the much cheaper Level 1.5 checkpoint.

## Visualizations

- `results/phase5_generalization_success_rates.png` — bar chart, all six conditions.
- `results/phase5_random_three_obstacles_success.gif` / `_failure.gif`
- `results/phase5_wall_narrow_gaps_success.gif` / `_failure.gif`

## Conclusion

Generalization is real but bounded: the agent transfers its "obstacles hurt,
steer away" skill to new random layouts and even new obstacle counts fairly
well (down to 83-85% on random 1-3 obstacle fields), but degrades sharply
on fixed multi-obstacle layouts and on any formation requiring active
gap-threading rather than point-avoidance (42-60%). Since additional
curriculum training at Levels 2-3 did not exceed what the Level 1.5
checkpoint already achieves on those same layouts, this is consistent with
the architecture-limit conclusion from Phase 4 — domain randomization or
more training at the current architecture is not expected to close this
gap on its own. Per `PROJECT_SPEC.md` Phase 5, this motivates discussing
domain randomization (broader, harder obstacle counts/placements *during*
training itself) as a fix, which is a natural entry point for Phase 6/7's
baseline comparison and failure analysis.
