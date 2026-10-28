# Phase 4 Curriculum Learning — Results

Deterministic 100-episode decision evals (see `eval-batch-size` convention: 100
episodes for accept/reject decisions, 20-episode "pulses" only for quick
in-training checks). "Collision" below is always the clean/exploration-free
rate from `scripts/eval_outcome_breakdown.py`-style deterministic eval, not
the exploration-inflated training rollout mix.

| Level | Obstacles | Success | Collision | Outcome |
|---|---|---|---|---|
| 1 (no obstacles) | 0 | high | — | solved |
| 1.5 (single obstacle, clear LOS) | 1 | **99%** | 0% | solved at update 20, warm-started Level 2 |
| 2 (wide gaps) | 2 | **61%** | **34%** | plateaued — accepted, `phase4_level2_from15_update0050.pt` |
| 3 (moderate gaps) | 5 | **44%** | **38%** | plateaued — accepted, `phase4_level3_from_level2_update0100.pt` |

## Level 3: accepted at 44% success / 38% collision

Level 3 was warm-started from the accepted Level 2 checkpoint and run for 100
updates (`scripts/train_level3_warmstart.py`, logs in
`results/logs/level3_from_level2_u1-50.log` and `_u51-100.log`). Both blocks
showed the same plateau signature as Level 2 — entropy converging smoothly
with no re-spike, success rate flat/noisy across the block — but Level 3
reached it in half the updates Level 2 needed (100 vs. Level 2's ~150).

`scripts/diagnose_level3_failure_geometry.py` (100 fresh deterministic
episodes, `results/logs/level3_failure_geometry.log`) confirms this is a
broad-based avoidance-capability limit, not a fixable geometry problem:

- **68% of failures** occur when the straight start→goal line is occluded by
  an obstacle (81% failure rate when occluded vs. 34% when clear).
- **No single obstacle dominates** — each obstacle's share of collisions
  roughly tracks its share of path exposure (obs4, the center obstacle most
  often in the way, accounts for 32% of collisions vs. 26% exposure; no
  obstacle is wildly over- or under-represented).
- Mean crash-to-obstacle-surface distance is 0.14 m, confirming these are
  genuine surface hits, not measurement artifacts.

**Conclusion:** Level 3 is accepted at its current performance. Further
training at Level 3 is not expected to move the needle — this is the same
architecture-limited ceiling diagnosed at Level 2
(`level2-plateau-diagnosis` memory), now confirmed to generalize to a harder
5-obstacle field.

## Level 4: skipped

Level 4 (dense obstacle field, narrow gaps) is the next curriculum step in
`PROJECT_SPEC.md`, but it is being skipped rather than trained.

**Reasoning:** curriculum progression only worked through Level 1.5 (single
obstacle, guaranteed clear line of sight to goal — 99% success). From Level 2
onward, every additional increase in obstacle count/density has *not* been
met with a corresponding increase in agent capability:

- Level 2 (2 obstacles): 61% success, 34% collision — plateaued.
- Level 3 (5 obstacles): 44% success, 38% collision — same plateau signature
  (converged entropy, flat success rate), reached in half the updates.

Going from Level 2 to Level 3 cost 17 points of success rate and the collision
rate held flat, meaning the extra difficulty translated directly into more
failures rather than being absorbed by learning. Level 4's denser field with
narrower gaps only increases occlusion frequency and lowers corridor
clearance — exactly the two variables the Level 3 failure-geometry diagnostic
identified as the failure driver. There is no evidence from Levels 2 or 3
that more updates, warm-starting, or reward tweaks break this ceiling (four
independent Level 2 attempts all landed in the same 30-34% collision band;
see `level2-plateau-diagnosis` memory), so Level 4 would be expected to
perform *worse* than Level 3, not better, at a much higher training-time cost.

**Decision:** end the obstacle curriculum at Level 3. Do not train Level 4.

## Primary trained agent for Phases 5-7

Because Levels 2 and 3 plateaued below Level 1.5's performance rather than
surpassing it, the highest-fidelity policy produced by this curriculum is
still the **Level 1.5 checkpoint** (`phase4_level1_5_final.pt`: 99% success,
0% collision on a single-obstacle, clear-line-of-sight task). The Level 2/3
checkpoints demonstrate exposure to more obstacles but at strictly lower and
architecture-limited success rates, so they are not an improvement to build
on for the phases ahead.

`phase4_level1_5_final.pt` has been copied to `model_checkpoints/final_model.pt`
— the one checkpoint path exempted from `.gitignore` — and is the agent
Phases 5 (generalization testing), 6 (baseline comparison), and 7 (failure
analysis) will use. Phase 5's generalization tests against multi-obstacle
courses are expected to surface the same avoidance ceiling documented above;
that is itself the generalization result to report, not a reason to retrain.
