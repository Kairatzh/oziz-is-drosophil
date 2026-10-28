# Phase 7: Failure Analysis

Systematic characterization of when and how `model_checkpoints/final_model.pt`
fails, using `scripts/phase7_failure_analysis.py`. Four analyses, all
read-only (no training), all deterministic evaluation.

## 1. Obstacle-count sweep: gradual erosion, no cliff

Random obstacle placement (no line-of-sight guarantee), count 1→6, 100
episodes each:

| Obstacles | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Success | 85% | 87% | 83% | 74% | 59% | 54% |

![obstacle count sweep](results/phase7_obstacle_count_sweep.png)

Performance degrades **gradually**, not catastrophically — there is no
sudden cliff at "more than 1 obstacle" (the training count). It holds
roughly flat through 3 obstacles, then erodes steadily. This is consistent
with Phase 5's finding that *count* alone isn't what hurts most — the
agent generalizes reasonably to more random obstacles as long as they
don't systematically block the path.

## 2. Corridor-clearance sweep: a clean dose-response curve at the training boundary

Single obstacle (r=0.8, matching training), sweeping how much the straight
start→goal line is blocked, from clearly clear to clearly occluded. Training
(Level 1.5) only ever showed clearance ≥ +0.5m (`GOAL_LOS_MARGIN`):

| Clearance | −1.2m | −0.8m | −0.5m | −0.2m | 0.0m | +0.2m | **+0.5m (boundary)** | +0.8m | +1.2m |
|---|---|---|---|---|---|---|---|---|---|
| Success | 1% | 1% | 6% | 28% | 41% | 52% | **76%** | 96% | 99% |

![clearance sweep](results/phase7_clearance_sweep.png)

**This is the clearest boundary effect found in the whole project.** Success
climbs almost linearly from 1% (deeply occluded) to 99% (clearly open),
crossing the exact training boundary (+0.5m) at 76% — right in the middle
of the transition, not at a plateau. The agent was never trained on
anything below this line, and its performance falls off a smooth ramp
exactly where that training data ends, rather than degrading unpredictably
or holding up unexpectedly well past the boundary. This is about as clean
a confirmation of "the model learned the training distribution and nothing
reliably beyond it" as this project has produced.

## 3. Catastrophic vs. graceful failure

Pooled over 500 held-out episodes (the 5 non-control Phase 5/6 conditions):
167 collisions, 12 timeouts.

**Collisions are fast and blind, not tentative:**
- Impact speed: mean 3.88 m/s, median 3.98 m/s — nearly **2x** the 2.10 m/s
  mid-flight cruise speed of successful episodes.
- **91% of collisions hit faster than typical cruise speed** — there is no
  slow-down-and-bump pattern; the agent is often still accelerating.
- In the last 10 steps before impact, 126/167 (75%) were *speeding up*, only
  29/167 (17%) were braking.
- The lidar clearly sees it coming (mean min-ray drops from 0.26 ten steps
  out to 0.04 at impact) — this matches Phase 4's `diagnose_lidar_at_collision.py`
  finding that the sensor works; the agent has the information and still
  collides at speed.
- **100% of collisions happen within the first 30% of the episode's step
  budget** — the agent doesn't wander into trouble late; it fails fast,
  early in the approach.

**Conclusion: failures are catastrophic, not graceful.** The agent does not
slow down or hedge when uncertain; it commits to a trajectory and either it
works or it doesn't, with collisions typically happening at or above normal
cruising speed. This is a real limitation for a task in the real world 
where a graceful, cautious failure mode (slowing down near clutter) would
be much safer than a confident high-speed collision.

**Timeouts, by contrast, are the opposite: totally passive.** All 12 pooled
timeouts ended parked (mean final speed 0.00 m/s) within 0.43m of the goal
on average — the agent gets close, stops, and simply never satisfies the
speed-limited arrival condition. This is a "gives up near the goal" failure,
not a "lost and wandering" one.

## 4. Wall formation deep-dive (known failure case)

Re-examined the Phase 6 wall formation (3-obstacle vertical wall, ~0.6-0.7m
gaps) at higher resolution: 51% success, 44% collision, 5% timeout, 0%
out-of-bounds.

The 5 timeout episodes average only 7.6 x-velocity sign changes — **not**
the sustained back-and-forth oscillation the potential-field baseline
showed in the same formation (`PHASE6_BASELINE_COMPARISON_RESULTS.md`).
The RL agent's rare non-collision failures here look like **stalls near an
obstacle**, not an active local-minimum trap. This is a meaningful
qualitative difference between the two approaches' failure mechanisms even
on the identical obstacle layout: the classical controller gets
mathematically stuck oscillating between competing force fields; the RL
agent, when it doesn't collide, just stops.

## Representative failure GIFs

| GIF | What it shows |
|---|---|
| `phase7_catastrophic_collision.gif` | High-speed (~6 m/s) collision in a dense 5-obstacle field — no braking. |
| `phase7_graceful_collision.gif` | The slowest collision found in the same pool — closest thing to a "soft" failure this agent produces. |
| `phase7_wall_formation_collision.gif` | Wall formation: fast collision into one of the three obstacles. |
| `phase7_wall_formation_timeout.gif` | Wall formation: a stall/give-up case, not oscillation. |
| `phase7_boundary_clearance_failure.gif` | A failure right at the edge of the training distribution (clearance +0.3m, just inside the 0.5m boundary). |
| `phase7_deep_occlusion_failure.gif` | A failure deep outside the training distribution (clearance −1.2m, fully occluded). |

## Overall picture

Three independent analyses (obstacle count, corridor clearance, and the
earlier Phase 5 fixed-layout tests) all point at the same underlying
limitation: **the agent learned to fly cleanly to a visible goal and to
react to one obstacle it never had to route around, and that skill
transfers gradually and predictably outward from that exact training
distribution — well for small deviations, poorly for large ones, with the
clearance axis showing the boundary most cleanly.** Failures are fast,
high-speed, and early-in-episode rather than slow or exploratory — the
agent commits rather than hedges. This is consistent with, and sharpens,
every prior phase's "architecture/training-distribution limit" conclusion
(`PHASE4_CURRICULUM_RESULTS.md`, `PHASE5_GENERALIZATION_RESULTS.md`): it is
not a mysterious or inconsistent failure pattern, but a predictable
generalization gradient around a narrow training distribution.
