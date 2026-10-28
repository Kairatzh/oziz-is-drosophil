# Autonomous Drone Navigation via Deep Reinforcement Learning

A 2D quadrotor learns to fly itself to a goal and avoid obstacles, from
nothing but trial and error — no physics equations given to the learner,
no hand-coded control rules. Everything in this repo is built from
scratch: the physics simulation, the PPO reinforcement-learning algorithm,
and the curriculum that teaches the agent progressively harder tasks.

![progression demo](results/phase8_progression_demo.gif)

*Four real checkpoints, four real episodes: a randomly-initialized network
that free-falls out of bounds, the same network 10 PPO updates later
learning to fight gravity, the fully-trained Phase 3 agent flying straight
to a goal, and the final agent (`final_model.pt`) also steering around an
obstacle.*

## Why this project

I build drones by hand as a hobby, and my other simulation projects
(`rocket-sim`, `kalman-sensor-fusion`, `gnc-closed-loop`,
`fea-structural-failure`) are all classical: analytical physics, Kalman
filters, LQR optimal control. Every one of those required *me* to derive
the control law. This project asks the opposite question: can a neural
network discover a working control policy purely from trial and error,
with no knowledge of the physics equations governing the very vehicle it's
flying? Phase 6 answers a second, related question directly: when a
correct physics model *is* available, how much does hand-derived control
still beat a learned one, and at what cost?

## Table of contents

- [The environment](#the-environment)
- [Method: PPO from scratch](#method-ppo-from-scratch)
- [Curriculum learning](#curriculum-learning-phase-4)
- [Generalization testing](#generalization-testing-phase-5)
- [Baseline comparison](#baseline-comparison-phase-6)
- [Failure analysis](#failure-analysis-phase-7)
- [Limitations](#limitations)
- [How to run](#how-to-run)
- [Project structure](#project-structure)

## The environment

`env/quadrotor_env.py` — a planar (2D) quadrotor, built without any RL
framework (no Gym/Gymnasium). The drone is a rigid rod with a rotor at
each end; each rotor produces a thrust force along the body's own "up"
direction. This is what makes the control problem hard: there are only
**two scalar controls** (left thrust, right thrust) for **three degrees of
freedom** (x, z, θ) — the drone is underactuated and cannot move sideways
without first rotating to tilt its thrust vector, exactly like a real
quadrotor.

**Dynamics** (semi-implicit Euler integration at 50 Hz):

```
total_thrust = left + right
ax = (-total_thrust·sin(θ) - drag·vx) / mass
az = ( total_thrust·cos(θ) - mass·g - drag·vz) / mass
α  = (right - left)·arm_length / inertia
```

Torque depends only on the thrust *difference*, never on the current
orientation — both rotor forces are always perpendicular to the rod.

**Observation (16-dim):** `[x, z, vx, vz, θ, ω, dx, dz, lidar_0..lidar_7]`
— own state, offset to the goal, and an 8-ray simulated lidar scan (fixed
world-frame angles, normalized distance to the nearest obstacle, capped at
3m). The lidar is the agent's *only* source of obstacle information; it
was added mid-Phase-4 after diagnosing that the original 8-dim
state+goal-only observation gave the agent no way to perceive obstacles at
all.

**Reward** (7 terms, tuned iteratively — see `env/quadrotor_env.py` for the
full derivation of each):

| Term | Purpose |
|---|---|
| Progress shaping | potential-based, rewards net distance closed toward the goal |
| Time penalty | discourages loitering |
| Control-effort penalty | discourages jittery, max-thrust control |
| Velocity-toward-goal | live gradient even at near-zero speed |
| Distance penalty | persistent pull toward the goal regardless of motion |
| Obstacle-clearance penalty | dense, local "steer away" signal near obstacles |
| Goal-proximity bonus | obstacle- and line-of-sight-aware; fixes a "stalls just short of goal" failure mode without rewarding loitering next to an occluded goal |

Episodes end in **success** (within the goal radius, arriving slowly
enough to count as "landed" rather than "passing through"), **collision**,
**out-of-bounds**, or **timeout** (500 steps).

## Method: PPO from scratch

`ppo/` implements Proximal Policy Optimization directly in PyTorch — no
`stable-baselines3`, no RL library of any kind.

**Actor-critic split.** Two independent small MLPs (2 hidden layers of 64
units, tanh activations, orthogonal initialization) rather than a shared
trunk — this keeps the actor/critic conceptual split visible in the code.
The actor (`GaussianActor`) outputs a mean and a (state-independent,
learned) standard deviation per action dimension — a diagonal Gaussian
policy, since the two thrust commands are continuous, not a small discrete
set a DQN-style argmax could enumerate.

**Generalized Advantage Estimation (GAE(λ)).** Advantage answers "was this
action better or worse than the critic expected from this state?" — it's
what lets PPO learn from *relative* quality rather than raw returns. GAE
blends the low-variance/high-bias one-step TD residual with the
high-variance/low-bias Monte-Carlo return:

```
delta_t   = r_t + γ·V(s_{t+1})·(1 - done_t) - V(s_t)
A_t (GAE) = delta_t + γ·λ·(1 - done_t)·A_{t+1}
returns_t = A_t + V(s_t)                      # value-function regression target
```

λ=0.95, γ=0.99 (the standard values from the original GAE paper, Schulman
et al. 2015).

**The clipping objective.** Let `ratio_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`
— how much more (or less) likely the *new* policy makes the action actually
taken, versus the policy that collected the data. The naive objective
`ratio_t · A_t`, fully optimized, lets the ratio grow unbounded whenever
`A_t > 0`: "if this action was good, make it infinitely more likely right
now" — which overshoots wildly and can collapse the policy. PPO instead
optimizes:

```
L^CLIP(θ) = E[ min( ratio_t(θ)·A_t,  clip(ratio_t(θ), 1-ε, 1+ε)·A_t ) ]
```

with ε=0.2. Once `ratio_t` moves outside `[1-ε, 1+ε]`, the clipped branch
stops contributing gradient in the direction that would push it further
away — capping how far a single update can move the policy from the
data-collecting one. This is what "proximal" means, and it's why PPO can
reuse the same batch of rollout data for multiple epochs of gradient
updates (10 here) without the instability that would cause for a
non-clipped policy gradient.

**Value loss and total objective:**

```
L^VF(φ) = MSE( V_φ(s_t), returns_t )
L(θ,φ)  = L^CLIP(θ) - c1·L^VF(φ) + c2·H[π_θ](s_t)
```

The entropy bonus `H[π_θ]` (coefficient c2) rewards a *less* certain
policy, counteracting the natural tendency to collapse to a deterministic
action before exploration has found a good strategy. `GaussianActor`
clamps `log_std` to a fixed range — without this, the entropy bonus gives
the optimizer a persistent, unopposed incentive to keep growing the
standard deviation, which one Phase 4 run showed causing entropy to climb
6x over 325 updates with success rate plateaued well below target.

**Training loop:** collect a fixed-length rollout → compute GAE
advantages/returns → run `n_epochs` of minibatch gradient updates on the
clipped objective → repeat. Validated first on a hand-rolled CartPole
(Phase 2) before ever touching the drone environment.

## Curriculum learning (Phase 4)

Sparse rewards make dense obstacle fields nearly unlearnable from a random
initial policy: almost every random action sequence crashes within the
first second, so the agent essentially never experiences a success to
learn from. Curriculum learning fixes this by mastering an easy version of
the task first and increasing difficulty only once each stage is reliably
solved — the same agent (actor + critic weights, optimizer state) carries
over unchanged between levels, so skill actually transfers stage to stage.

| Level | Obstacles | Success | Collision | Outcome |
|---|---|---|---|---|
| 1 (no obstacles) | 0 | high | — | solved |
| 1.5 (single obstacle, clear line of sight) | 1 | **99%** | 0% | solved at update 20 — chosen as the primary agent |
| 2 (wide gaps) | 2 | 61% | 34% | plateaued, accepted |
| 3 (moderate gaps) | 5 | 44% | 38% | plateaued at the same ceiling, confirmed by failure-geometry diagnostic |

Levels 2 and 3 both plateaued **below** Level 1.5's performance rather than
building on it — four independent Level 2 attempts (different warm-starts,
reward tweaks, training durations) all landed in the same 30-34% collision
band. That is the signature of an **architecture/observation-space
limit** (an 8-ray lidar and this network size can only learn so much
general obstacle avoidance), not undertraining or bad reward shaping.
Level 4 (dense field, narrow gaps) was therefore **skipped** — it would
only worsen the two variables already diagnosed as the failure driver
(occlusion frequency, corridor clearance), with no evidence more training
breaks the ceiling. `model_checkpoints/final_model.pt` is the **Level 1.5
checkpoint**, not a later one: it's the last stage that actually mastered
its task rather than plateauing below its predecessor. Full detail:
`PHASE4_CURRICULUM_RESULTS.md`.

## Generalization testing (Phase 5)

`final_model.pt` was trained on exactly one distribution — a single
obstacle, always placed clear of the straight line to the goal. Phase 5
tested it against six conditions never seen in training:

| Condition | Success |
|---|---|
| In-distribution control | **99%** |
| 1 obstacle, no clear-path guarantee | 85% |
| 3 random obstacles, no guarantee | 83% |
| Fixed 2-obstacle layout (Level 2) | 60% |
| Wall formation (novel gap-threading) | 51% |
| Fixed 5-obstacle layout (Level 3) | 42% |

Held-out average: **64.2%**, a 34.8-point gap. The gap is driven less by
obstacle *count* than by whether the layout is **fixed/repeated** and
whether the drone must actually **thread a gap** rather than just steer
around a point obstacle — 3 random obstacles (83%) held up far better than
a fixed 2-obstacle layout (60%). Most notably: on the curriculum's own
Level 2/3 layouts, this Level-1.5-only checkpoint (60%/42%) nearly matches
the checkpoints that were *specifically trained* on those layouts for
50-100 more updates (61%/44%) — independent confirmation, via a completely
different checkpoint, that the extra curriculum training bought almost no
net capability. Full detail: `PHASE5_GENERALIZATION_RESULTS.md`.

## Baseline comparison (Phase 6)

`baseline/potential_field_controller.py` — a classical, hand-tuned
controller with no learning at all: an outer **artificial potential
field** (Khatib, 1986) combining an attractive pull toward the goal with a
repulsive push away from nearby obstacles, feeding a desired acceleration
into an inner **PD attitude loop** that solves the same underactuated
tilt-to-translate problem the RL agent has to. Unlike the RL agent
(8-ray lidar only), the baseline is given the exact obstacle positions and
the drone's full true state — the standard classical-control assumption,
and a deliberate, disclosed asymmetry.

100 **paired** episodes per condition (identical seeds, so each episode is
the same start/goal/obstacle draw for both controllers):

| | RL agent | Baseline |
|---|---|---|
| Average success rate (6 conditions) | 70.0% | **84.3%** |
| Speed when successful | **~2x faster** | slower |
| Path length when successful | **shorter** | longer |
| Control effort (smoothness) | jerkier | **smoother** |

**The baseline wins success rate in all 6 conditions** — expected, given
it gets a perfect model of the world and the RL agent doesn't. What's
genuinely interesting: the RL agent is consistently **about twice as fast**,
via a shorter, more direct path, whenever it succeeds. And the
baseline has its own honest, textbook weakness that the RL agent never
exhibits:

> **On the wall-gap formation, the potential-field controller gets
> trapped oscillating in a local minimum for the entire 500-step episode**
> — the attractive pull toward the goal and the repulsive push from two
> nearby obstacles nearly cancel, and it never resolves one way or the
> other (Koren & Borenstein, 1991). The RL agent, on the same formation,
> either succeeds or fails fast; it never idles.

![wall formation: RL collides fast, baseline oscillates for the full episode](results/phase6_wall_narrow_gaps_side_by_side.png)

Full detail, including why the RL agent's speed advantage is real and not
an artifact of the success-rate gap: `PHASE6_BASELINE_COMPARISON_RESULTS.md`.

## Failure analysis (Phase 7)

Four systematic diagnostics on `final_model.pt`, all read-only. The
headline result:

### The corridor-clearance dose-response curve

Sweeping exactly how much the straight path to the goal is blocked by a
single training-radius obstacle, from clearly open to clearly blocked:

![corridor clearance sweep](results/phase7_clearance_sweep.png)

Training (Level 1.5) only ever showed clearance ≥ +0.5m (the dashed line).
Success forms a **smooth S-curve** from 1% (deeply occluded) to 99%
(clearly open), crossing that exact training boundary at 76% — right in
the middle of the transition, not at a plateau on either side. This is the
single cleanest result in the project: the agent generalizes *outward*
from its training distribution in a smooth, predictable gradient — well
close to what it saw, poorly far from it — rather than failing randomly or
holding up unexpectedly well past the boundary.

### Catastrophic, not graceful

Pooling 167 collisions across held-out conditions: **91% hit faster than
the agent's own typical cruising speed**, and **100% happened within the
first 30% of the episode's step budget**. The lidar clearly sees the
obstacle coming (mean reading drops from 0.26 to 0.04 in the final 10
steps) — the agent has the information and still fails to brake. Timeouts
are the opposite extreme: 100% end fully stopped near the goal, a
"gives up" failure rather than a "lost" one. On the wall formation
specifically, the RL agent's rare non-collision failures are stalls, not
the sustained oscillation the potential-field baseline showed on the
identical layout (see above) — a genuine difference in *how* the two
approaches fail on the same obstacle course.

Also swept: obstacle count (1→6 random obstacles), a gradual 85%→54%
erosion with no cliff. Full detail and all 6 representative failure GIFs:
`PHASE7_FAILURE_ANALYSIS_RESULTS.md`.

## Limitations

- **Narrow training distribution.** The primary agent trained on exactly
  one obstacle-placement rule (always clear line of sight). Everything in
  Phases 5-7 is a consequence of that: smooth but real degradation outside
  it.
- **Architecture-limited ceiling, not a reward or undertraining problem.**
  Four independent Level 2 attempts and the Phase 5 cross-checks all
  converge on the same ~30-38% collision floor regardless of how much more
  training is thrown at it (`PHASE4_CURRICULUM_RESULTS.md`).
- **No braking reflex.** The agent commits to a trajectory rather than
  hedging; collisions are typically at or above cruising speed
  (`PHASE7_FAILURE_ANALYSIS_RESULTS.md`).
- **2D only.** No yaw/roll coupling, no real aerodynamic effects beyond
  simple linear drag.
- **The baseline has an information advantage** (exact state vs. 8-ray
  lidar) — Phase 6's comparison is honest about this and should not be
  read as "classical control is better than RL" in general.

## How to run

```bash
pip install -r requirements.txt   # torch is CPU-only by design, see CLAUDE.md
pytest                            # 111 tests across physics, PPO, curriculum, baseline, and diagnostics
```

Train from scratch (each writes checkpoints to `model_checkpoints/` and
plots/GIFs to `results/`):

```bash
python -m scripts.train_cartpole              # Phase 2: PPO validation on CartPole
python -m scripts.train_drone_hover_reach      # Phase 3: hover + reach, no obstacles
python -m scripts.train_drone_curriculum       # Phase 4: full curriculum (Levels 1-4)
python -m scripts.train_level15                # Phase 4: Level 1.5 step specifically
```

Evaluate / reproduce the reported results (all read-only, load an existing
checkpoint, no training):

```bash
python -m scripts.eval_outcome_breakdown model_checkpoints/final_model.pt
python -m scripts.phase5_generalization_eval
python -m scripts.phase6_baseline_comparison
python -m scripts.phase7_failure_analysis
python -m scripts.make_progression_demo        # regenerates the GIF at the top of this file
```

## Project structure

```
env/quadrotor_env.py          physics, reward, lidar -- the environment, from scratch
ppo/                           PPO agent, actor/critic networks, GAE rollout buffer -- from scratch
baseline/                      hand-coded potential-field + PD attitude controller (Phase 6)
scripts/                       training entry points, evaluation, and diagnostics for every phase
training/utils.py              shared eval-loop / plotting / demo-recording helpers
viz/                           GIF renderers (single-episode and the multi-stage progression demo)
tests/                         111 tests: physics, PPO, curriculum config, baseline, diagnostics
model_checkpoints/final_model.pt   the primary trained agent (Level 1.5 checkpoint)
results/                       plots, learning curves, and demo/failure GIFs (regeneratable, not version-controlled)
PHASE4_CURRICULUM_RESULTS.md   curriculum results + the Level 4 skip decision
PHASE5_GENERALIZATION_RESULTS.md
PHASE6_BASELINE_COMPARISON_RESULTS.md
PHASE7_FAILURE_ANALYSIS_RESULTS.md
PROJECT_SPEC.md                the original phased plan this project followed
```
