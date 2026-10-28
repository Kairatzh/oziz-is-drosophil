# Autonomous Drone Navigation via Deep Reinforcement Learning

## Project Goal
Build a complete deep reinforcement learning system — custom physics
environment, PPO implementation from scratch, curriculum learning, and
rigorous evaluation — that trains a 2D quadrotor drone to navigate through
obstacle courses entirely from self-play, with no hand-coded control rules.
Compare the trained agent against a hand-coded PID/baseline controller to
quantify what reinforcement learning buys over classical control. This is a
college application portfolio project demonstrating ML/AI capabilities
distinct from my existing simulation work.

## Context and Motivation
My prior projects (rocket-sim, kalman-sensor-fusion, gnc-closed-loop,
fea-structural-failure) all use classical methods: analytical physics,
Kalman filtering, LQR optimal control, finite element solvers. None involve
learning from data. This project asks a fundamentally different question:
instead of deriving the optimal control law mathematically (as I did with
LQR in gnc-closed-loop), can a neural network LEARN a control policy
through trial and error, with no knowledge of the physics equations? When
does learned control beat classical control, and when does it fail?

I build drones from scratch as a hobby. This project bridges my physical
drone-building experience with computational learning — I've always had to
tune controllers manually; this project asks whether a drone can learn to
fly itself.

## Important Working Style Requirements
- Build INCREMENTALLY, phases in order. Never skip ahead.
- After each phase, STOP and explain in plain language: what was
  implemented, why it matters for RL (not just "it works"), what to check
  in the output. This is my first ML project — explain RL concepts
  clearly, don't assume prior knowledge.
- Write tests for each phase before moving on.
- All plots MUST use plt.savefig() + plt.close('all'). NEVER call
  plt.show() — this causes indefinite blocking on Windows.
- Do NOT add "Co-authored-by" trailers to git commits.
- Commit at end of each validated phase with clear descriptive message.
- If training seems stuck or a phase is taking very long with no clear
  path forward, STOP and summarize: what's been tried, current best
  theory, one minimal next experiment. Do not iterate indefinitely.
- If something is ambiguous, ask rather than assuming.
- Default to CPU training. Do not attempt AMD GPU acceleration without
  explicitly discussing ROCm setup first.

## Phase 1: Custom 2D Quadrotor Physics Environment
- Implement a 2D quadrotor dynamics model from scratch (NOT using OpenAI
  Gym or any pre-built RL environment — write your own):
  - State: [x, z, vx, vz, theta (pitch angle), omega (angular velocity)]
  - Actions: [left_thrust, right_thrust] (two rotors, differential thrust
    produces both lift and rotation — this is what makes quadrotor control
    interesting)
  - Physics: Newton's laws for translational motion, torque for rotation,
    gravity, simple drag. No atmosphere model needed — keep it clean.
  - Episode structure: random start position, random goal position, episode
    ends on collision with obstacle, going out of bounds, or reaching goal
  - Reward function: discuss and design explicitly before implementing.
    A good reward function is the single most important design decision in
    RL — get this wrong and the agent learns the wrong thing. Explain the
    components and tradeoffs clearly.
- Implement a basic obstacle representation: static circular or rectangular
  obstacles placed in the 2D environment.
- Validate the physics with a simple sanity check: apply constant upward
  thrust, confirm the drone hovers at the right altitude given gravity.
  Apply differential thrust, confirm it rotates correctly.
- Output: an animated visualization (saved as .gif or series of frames,
  NOT plt.show()) showing a drone responding to manual test control inputs
  in an obstacle environment.
- Explain: what makes quadrotor control hard (underactuated — only 2
  control inputs for 3 DOF), what the reward function components mean
  physically, and why we're building the environment from scratch rather
  than using OpenAI Gym.

## Phase 2: PPO Implementation from Scratch
- Implement Proximal Policy Optimization (PPO) directly in PyTorch from
  the start (numpy PPO is too slow for meaningful training). Implement it
  yourself — don't use stable-baselines or any RL library. Clearly comment
  every component.
- Key components to implement and explain:
  - Actor network: maps state -> action probability distribution
  - Critic network: maps state -> value estimate (how good is this state?)
  - The PPO clipping objective: why it exists (prevents too-large policy
    updates that destabilize training), what the clip ratio means
  - GAE (Generalized Advantage Estimation): what advantage means in RL,
    why GAE reduces variance
  - The training loop: collect rollouts -> compute advantages -> update
    policy with clipped objective -> repeat
- Validate on CartPole before connecting to the drone environment.
- Output: a learning curve showing CartPole reward increasing over
  training episodes, confirming PPO is working.
- Explain: what "policy" means in RL, what "value function" means, why PPO
  is the right algorithm choice over simpler alternatives (REINFORCE, DQN),
  and what the clipping trick actually does mathematically.

## Phase 3: Basic Drone Training — Hover and Reach
- Connect the PPO implementation to the custom drone environment.
- Start with the simplest possible task: no obstacles, just hover and
  reach a target position from a random start.
- Train until the agent reliably reaches the target.
- Implement proper training infrastructure:
  - Episode logging (reward per episode, success rate, moving average)
  - Periodic evaluation runs (separate from training, no exploration noise)
  - Model checkpointing (save weights every N episodes)
- Output: learning curve showing reward/success rate improving over
  training, plus a visualization of the trained agent navigating to a
  target (saved as animated .gif).
- Explain: what "exploration vs. exploitation" means in RL, why random
  exploration is necessary early in training, and what the learning curve
  shape tells you about whether training is going well.

## Phase 4: Curriculum Learning — Progressive Obstacle Difficulty
- Implement curriculum learning with automatic progression:
  - Level 1: no obstacles
  - Level 2: 1-2 large obstacles, wide gaps
  - Level 3: 3-5 obstacles, moderate gaps
  - Level 4: dense obstacle field, narrow gaps
  - Level 5: moving obstacles (if training succeeds on static first)
- Automatic progression criterion: advance when agent achieves X% success
  rate over last N evaluation episodes.
- Output: training curves for each curriculum level, plus visualizations
  of the agent navigating at each difficulty level.
- Explain: sparse rewards, why curriculum learning solves this, what
  agent behavior looks like at each stage.

## Phase 5: Generalization Testing
- Test on obstacle courses never seen during training.
- Quantify success rate on training vs. held-out test environments.
- If generalization is poor, discuss domain randomization as a fix.
- Output: comparison table of training vs. test performance, plus
  visualizations of agent succeeding and failing on novel environments.

## Phase 6: Comparison Against a Hand-Coded Baseline
- Implement a hand-coded PID or potential-field controller.
- Compare RL agent vs. baseline on: success rate, time to goal,
  trajectory smoothness, robustness to novel configs.
- Be honest about results — report where baseline wins.
- Output: comparison table and side-by-side trajectory visualizations.

## Phase 7: Failure Analysis and Limitations
- Systematically characterize where the agent fails.
- Adversarial testing if time allows.
- Output: failure mode analysis with visualizations of representative
  failure cases.

## Phase 8: Polish and Documentation
- Clean up all visualizations and plots.
- Create compelling demo GIF showing progression from random crashing
  to smooth navigation.
- Write README.md as a full technical report covering: problem statement,
  methods, key equations (PPO clipping, GAE, value function loss),
  results, limitations, how to run.

## Deliverables
1. Custom 2D quadrotor physics environment from scratch
2. PPO implementation from scratch in PyTorch (no RL libraries)
3. Trained agent navigating obstacle courses via curriculum learning
4. Generalization testing + honest baseline comparison
5. Compelling animated visualization
6. Polished README
7. Deep understanding of all RL concepts for interviews
