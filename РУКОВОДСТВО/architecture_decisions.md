# Architecture Decision Records (all PROPOSED unless stated; gated by Stage 2 spikes)

## ADR-001 Reuse, don't reinvent (ACCEPTED)
No custom rigid-body physics, renderer, SLAM or flight controller. Alternatives: custom Python physics (rejected: validity burden, no credibility). Consequence: fidelity comes from adapters over upstream simulators.

## ADR-002 Simulator-agnostic core + backend adapters
Core package defines typed contracts: `SimBackend`, `Vehicle`, `Sensor`, `OracleView`, `PolicyAdapter`, `CommsChannel`, `MapEstimator`, `Coordinator`, `Benchmark`, `ExperimentRunner`. Backends live behind a process boundary (client library / ROS 2 / ZeroMQ), so conflicting dependencies (UE5, ROS, PyTorch) never share one Python process.
Alternative: one monolithic Isaac Lab project (rejected: hardware lock-in, hard to compare simulators).

## ADR-003 Fidelity ladder via backends (GATED)
- L1 fast: Project AirSim headless Runtime (Fast Physics) and/or gym-pybullet-drones on CPU.
- L2 standard: Project AirSim Runtime/UE5 rigid-body multirotor.
- L3 high: PX4 SITL as independent controller with Project AirSim+UE5, fallback PX4+Gazebo Harmonic.
- Optional L1-GPU: Isaac Lab/Aerial Gym for batched RL training, only if hardware allows.
Gate: Stage 2 spike must measure multi-agent throughput, sensor availability without Unreal, seed determinism, and PX4 multi-vehicle behaviour. If Project AirSim fails the gate, fall back to PX4+Gazebo Harmonic for L2/L3.

## ADR-004 Oracle/sensor separation enforced by types and tests
Policies receive only `SensorFrame` objects. Ground truth is exposed via a separate `OracleView` that benchmarks must explicitly request and that is logged as "privileged". A leakage test scans each benchmark's observation schema and fails if world-frame ground truth or true-map fields appear.

## ADR-005 Frames and units
Core convention: world ENU, body FLU, SI units, quaternions (w,x,y,z), time in simulator seconds. Adapters convert from PX4 NED/FRD and each simulator's native frame. Automated round-trip transform tests are mandatory.

## ADR-006 Coordination layer is pure Python and simulator-independent
Event-driven comms simulator (range, bandwidth, loss, delay, partitions, dropout, prioritisation) running in sim time; pheromone fields as explicit voxel/grid arrays with deposition, decay, diffusion; coordinators see only locally received messages in decentralised mode. This layer is unit-testable on CPU without any simulator.

## ADR-007 Mapping and exploration baselines
First-party minimal log-odds voxel map and frontier baseline (small, testable); OctoMap/Voxblox via adapters later; FUEL/RACER only as an optional external baseline in its own ROS 1 container (GPLv3, EOL middleware; never linked into the core).

## ADR-008 Learning stack
Expose a PettingZoo-style parallel API for training; use BenchMARL/TorchRL for MARL training and reporting; evaluation harness stays independent so pretrained and classical agents run under identical protocols.

## ADR-009 Experiment manifests
Every run writes JSON: git revision, config hash, checkpoint hashes, simulator and dependency versions, hardware, seeds, metrics, status. Weights stay out of Git.

## ADR-010 Simulation-only boundary (ACCEPTED)
No hardware connection in the first release. Bounded commands, crash/instability detection and safe reset are part of the core contract.
