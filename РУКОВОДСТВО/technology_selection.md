# Technology Selection (Stage 1 recommendation, pending Stage 2 gate)

## Recommended minimal set
| Layer | Choice | Alternatives considered | Why / trade-off |
|---|---|---|---|
| Primary simulator | Project AirSim (headless Runtime first, UE5 later) | Isaac Lab; Gazebo Harmonic+PX4; gym-pybullet-drones | MIT, runs without Unreal for CPU smoke tests, Python step() client, PX4 SITL, UE5 path to rich worlds. Risk: pre-release maturity, unmeasured scaling |
| Flight controller (L3) | PX4 (v1.16+) | Betaflight SITL; custom | Independent, widely used, ROS 2 and Gazebo Harmonic support |
| CPU smoke / L1 baseline | gym-pybullet-drones | Project AirSim Fast Physics only | Cheap CI, existing multi-agent examples, pretrained-policy loading test |
| GPU RL (optional) | Isaac Lab / Aerial Gym | OmniDrones (unmaintained) | Only if 16 GB VRAM class GPU is available; version churn risk |
| Middleware | ROS 2 (Humble first, matching Project AirSim bridge) | ZeroMQ only | Needed for L3 and for later SLAM adapters |
| MARL | BenchMARL on TorchRL | RLlib, Sample Factory [? not audited] | Standardised reporting with bootstrap CIs |
| Exploration baseline | Own frontier planner; FUEL as external reference | FUEL integrated | Avoids ROS 1/GPLv3 coupling |
| Mapping | Own sparse voxel log-odds map; OctoMap/Voxblox adapters later | Voxblox first [? not audited] | Smallest testable core |

## Stage 2 spike (acceptance tests, each must produce logged numbers)
1. Install Project AirSim Runtime from the Python client; run a hover for 1, 8, 32 agents headless; record steps/s, memory, latency.
2. Confirm which sensors work without Unreal; confirm seeded determinism.
3. Load one pretrained aerial policy in its original environment (candidate: gym-pybullet-drones or Aerial Gym; checkpoint availability must be confirmed first).
4. PX4 SITL with two vehicles under Project AirSim or Gazebo Harmonic.
5. Frame-transform unit tests (ENU/FLU vs NED/FRD).

## Decision rule
If tests 1-2 fail on target hardware, switch primary to PX4+Gazebo Harmonic and keep the core contracts unchanged.

## Hardware needed to finalise
GPU model/VRAM, OS, CPU core count. Unreal Engine 5 and Isaac Sim both assume a capable NVIDIA GPU.
