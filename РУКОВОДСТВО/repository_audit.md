# Repository Audit (Stage 1 partial, 2026-10-28)

Fields not confirmed this session are marked NOT VERIFIED. "Inspected" = documentation/README/release pages only; no source code was read and nothing was executed.

## Project AirSim (github.com/iamaisim/ProjectAirSim)
- Maintenance: v1.1.0 published as *pre-release* (adds wheeled vehicles, UE 5.8 support, Apple Silicon source builds); prebuilt environment/plugin packages for that release were "pending verification". PyPI client seen at 1.0.2.
- License: MIT (project page and PyPI).
- Fidelity: C++ Fast Physics, JSBSim, Simulink physics; per-robot physics type; PX4 SITL.
- OS/HW: Ubuntu 22.04 primary; Windows 11; UE 5.2 or 5.7 per wiki; ROS 2 Humble bridge. Headless/off-screen rendering supported.
- Sensors: rendered sensors need Unreal; Runtime offers non-rendered sensors (exact list NOT VERIFIED).
- Multi-agent: multiple robots in a scene supported in principle; scaling NOT VERIFIED.
- Checkpoints: none (simulator).
- Integration: Python `step()` client, ROS 2 bridge. Cost: medium (UE5 build is the burden).
- Limits/risks: young release line; small community; assets/environments must be sourced and license-checked.

## Isaac Sim 5.0 / Isaac Lab (3.0 / 2.x)
- Maintenance: active; Isaac Sim 5.0 open source. License terms per component NOT VERIFIED.
- HW: NVIDIA RTX GPU required; docs recommend 32 GB RAM and 16 GB VRAM.
- Interfaces: ROS 2 Jazzy, ZeroMQ bridge, Docker image.
- Aerial support specifics in Lab 3.0: NOT VERIFIED.
- Cost: high (hardware + version churn).

## Aerial Gym
- Isaac Gym based; parallel multirotor sim with geometric controllers; GPU ray-cast depth/segmentation; RL navigation examples; sim2real reported in paper.
- License NOT VERIFIED; current Isaac Gym dependency status NOT VERIFIED (original README pins Isaac Gym Preview 4).
- Pretrained checkpoints: NOT VERIFIED.

## OmniDrones
- Isaac Sim 4.1.0, MIT, Linux only. Author notice: hard to maintain/update further. Treat as read-only reference.

## gym-pybullet-drones
- PyBullet; Gymnasium + SB3 2.0; RLlib multi-agent scripts; physics options incl. ground effect, drag, downwash; Crazyflie/Betaflight SITL.
- Rendering with vision is a known cost center (numbers NOT VERIFIED). License NOT VERIFIED (believed permissive).
- Pretrained checkpoints: NOT VERIFIED (repo ships training/replay scripts).

## PX4 Autopilot
- v1.16: Gazebo Harmonic LTS, faster-than-real-time simulation, experimental ROS 2 interface library. v1.17 docs exist. Licenses are BSD-3 for PX4 core [B].
- Role: independent flight controller in the high-fidelity tier.

## FUEL / Fast-Planner / RACER
- ROS 1 (Melodic/Noetic), Fast-Planner GPLv3; multi-UAV exploration released Feb 2023; MARSIM provides LiDAR-based sim worlds.
- Cost of integration: high (ROS 1 EOL stack, GPL). Use as external reference baseline only.

## BenchMARL
- TorchRL backend; VMAS and PettingZoo supported; Hydra; reproducible benchmark/experiment abstractions; IQM with bootstrap CIs. License NOT VERIFIED (believed permissive).

## Audit TODO before Stage 3
Run `pip download`/clone-and-import checks, read licenses from LICENSE files, confirm checkpoints, and record results here with commit hashes.
