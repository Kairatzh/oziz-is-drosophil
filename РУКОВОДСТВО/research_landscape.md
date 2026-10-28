# Research Landscape (Stage 1, snapshot 2026-10-28)

## Method and limits (read first)
- Evidence = project documentation, release pages, READMEs and papers retrieved via web search/fetch on 2026-10-28.
- **No code was built or run.** The sandbox has no network egress and no simulator. Nothing here is a benchmark result.
- Status tags: **[V]** seen in a retrieved page this session; **[B]** background knowledge, NOT re-verified; **[?]** not investigated yet.
- Stage 2 must convert every [B]/[?] that influences a decision into [V] by running something.

## Landscape summary

| Area | Project | What was verified [V] | Main implication |
|---|---|---|---|
| Sim (photoreal) | Project AirSim (MIT) | Successor to Microsoft AirSim; headless Runtime without Unreal (Fast Physics, JSBSim, Simulink models); UE5 for rendered worlds; Python/C++ clients, ROS 2 C++ bridge (Humble, Ubuntu 22.04); PX4 SITL; v1.1.0 is flagged pre-release | Strongest single candidate for fidelity ladder; maturity risk |
| Sim (photoreal, legacy) | AirSim | Maintainers' notice says effort moved to Project AirSim | Do not build on legacy AirSim |
| Sim (GPU RL) | Isaac Sim 5.0 / Isaac Lab | Isaac Sim 5.0 open-sourced, ROS 2 Jazzy, docker image; Isaac Lab docs list releases 3.0, 2.3, 2.2, 2.1; recommended 32 GB RAM / 16 GB VRAM | Heavy hardware floor; best for batched RL if GPU exists |
| Aerial RL | Aerial Gym | Isaac Gym based, thousands of parallel multirotors, SE(3) geometric controllers, GPU ray-cast depth/segmentation, sim2real shown in paper | Strong for learned nav policies; Isaac Gym dependency needs checking [?] |
| Aerial RL | OmniDrones | Built on Isaac Sim 4.1.0; author states the project is hard to maintain further | Reference only; do not depend on it |
| Aerial RL (CPU) | gym-pybullet-drones | PyBullet, Gymnasium + SB3, RLlib multi-agent, physics variants (ground effect, drag, downwash), Crazyflie/Betaflight SITL | Good CPU smoke-test and low-fidelity tier |
| Flight stack | PX4 | v1.16 moved to Gazebo Harmonic LTS, faster-than-real-time sim, experimental ROS 2 interface library; v1.17 docs exist | Use as independent flight controller for the high-fidelity tier |
| Exploration | FUEL (HKUST) | ROS 1 (tested Melodic/Noetic, Ubuntu 18.04/20.04); built on Fast-Planner which is GPLv3; multi-UAV code released Feb 2023; point-cloud (.pcd) sim worlds | Valuable baseline, but ROS 1 + GPL: isolate in a container, do not link |
| MARL | BenchMARL | TorchRL backend, VMAS/PettingZoo support, Hydra configs, stratified-bootstrap IQM reporting | Use for training/benchmark orchestration of learned policies |

## Not yet investigated [?]
Flightmare, FlightGoggles, MuJoCo aerial projects, MAVSDK, ROS 2 specifics, ORB-SLAM3, OpenVINS, VINS-Fusion, RTAB-Map, Nav2, OctoMap, Voxblox, RLlib multi-agent API, PettingZoo, Sample Factory, MAPPO reference code, ant-colony/stigmergy repos, object detectors, depth-estimation models, pretrained drone-racing/obstacle-avoidance checkpoints, Gazebo Sim details.

## Key open questions
1. Which pretrained aerial policy will serve as the Stage 2 "load and evaluate in original environment" proof, and are its checkpoints actually downloadable?
2. What GPU/OS does the team have? This decides whether Isaac Lab / UE5 are viable.
3. Does Project AirSim's headless Runtime scale to the multi-agent counts we need, with which sensors available without Unreal?
