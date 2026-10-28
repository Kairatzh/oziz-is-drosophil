# 🐝 OZIZ: Swarm Collective Intelligence & Drone Navigation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kairatzh/oziz-is-drosophil/blob/main/swarm_collective.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Multi-Agent Reinforcement Learning (MARL) framework for heterogeneous drone swarms with emergent, targeted communication (TarMAC), monotonic value factorization (QMIX), and frozen spatial perception backbones.

---

## 🚀 Quick Start in Google Colab

Click the badge above or open **`swarm_collective.ipynb`** in Google Colab:
1. Select **GPU** (Runtime → Change runtime type → T4 GPU).
2. Run cells sequentially to install dependencies, load the pretrained ConvNeXt perception backbone from Hugging Face, run smoke tests, and launch the 5-stage curriculum training.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      SwarmEnv (Multi-Role)                     │
│                                                                │
│  🔭 SCOUT × N         📦 CARRIER × M       🔬 ANALYZER × K     │
│  (Exploration)        (Transport)          (Hover/Inspect)     │
│        │                    │                     │            │
│        └────────────────────┴─────────────────────┘            │
│                     ▼ TarMAC Module ▼                          │
│            Attention-gated message routing                     │
│            L2 Bandwidth Penalty: r_comm = -λ ||m||_2           │
│                                                                │
│                     ▼ QMIX Mixer ▼                             │
│            Monotonic Value Factorization                       │
│            Centralized Training, Decentralized Execution       │
└────────────────────────────────────────────────────────────────┘
```

### Key Pillars:
1. **Targeted Communication (TarMAC)**: Each agent dynamically broadcasts $(m_i, s_i, q_i)$ to neighbours within communication range. Attention masks prevent cross-talk from distant or damaged drones.
2. **Bandwidth Penalty**: Prevents communication spam by penalizing message norm ($\|m_i\|_2$). Drones learn to remain silent unless broadcasting critical target or threat coordinates.
3. **Pretrained Perception**: Reuses frozen spatial feature extractors (`GlobalMapCNN` + `LocalCNN`) from `Multi-Agent-Systems-Project-main` to eliminate cold-start training overhead.
4. **5-Stage Curriculum**:
   - `Stage 1`: Scouts only (Target Search)
   - `Stage 2`: Scouts + Carriers (Search & Payload Transport)
   - `Stage 3`: Scouts + Analyzers (Search & Diagnostic Hovering)
   - `Stage 4`: Full Heterogeneous Swarm Pipeline
   - `Stage 5`: Swarm with Random Drone Faults & Packet Dropout

---

## 📁 Repository Structure

```
oziz/
├── swarm_collective/            # Main Collective Intelligence Package
│   ├── env/                     # Multi-role environment & task state machines
│   ├── agents/                  # TarMAC, Backbone loader, QMIX mixer & SwarmAgent
│   ├── training/                # QMIX trainer, Replay buffer, Curriculum manager
│   ├── configs/                 # YAML hyperparameters
│   ├── utils/                   # Telemetry metrics & graph visualizers
│   ├── train.py                 # Standalone / CLI training entrypoint
│   └── requirements.txt         # Package dependencies
│
├── swarm_collective.ipynb       # 📓 Google Colab Master Notebook
│
├── РУКОВОДСТВО/                 # Architecture Decision Records & System Audits
├── Multi-Agent-Systems-Project-main/  # Pretrained spatial navigation reference
├── rl-drone-navigation-main/          # Continuous quadrotor baseline reference
└── DGPPO_MRTransport-main/            # Collaborative payload transport reference
```

---

## 💻 Local Installation

```bash
git clone https://github.com/Kairatzh/oziz-is-drosophil.git
cd oziz-is-drosophil
pip install -r swarm_collective/requirements.txt
python -m swarm_collective.train --episodes 500
```
