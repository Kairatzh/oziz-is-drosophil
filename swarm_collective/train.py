"""
train.py — Main training entrypoint for the Swarm Collective Intelligence system.

Can be run from command line:
    python -m swarm_collective.train --episodes 1000 --device cuda
or imported in Google Colab:
    from swarm_collective.train import run_training
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import yaml

# Allow imports from project
_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR    = _CURRENT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from swarm_collective.env import SwarmEnv, ROLE_NAMES
from swarm_collective.agents import (
    BackboneExtractor, build_backbone,
    SwarmAgent, SwarmSystem
)
from swarm_collective.training import (
    MultiAgentReplayBuffer, MultiAgentTransition,
    QMixTrainer, CurriculumManager
)
from swarm_collective.utils import (
    SwarmMetricsTracker, plot_training_curves, plot_communication_graph
)


def load_config(config_path: str = "swarm_collective/configs/swarm_config.yaml") -> dict:
    if not os.path.exists(config_path):
        # Fallback to local
        config_path = str(_CURRENT_DIR / "configs" / "swarm_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(
    config: Optional[dict] = None,
    total_episodes: Optional[int] = None,
    pretrained_path: Optional[str] = None,
    device_str: Optional[str] = None,
) -> dict:
    """
    Main training execution function.
    """
    if config is None:
        config = load_config()

    # Device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[SwarmTrain] Using compute device: {device}")

    # Checkpoint path
    if pretrained_path is None:
        pretrained_path = config.get("backbone", {}).get("pretrained_path", None)
    if pretrained_path and not os.path.exists(pretrained_path):
        print(f"[SwarmTrain] Pretrained backbone checkpoint {pretrained_path} not found locally.")
        print(f"             Proceeding with freshly initialized backbone for training.")
        pretrained_path = None

    # Backbone setup
    backbone = build_backbone(
        ckpt_path=pretrained_path,
        freeze=config["backbone"].get("freeze_encoder", True),
        finetune_lr_mult=config["backbone"].get("fine_tune_lr_multiplier", 0.1),
        grid_size=config["env"].get("grid_size", 32)
    )

    # Swarm Agent & System
    agent = SwarmAgent(
        backbone=backbone,
        role_embedding_dim=config["agent"].get("role_embedding_dim", 64),
        hidden_dim=config["agent"].get("hidden_dim", 512),
        dim_msg=config["communication"].get("dim_message", 16),
        dim_key=config["communication"].get("dim_key", 8),
        comm_cost=config["communication"].get("comm_cost", 0.01),
    )
    system = SwarmSystem(agent=agent).to(device)

    # Curriculum manager
    stages = config.get("curriculum", {}).get("stages", [])
    curriculum = CurriculumManager(
        stages_config=stages,
        window_size=config.get("curriculum", {}).get("window_size", 100)
    )
    current_stage = curriculum.current_stage

    # Environment
    env = SwarmEnv(
        grid_size=config["env"].get("grid_size", 32),
        n_scouts=current_stage.get("n_scouts", 2),
        n_carriers=current_stage.get("n_carriers", 1),
        n_analyzers=current_stage.get("n_analyzers", 1),
        task=current_stage.get("task", "search"),
        fault_prob=current_stage.get("fault_prob", 0.0),
        comm_range=config["env"].get("comm_range", 5),
        vision_radius=config["env"].get("vision_radius", 3),
        max_steps=config["env"].get("max_steps", 300),
    )

    max_agents = config["env"].get("max_agents", 6)
    buffer = MultiAgentReplayBuffer(
        capacity=config["training"].get("buffer_size", 100000),
        max_agents=max_agents
    )

    trainer = QMixTrainer(
        swarm_system=system,
        max_agents=max_agents,
        obs_dim_per_agent=6144,
        mixing_embed_dim=config["qmix"].get("mixing_embed_dim", 32),
        hypernet_embed_dim=config["qmix"].get("hypernet_embed_dim", 64),
        lr=config["training"].get("lr", 3e-4),
        gamma=config["training"].get("gamma", 0.97),
        target_update_freq=config["training"].get("target_update_freq", 500),
        grad_clip=config["training"].get("grad_clip", 1.0),
        device=device,
    )

    metrics = SwarmMetricsTracker()

    # Training parameters
    if total_episodes is None:
        total_episodes = config["training"].get("total_episodes", 1000)
    batch_size = config["training"].get("batch_size", 32)
    epsilon = config["training"].get("epsilon_start", 1.0)
    eps_end = config["training"].get("epsilon_end", 0.05)
    eps_decay = config["training"].get("epsilon_decay", 0.9995)
    eval_interval = config["training"].get("eval_interval", 100)

    history = {
        "loss": [],
        "rewards": [],
        "success_rate": [],
        "msg_norm": [],
    }

    print(f"\n🚀 Starting Swarm Collective Training for {total_episodes} episodes...")
    print(f"   Initial Stage: {curriculum.current_stage_name}")

    for ep in range(1, total_episodes + 1):
        obs_dict, info = env.reset()
        # Convert dict or list of obs into list of dicts
        obs_list = [obs_dict[i] for i in range(len(env.role_ids))] if isinstance(obs_dict, dict) else obs_dict

        ep_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        last_attn = None

        while not (terminated or truncated):
            positions = [list(pos) for pos in env.agent_pos]
            alive_mask = list(env.agent_alive)

            actions, comm_pen, attn = trainer.select_actions(
                obs_list=obs_list,
                role_ids=env.role_ids,
                positions=positions,
                alive_mask=alive_mask,
                comm_range=env.comm_range,
                epsilon=epsilon
            )
            last_attn = attn

            # Inject TarMAC comm penalty into environment reward
            env.set_comm_penalty(comm_pen)

            next_obs_dict, rewards, terminated, truncated, next_info = env.step(actions)
            next_obs_list = [next_obs_dict[i] for i in range(len(env.role_ids))] if isinstance(next_obs_dict, dict) else next_obs_dict

            # Store in buffer
            M = len(env.role_ids)
            trans = MultiAgentTransition(
                obs_global=np.stack([o["global"] for o in obs_list]),
                obs_local=np.stack([o["local"] for o in obs_list]),
                ctx=np.stack([o["ctx"] for o in obs_list]),
                role_ids=np.array(env.role_ids, dtype=np.int64),
                positions=np.array(positions, dtype=np.int64),
                alive_mask=np.array(alive_mask, dtype=bool),
                actions=np.array(actions, dtype=np.int64),
                rewards=np.array(rewards, dtype=np.float32),
                next_obs_global=np.stack([o["global"] for o in next_obs_list]),
                next_obs_local=np.stack([o["local"] for o in next_obs_list]),
                next_ctx=np.stack([o["ctx"] for o in next_obs_list]),
                next_positions=np.array([list(p) for p in env.agent_pos], dtype=np.int64),
                next_alive_mask=np.array(list(env.agent_alive), dtype=bool),
                terminated=terminated,
                truncated=truncated
            )
            buffer.push(trans)

            obs_list = next_obs_list
            ep_reward += sum(rewards)
            step_count += 1

            # Train step
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size, device=device)
                loss_info = trainer.update(batch)
                history["loss"].append(loss_info["loss"])

        # Decay exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Record metrics
        is_success = bool(info.get("target_found", False) or next_info.get("task_phase") == "done")
        curriculum.record_episode(is_success)
        metrics.record_episode_end(
            success=is_success,
            length=step_count,
            total_reward=ep_reward,
            had_fault=bool(env.fault_prob > 0 and not all(env.agent_alive))
        )
        history["rewards"].append(ep_reward)
        history["success_rate"].append(curriculum.rolling_success_rate)

        # Check curriculum advancement
        new_stage = curriculum.advance_stage()
        if new_stage is not None:
            # Recreate env with new stage requirements
            env = SwarmEnv(
                grid_size=config["env"].get("grid_size", 32),
                n_scouts=new_stage.get("n_scouts", 2),
                n_carriers=new_stage.get("n_carriers", 1),
                n_analyzers=new_stage.get("n_analyzers", 1),
                task=new_stage.get("task", "search"),
                fault_prob=new_stage.get("fault_prob", 0.0),
                comm_range=config["env"].get("comm_range", 5),
                vision_radius=config["env"].get("vision_radius", 3),
                max_steps=config["env"].get("max_steps", 300),
            )

        # Progress log
        if ep % 20 == 0 or ep == total_episodes:
            print(f"Ep {ep:4d}/{total_episodes} | Stage: {curriculum.current_stage_name:<16} | "
                  f"WinRate: {curriculum.rolling_success_rate*100:5.1f}% | "
                  f"Reward: {ep_reward:7.1f} | Eps: {epsilon:.3f}")

        # Periodic visual evaluation
        if ep % eval_interval == 0 or ep == total_episodes:
            plot_training_curves(history, save_path="training_curves.png")
            if last_attn is not None:
                role_names = [ROLE_NAMES[r] for r in env.role_ids]
                plot_communication_graph(
                    positions=[list(p) for p in env.agent_pos],
                    role_names=role_names,
                    alive_mask=list(env.agent_alive),
                    attn_matrix=last_attn,
                    target_pos=list(env.target_pos),
                    save_path="last_comm_graph.png"
                )

    print("\n✅ Training complete! Saving final model checkpoint...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "swarm_system": system.state_dict(),
        "mixer": trainer.mixer.state_dict(),
        "config": config,
    }, "checkpoints/swarm_collective_final.pt")
    print("Checkpoint saved to: checkpoints/swarm_collective_final.pt")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="Total episodes")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--config", type=str, default="swarm_collective/configs/swarm_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(config=cfg, total_episodes=args.episodes, device_str=args.device)
