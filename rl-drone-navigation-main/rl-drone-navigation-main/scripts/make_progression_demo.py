"""
Phase 8: the visual centerpiece -- a single GIF showing the agent's arc
from random initialization through to the final trained Level 1.5 agent
(model_checkpoints/final_model.pt), i.e. PROJECT_SPEC.md Phase 8's
"compelling demo GIF showing progression from random crashing to smooth
navigation."

Four stages, each an actual episode run with the real checkpoint from that
point in training (not staged/faked):
  1. Random initialization -- a freshly constructed, never-trained agent,
     run STOCHASTICALLY (agent.act(), with exploration noise) since that's
     what "random behavior" actually looks like early in training; every
     other stage uses the deterministic policy mean, like every other eval
     in this project.
  2. Phase 3, update 10 -- partway through learning to hover and reach a
     goal with no obstacles.
  3. Phase 3 final -- hover + reach solved.
  4. Level 1.5 final (= model_checkpoints/final_model.pt) -- adds a single
     obstacle; the final, primary trained agent from Phase 4 onward.

Stages 1-3 use an 8-dim observation (no lidar): the lidar rays
(env.quadrotor_env.QuadrotorEnv.OBS_DIM's extra 8 entries) were added
AFTER Phase 3, so those checkpoints' networks are literally sized for the
pre-lidar observation. Since stages 1-3 all run in the obstacle-free Phase
3 environment anyway (where lidar would read "all clear" and add no
information), this is simply how those checkpoints were actually trained
and evaluated, not an approximation.

Read-only. Does not train.
"""

import os

import numpy as np
import torch

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from scripts.train_drone_curriculum import make_level15_obstacle_fn
from viz.render_progression import render_progression_gif

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEMO_SEED = 3
LEGACY_OBS_DIM = 8  # pre-lidar observation size Phase 3 checkpoints were trained with


def run_episode(agent, env, obs_dim, stochastic=False, min_steps=15, max_attempts=15):
    """Runs episodes until one lasts at least min_steps (so a stage that
    crashes almost instantly still produces a watchable clip), returning
    the longest one found if none clears that bar. obs_dim truncates the
    observation fed to the agent, for the legacy pre-lidar checkpoints."""
    best = None
    for _ in range(max_attempts):
        obs = env.reset()[:obs_dim]
        states = [env.state.copy()]
        actions = [tuple(agent.act(obs)[0] if stochastic else agent.act_deterministic(obs))]
        done, info = False, {"outcome": "running"}
        while not done:
            action = agent.act(obs)[0] if stochastic else agent.act_deterministic(obs)
            obs, _, done, info = env.step(action)
            obs = obs[:obs_dim]
            states.append(env.state.copy())
            actions.append(tuple(action))
        result = {
            "states": np.array(states), "actions": np.array(actions),
            "outcome": info["outcome"], "env": env,
        }
        if best is None or len(states) > len(best["states"]):
            best = result
        if len(states) >= min_steps:
            return result
    return best


def make_agent(obs_dim, checkpoint=None, entropy_coef=0.003):
    agent = PPOAgent(
        obs_dim=obs_dim, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=entropy_coef),
    )
    if checkpoint is not None:
        agent.load(checkpoint)
    return agent


def main():
    torch.manual_seed(DEMO_SEED)
    np.random.seed(DEMO_SEED)

    no_obstacle_env = QuadrotorEnv(seed=DEMO_SEED)
    level15_env = QuadrotorEnv(obstacle_fn=make_level15_obstacle_fn(), seed=DEMO_SEED)

    stages = [
        ("Stage 1: random initialization (before any training)",
         make_agent(LEGACY_OBS_DIM, checkpoint=None), no_obstacle_env, LEGACY_OBS_DIM, True),
        ("Stage 2: Phase 3, update 10 (learning to hover)",
         make_agent(LEGACY_OBS_DIM, os.path.join(CHECKPOINT_DIR, "phase3_hover_reach_update0010.pt")),
         no_obstacle_env, LEGACY_OBS_DIM, False),
        ("Stage 3: Phase 3 final (hover + reach solved)",
         make_agent(LEGACY_OBS_DIM, os.path.join(CHECKPOINT_DIR, "phase3_hover_reach_final.pt")),
         no_obstacle_env, LEGACY_OBS_DIM, False),
        ("Stage 4: Level 1.5 final -- primary trained agent (final_model.pt)",
         make_agent(QuadrotorEnv.OBS_DIM, os.path.join(CHECKPOINT_DIR, "final_model.pt")),
         level15_env, QuadrotorEnv.OBS_DIM, False),
    ]

    segments = []
    for label, agent, env, obs_dim, stochastic in stages:
        ep = run_episode(agent, env, obs_dim, stochastic=stochastic)
        print(f"{label}: outcome={ep['outcome']}, {len(ep['states'])} steps")
        segments.append({
            "label": label, "env": ep["env"], "states": ep["states"],
            "actions": ep["actions"], "outcome": ep["outcome"],
        })

    out_path = os.path.join(RESULTS_DIR, "phase8_progression_demo.gif")
    n_frames = render_progression_gif(segments, out_path)
    print(f"saved {out_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
