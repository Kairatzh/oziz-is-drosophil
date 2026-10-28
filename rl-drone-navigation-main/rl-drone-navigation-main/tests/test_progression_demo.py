"""
Tests for the Phase 8 progression-demo helpers. The interesting things to
check here: the legacy (pre-lidar) observation truncation actually produces
an 8-dim vector the old checkpoints accept, and the multi-segment GIF
renderer produces the frame count its own hold/stride math promises
(a real file, not just "didn't crash").
"""

import os

from env.quadrotor_env import QuadrotorEnv
from ppo.ppo_agent import PPOAgent
from scripts.make_progression_demo import LEGACY_OBS_DIM, make_agent, run_episode
from viz.render_progression import render_progression_gif

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")


def test_legacy_checkpoint_loads_with_truncated_obs_dim():
    agent = make_agent(LEGACY_OBS_DIM, os.path.join(CHECKPOINT_DIR, "phase3_hover_reach_final.pt"))
    env = QuadrotorEnv(seed=0)
    ep = run_episode(agent, env, LEGACY_OBS_DIM, stochastic=False, min_steps=1, max_attempts=1)
    assert ep["actions"].shape[1] == 2
    assert len(ep["states"]) == len(ep["actions"])


def test_run_episode_retries_until_min_steps_or_gives_up():
    agent = PPOAgent(obs_dim=LEGACY_OBS_DIM, act_dim=2, continuous=True)
    env = QuadrotorEnv(seed=1)
    ep = run_episode(agent, env, LEGACY_OBS_DIM, stochastic=True, min_steps=5, max_attempts=5)
    assert ep is not None
    assert len(ep["states"]) == len(ep["actions"])


def test_render_progression_gif_produces_expected_frame_count(tmp_path):
    env = QuadrotorEnv(seed=0)
    agent = PPOAgent(obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True)
    ep = run_episode(agent, env, QuadrotorEnv.OBS_DIM, stochastic=True, min_steps=10, max_attempts=10)
    segments = [
        {"label": "test stage", "env": env, "states": ep["states"], "actions": ep["actions"], "outcome": ep["outcome"]},
    ]
    out_path = str(tmp_path / "test_progression.gif")
    frame_stride, hold_frames = 3, 4
    n_frames = render_progression_gif(segments, out_path, frame_stride=frame_stride, hold_frames=hold_frames)

    expected_strided = len(range(0, len(ep["states"]), frame_stride))
    last_idx = list(range(0, len(ep["states"]), frame_stride))[-1]
    if last_idx != len(ep["states"]) - 1:
        expected_strided += 1
    assert n_frames == expected_strided + hold_frames
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
