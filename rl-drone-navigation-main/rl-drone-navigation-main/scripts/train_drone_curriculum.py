"""
Phase 4: curriculum learning with progressively harder obstacle fields.

Why curriculum learning at all? This is fundamentally about sparse
rewards. Picture dropping a randomly-initialized policy directly into
Level 4 (a dense obstacle field): almost every random action sequence
crashes into something within the first second, so the agent essentially
NEVER experiences a successful episode to learn from -- the +100 success
bonus is buried behind a wall of near-certain early failure, and the
progress-shaping reward (Phase 1) can't help thread a specific narrow gap
it has never once passed through. The gradient signal is there in theory,
but in practice the agent almost never encounters the states where it
matters.

Curriculum learning fixes this by having the agent master an easy version
of the task first (Phase 3: no obstacles at all), then gradually
introducing difficulty once each stage is reliably solved. The skills
transfer: "tilt to translate, level out to stop" (learned with zero
obstacles) is exactly what's needed to route around a single obstacle,
which is exactly what's needed to weave through several, and so on. Each
stage's reliable success means the harder stage's random exploration
starts from "usually gets to roughly the right place" instead of
"usually crashes at step 3" -- which is what makes the reward signal
findable at all in the harder settings.

THE AUTOMATIC PROGRESSION CRITERION (defined explicitly before training,
per instructions):
  - Every EVAL_EVERY=5 PPO updates, run EVAL_EPISODES=20 deterministic
    (no exploration noise) evaluation episodes in the CURRENT level's
    environment.
  - Advance to the next level as soon as eval success rate over those 20
    episodes is >= SUCCESS_THRESHOLD=0.9 (90%).
  - Safety valve: if a level does NOT reach 90% within
    MAX_UPDATES_PER_LEVEL=250 updates, STOP the entire curriculum run
    (do not advance) and report where it got stuck -- advancing an
    unreliable policy to a harder level would only make it fail worse,
    defeating the point of a curriculum.
  - The same agent (actor + critic weights, optimizer state) carries over
    between levels unchanged -- only the environment's obstacle list
    changes. This is what lets skill transfer actually happen; resetting
    the network between levels would make this no different from
    training 4 separate unrelated agents.

As required by CLAUDE.md: plt.show() is never called; CPU training only.
"""

import os
import time

import numpy as np
import torch

from env.quadrotor_env import CircleObstacle, QuadrotorEnv
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer
from training.utils import plot_learning_curve, record_demo_episode, run_evaluation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ROLLOUT_STEPS = 4096
MAX_UPDATES_PER_LEVEL = 250
EVAL_EVERY = 5
EVAL_EPISODES = 20
CHECKPOINT_EVERY = 10
SUCCESS_THRESHOLD = 0.9
MOVING_AVG_WINDOW = 20
SEED = 0

# Lower than PPOConfig's shared default (0.01). Level 2 diagnosis showed
# entropy saturating at the log_std clamp ceiling with success rate
# plateaued at 20-85% (no convergence) -- the entropy bonus was
# outweighing the policy-improvement signal on this harder, obstacle-
# avoidance task. Reduced 10x here rather than changing PPOConfig's
# default, since Phase 2/3 (CartPole, obstacle-free hover-reach) already
# work correctly at 0.01 and shouldn't be touched.
ENTROPY_COEF = 0.001

def _point_segment_distance(px, pz, ax, az, bx, bz):
    """Shortest distance from point (px,pz) to segment (ax,az)-(bx,bz)."""
    abx, abz = bx - ax, bz - az
    denom = abx * abx + abz * abz
    if denom < 1e-12:
        return float(np.hypot(px - ax, pz - az))
    t = max(0.0, min(1.0, ((px - ax) * abx + (pz - az) * abz) / denom))
    return float(np.hypot(px - (ax + t * abx), pz - (az + t * abz)))


def make_level15_obstacle_fn(radius=0.8, clear_margin=QuadrotorEnv.GOAL_LOS_MARGIN, max_tries=200):
    """Level 1.5 obstacle placement: ONE circular obstacle of the given radius
    (smaller than Level 2's 1.2 m), positioned randomly but with its SURFACE
    kept at least `clear_margin` from the straight start->goal segment. So the
    direct line of sight to the goal is always clear -- the agent never *needs*
    to route around, only to avoid blundering into an obstacle it can see. This
    is the stepping stone between "no obstacles" (Level 1) and "must route
    around" (Level 2): obstacles exist and hurt, but a straight path still
    works. Falls back to no obstacle in the rare case no clear spot is found,
    which keeps the path valid rather than risking a blocked line."""
    def obstacle_fn(rng, start, goal):
        sx, sz, gx, gz = float(start[0]), float(start[1]), float(goal[0]), float(goal[1])
        for _ in range(max_tries):
            x = float(rng.uniform(radius, QuadrotorEnv.WORLD_WIDTH - radius))
            z = float(rng.uniform(radius, QuadrotorEnv.WORLD_HEIGHT - radius))
            # surface clears the whole segment (which includes the start and
            # goal endpoints), so both LOS and the start/goal points stay clear.
            if _point_segment_distance(x, z, sx, sz, gx, gz) - radius >= clear_margin:
                return [CircleObstacle(x=x, z=z, radius=radius)]
        return []
    return obstacle_fn


CURRICULUM = [
    {
        "name": "level1_no_obstacles",
        "title": "Level 1: no obstacles",
        "obstacles": [],
    },
    {
        "name": "level1_5_single_clear_obstacle",
        "title": "Level 1.5: one small obstacle, clear line of sight to goal",
        "obstacles": [],
        "obstacle_fn": make_level15_obstacle_fn(),
    },
    {
        "name": "level2_wide_gaps",
        "title": "Level 2: 1-2 large obstacles, wide gaps",
        "obstacles": [
            CircleObstacle(x=3.5, z=2.5, radius=1.2),
            CircleObstacle(x=6.5, z=5.5, radius=1.2),
        ],
    },
    {
        "name": "level3_moderate_gaps",
        "title": "Level 3: 3-5 obstacles, moderate gaps",
        "obstacles": [
            CircleObstacle(x=2.5, z=2.0, radius=0.8),
            CircleObstacle(x=2.5, z=6.0, radius=0.8),
            CircleObstacle(x=7.5, z=2.0, radius=0.8),
            CircleObstacle(x=7.5, z=6.0, radius=0.8),
            CircleObstacle(x=5.0, z=4.0, radius=0.8),
        ],
    },
    {
        "name": "level4_dense_narrow_gaps",
        "title": "Level 4: dense obstacle field, narrow gaps",
        "obstacles": [
            CircleObstacle(x=2.0, z=2.0, radius=0.7), CircleObstacle(x=4.5, z=2.0, radius=0.7), CircleObstacle(x=7.0, z=2.0, radius=0.7),
            CircleObstacle(x=3.25, z=4.0, radius=0.7), CircleObstacle(x=5.75, z=4.0, radius=0.7), CircleObstacle(x=8.0, z=4.0, radius=0.7),
            CircleObstacle(x=2.0, z=6.0, radius=0.7), CircleObstacle(x=4.5, z=6.0, radius=0.7), CircleObstacle(x=7.0, z=6.0, radius=0.7),
        ],
    },
]


def print_progression_criterion():
    print("=" * 78)
    print("AUTOMATIC PROGRESSION CRITERION")
    print(f"  - evaluate every {EVAL_EVERY} PPO updates, over {EVAL_EPISODES} deterministic episodes")
    print(f"  - advance to next level when eval success rate >= {SUCCESS_THRESHOLD:.0%}")
    print(f"  - safety cap: stop entire run if a level fails to reach that within "
          f"{MAX_UPDATES_PER_LEVEL} updates")
    print(f"  - agent weights and optimizer state carry over between levels unchanged")
    print("=" * 78)


def train_one_level(agent, buffer, env, eval_env, level_config, max_updates):
    episode_rewards = []
    episode_outcomes = []
    eval_history = []

    current_episode_reward = 0.0
    obs = env.reset()
    solved_at_update = None

    for update in range(1, max_updates + 1):
        buffer.clear()
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=done, value=value)
            current_episode_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(current_episode_reward)
                episode_outcomes.append(info["outcome"])
                current_episode_reward = 0.0
                obs = env.reset()

        last_value = agent.get_value(obs)
        diagnostics = agent.update(buffer, last_value)

        recent_rewards = episode_rewards[-MOVING_AVG_WINDOW:] if episode_rewards else [0.0]
        recent_outcomes = episode_outcomes[-MOVING_AVG_WINDOW:]
        train_success_rate = (
            np.mean([o == "success" for o in recent_outcomes]) if recent_outcomes else 0.0
        )

        print(
            f"  [{level_config['name']}] update {update:3d}/{max_updates}  "
            f"episodes: {len(episode_rewards):5d}  moving_avg_reward: {np.mean(recent_rewards):8.2f}  "
            f"train_success_rate: {train_success_rate:5.1%}  "
            f"policy_loss: {diagnostics['policy_loss']:+.4f}  entropy: {diagnostics['entropy']:+.3f}"
        )

        if update % EVAL_EVERY == 0:
            eval_success_rate, eval_avg_reward = run_evaluation(agent, eval_env, EVAL_EPISODES)
            eval_history.append((update, eval_success_rate, eval_avg_reward))
            print(
                f"      [eval @ update {update}] success_rate: {eval_success_rate:5.1%}  "
                f"avg_reward: {eval_avg_reward:8.2f}"
            )
            if eval_success_rate >= SUCCESS_THRESHOLD:
                solved_at_update = update
                print(f"  *** {level_config['name']} solved at update {update} ***")
                break

        if update % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"phase4_{level_config['name']}_update{update:04d}.pt")
            agent.save(ckpt_path)

    return episode_rewards, episode_outcomes, eval_history, solved_at_update


def run_curriculum(
    start_level_index: int = 0,
    resume_checkpoint: str = None,
    max_updates_per_level: int = MAX_UPDATES_PER_LEVEL,
    num_levels: int = None,
):
    """
    start_level_index: skip earlier levels (e.g. 1 to start at Level 2),
        used when re-testing a specific level after a fix rather than
        re-running the whole curriculum from scratch.
    resume_checkpoint: path to load agent weights from before training
        the first level in this run -- pairs with start_level_index to
        pick up exactly where a previous run left off.
    max_updates_per_level: override for short diagnostic runs (e.g. "give
        me a 50-update trend check" rather than running the full cap).
    num_levels: stop after this many levels regardless of whether they
        solved -- used to deliberately pause for a check-in before
        continuing to the next level (e.g. "run Level 1 only, I'll
        confirm before Level 2").
    """
    print_progression_criterion()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent = PPOAgent(
        obs_dim=QuadrotorEnv.OBS_DIM, act_dim=2, continuous=True,
        init_mean_bias=[QuadrotorEnv.HOVER_ACTION, QuadrotorEnv.HOVER_ACTION],
        config=PPOConfig(entropy_coef=ENTROPY_COEF),
    )
    if resume_checkpoint is not None:
        agent.load(resume_checkpoint)
        print(f"resumed agent weights from {resume_checkpoint}")

    buffer = RolloutBuffer(gamma=agent.config.gamma, gae_lambda=agent.config.gae_lambda)

    summary = []

    levels_to_run = CURRICULUM[start_level_index:]
    if num_levels is not None:
        levels_to_run = levels_to_run[:num_levels]

    for level_config in levels_to_run:
        print(f"\n--- starting {level_config['name']}: {level_config['title']} ---")
        env = QuadrotorEnv(obstacles=level_config["obstacles"],
                           obstacle_fn=level_config.get("obstacle_fn"), seed=SEED)
        eval_env = QuadrotorEnv(obstacles=level_config["obstacles"],
                                obstacle_fn=level_config.get("obstacle_fn"), seed=SEED + 1000)

        start_time = time.time()
        episode_rewards, episode_outcomes, eval_history, solved_at_update = train_one_level(
            agent, buffer, env, eval_env, level_config, max_updates_per_level
        )
        elapsed = time.time() - start_time

        final_ckpt = os.path.join(CHECKPOINT_DIR, f"phase4_{level_config['name']}_final.pt")
        agent.save(final_ckpt)

        title = f"Phase 4: {level_config['title']}"
        if solved_at_update is not None:
            title += f" -- solved at update {solved_at_update}"
        plot_learning_curve(
            episode_rewards, eval_history, SUCCESS_THRESHOLD, MOVING_AVG_WINDOW,
            out_path=os.path.join(RESULTS_DIR, f"phase4_{level_config['name']}_learning_curve.png"),
            title=title,
        )
        demo_path = os.path.join(RESULTS_DIR, f"phase4_{level_config['name']}_demo.gif")
        outcome, n_frames = record_demo_episode(agent, eval_env, demo_path, title=title)
        print(f"  saved {demo_path} ({n_frames} frames, outcome={outcome})")

        summary.append({
            "level": level_config["name"],
            "solved": solved_at_update is not None,
            "solved_at_update": solved_at_update,
            "episodes": len(episode_rewards),
            "elapsed_sec": elapsed,
        })

        if solved_at_update is None:
            print(f"\n*** STOPPING CURRICULUM: {level_config['name']} did not reach "
                  f"{SUCCESS_THRESHOLD:.0%} success within {max_updates_per_level} updates ***")
            break
    else:
        if num_levels is not None:
            print(f"\n*** PAUSING CURRICULUM: completed the requested {num_levels} level(s); "
                  f"awaiting confirmation before continuing ***")

    print("\n" + "=" * 78)
    print("CURRICULUM SUMMARY")
    for s in summary:
        status = f"solved at update {s['solved_at_update']}" if s["solved"] else "NOT SOLVED (stopped)"
        print(f"  {s['level']:28s} {status:28s}  episodes={s['episodes']:5d}  time={s['elapsed_sec']:.1f}s")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-level", type=int, default=0, help="0-indexed level to start from")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES_PER_LEVEL,
                         help="override the per-level update cap, e.g. for a short diagnostic run")
    parser.add_argument("--num-levels", type=int, default=None,
                         help="stop after this many levels, regardless of outcome")
    args = parser.parse_args()

    run_curriculum(
        start_level_index=args.start_level,
        resume_checkpoint=args.resume_checkpoint,
        max_updates_per_level=args.max_updates,
        num_levels=args.num_levels,
    )
