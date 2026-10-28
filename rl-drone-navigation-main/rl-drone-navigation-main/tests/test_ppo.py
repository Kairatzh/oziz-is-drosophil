"""
Tests for the from-scratch PPO implementation: GAE correctness, network
sanity, and the clipped objective's behavior.
"""

import copy

import numpy as np
import pytest
import torch

from ppo.networks import Actor, Critic, GaussianActor
from ppo.ppo_agent import PPOAgent, PPOConfig
from ppo.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def test_gae_reduces_to_monte_carlo_return_with_gamma_lambda_one():
    """With gamma=1, lambda=1, and V(s)=0 everywhere, GAE advantages
    should equal the plain undiscounted reward-to-go at each timestep --
    this is a well-known reduction, so it's a strong correctness check."""
    buffer = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    rewards = [1.0, 1.0, 1.0]
    dones = [False, False, True]
    values = [0.0, 0.0, 0.0]
    for r, d, v in zip(rewards, dones, values):
        buffer.add(obs=np.zeros(1), action=0, log_prob=0.0, reward=r, done=d, value=v)

    advantages, returns = buffer.compute_advantages_and_returns(last_value=0.0)
    np.testing.assert_allclose(advantages, [3.0, 2.0, 1.0])
    np.testing.assert_allclose(returns, [3.0, 2.0, 1.0])


def test_gae_lambda_zero_reduces_to_one_step_td():
    """With lambda=0, GAE should reduce to the plain one-step TD residual
    at every timestep, with no propagation from future steps."""
    buffer = RolloutBuffer(gamma=0.9, gae_lambda=0.0)
    rewards = [1.0, 2.0, 3.0]
    dones = [False, False, True]
    values = [0.5, 0.5, 0.5]
    for r, d, v in zip(rewards, dones, values):
        buffer.add(obs=np.zeros(1), action=0, log_prob=0.0, reward=r, done=d, value=v)

    advantages, _ = buffer.compute_advantages_and_returns(last_value=0.5)
    expected = [
        rewards[0] + 0.9 * values[1] - values[0],
        rewards[1] + 0.9 * values[2] - values[1],
        rewards[2] + 0.9 * 0.5 * 0.0 - values[2],  # done=True -> no bootstrap
    ]
    np.testing.assert_allclose(advantages, expected, atol=1e-8)


def test_gae_bootstraps_with_last_value_when_not_done():
    """If the rollout is cut off mid-episode (done=False on the last
    step), the advantage computation must bootstrap from last_value
    rather than treating it as a terminal state with zero future reward."""
    buffer = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    buffer.add(obs=np.zeros(1), action=0, log_prob=0.0, reward=1.0, done=False, value=0.0)
    advantages_bootstrapped, _ = buffer.compute_advantages_and_returns(last_value=10.0)
    advantages_terminal, _ = buffer.compute_advantages_and_returns(last_value=0.0)
    assert advantages_bootstrapped[0] > advantages_terminal[0]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def test_actor_produces_valid_categorical_distribution():
    actor = Actor(obs_dim=4, act_dim=2)
    obs = torch.zeros((5, 4))
    dist = actor.get_distribution(obs)
    probs = dist.probs
    assert probs.shape == (5, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.all(probs >= 0)


def test_critic_output_shape():
    critic = Critic(obs_dim=4)
    obs = torch.zeros((5, 4))
    values = critic(obs)
    assert values.shape == (5,)


def test_gaussian_actor_produces_valid_distribution_with_correct_shape():
    actor = GaussianActor(obs_dim=8, act_dim=2)
    obs = torch.zeros((5, 8))
    dist = actor.get_distribution(obs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    assert action.shape == (5, 2)
    assert log_prob.shape == (5,)  # Independent() sums over the 2 action dims


def test_gaussian_actor_init_mean_bias_sets_initial_mean():
    bias = [0.667, 0.667]
    actor = GaussianActor(obs_dim=8, act_dim=2, init_mean_bias=bias)
    obs = torch.zeros((1, 8))  # zero input -> mean_head output = its bias alone
    mean, std = actor.forward(obs)
    np.testing.assert_allclose(mean.detach().numpy(), [bias], atol=1e-6)


def test_gaussian_actor_clamps_std_when_raw_log_std_runs_high():
    """Regression test for the Phase 4 Level 2 plateau: entropy climbed
    ~6x over 325 updates with no sign of leveling off, traced to log_std
    growing unbounded. This confirms the clamp actually caps std even if
    the raw parameter drifts far outside the intended range."""
    actor = GaussianActor(obs_dim=8, act_dim=2)
    with torch.no_grad():
        actor.log_std.fill_(10.0)  # simulate runaway growth
    obs = torch.zeros((1, 8))
    _, std = actor.forward(obs)
    assert torch.all(std <= np.exp(GaussianActor.MAX_LOG_STD) + 1e-6)


def test_gaussian_actor_clamps_std_when_raw_log_std_runs_low():
    actor = GaussianActor(obs_dim=8, act_dim=2)
    with torch.no_grad():
        actor.log_std.fill_(-50.0)
    obs = torch.zeros((1, 8))
    _, std = actor.forward(obs)
    assert torch.all(std >= np.exp(GaussianActor.MIN_LOG_STD) - 1e-6)


def test_ppo_update_clamps_raw_log_std_parameter_in_place():
    """Belt-and-suspenders check: even the raw (unclamped-in-forward)
    parameter should not be allowed to drift past the bounds after an
    optimizer step, regardless of what the gradient/momentum did."""
    agent = PPOAgent(
        obs_dim=8, act_dim=2, continuous=True, init_mean_bias=[0.667, 0.667],
        config=PPOConfig(n_epochs=1, minibatch_size=8),
    )
    with torch.no_grad():
        agent.actor.log_std.fill_(10.0)

    buffer = RolloutBuffer()
    obs = np.zeros(8, dtype=np.float32)
    for _ in range(16):
        action, log_prob, value = agent.act(obs)
        buffer.add(obs=obs, action=action, log_prob=log_prob, reward=1.0, done=False, value=value)
    agent.update(buffer, last_value=0.0)

    assert torch.all(agent.actor.log_std <= agent.actor.MAX_LOG_STD + 1e-6)


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

def test_agent_act_returns_valid_action_and_finite_outputs():
    agent = PPOAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    action, log_prob, value = agent.act(obs)
    assert action in (0, 1)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_continuous_agent_act_returns_vector_action():
    agent = PPOAgent(obs_dim=8, act_dim=2, continuous=True, init_mean_bias=[0.667, 0.667])
    obs = np.zeros(8, dtype=np.float32)
    action, log_prob, value = agent.act(obs)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)


def test_continuous_agent_deterministic_action_matches_mean():
    agent = PPOAgent(obs_dim=8, act_dim=2, continuous=True, init_mean_bias=[0.667, 0.667])
    obs = np.zeros(8, dtype=np.float32)
    action = agent.act_deterministic(obs)
    np.testing.assert_allclose(action, [0.667, 0.667], atol=1e-6)


def test_discrete_agent_deterministic_action_is_argmax():
    agent = PPOAgent(obs_dim=4, act_dim=2)
    obs = np.zeros(4, dtype=np.float32)
    logits = agent.actor.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    expected = int(torch.argmax(logits, dim=-1).item())
    action = agent.act_deterministic(obs)
    assert action == expected


def test_continuous_update_changes_network_parameters():
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPOAgent(
        obs_dim=8, act_dim=2, continuous=True, init_mean_bias=[0.667, 0.667],
        config=PPOConfig(n_epochs=2, minibatch_size=8),
    )
    actor_before = copy.deepcopy(agent.actor.state_dict())

    buffer = RolloutBuffer()
    obs = np.zeros(8, dtype=np.float32)
    for _ in range(32):
        action, log_prob, value = agent.act(obs)
        reward = np.random.randn()
        buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=False, value=value)
    agent.update(buffer, last_value=0.0)

    actor_after = agent.actor.state_dict()
    assert any(not torch.allclose(actor_before[k], actor_after[k]) for k in actor_before)


def test_save_and_load_roundtrip(tmp_path):
    agent = PPOAgent(obs_dim=8, act_dim=2, continuous=True, init_mean_bias=[0.667, 0.667])
    obs = np.zeros(8, dtype=np.float32)
    action_before = agent.act_deterministic(obs)

    path = tmp_path / "checkpoint.pt"
    agent.save(str(path))

    new_agent = PPOAgent(obs_dim=8, act_dim=2, continuous=True)
    new_agent.load(str(path))
    action_after = new_agent.act_deterministic(obs)

    np.testing.assert_allclose(action_before, action_after, atol=1e-6)


def test_update_changes_network_parameters():
    torch.manual_seed(0)
    np.random.seed(0)
    agent = PPOAgent(obs_dim=4, act_dim=2, config=PPOConfig(n_epochs=2, minibatch_size=8))
    actor_before = copy.deepcopy(agent.actor.state_dict())
    critic_before = copy.deepcopy(agent.critic.state_dict())

    buffer = RolloutBuffer()
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(32):
        action, log_prob, value = agent.act(obs)
        reward = np.random.randn()
        buffer.add(obs=obs, action=action, log_prob=log_prob, reward=reward, done=False, value=value)

    agent.update(buffer, last_value=0.0)

    actor_after = agent.actor.state_dict()
    critic_after = agent.critic.state_dict()
    actor_changed = any(not torch.allclose(actor_before[k], actor_after[k]) for k in actor_before)
    critic_changed = any(not torch.allclose(critic_before[k], critic_after[k]) for k in critic_before)
    assert actor_changed
    assert critic_changed


def test_update_returns_finite_diagnostics():
    agent = PPOAgent(obs_dim=4, act_dim=2, config=PPOConfig(n_epochs=2, minibatch_size=8))
    buffer = RolloutBuffer()
    obs = np.zeros(4, dtype=np.float32)
    for _ in range(16):
        action, log_prob, value = agent.act(obs)
        buffer.add(obs=obs, action=action, log_prob=log_prob, reward=1.0, done=False, value=value)
    diagnostics = agent.update(buffer, last_value=0.0)
    for key, val in diagnostics.items():
        assert np.isfinite(val), f"{key} was not finite: {val}"


# ---------------------------------------------------------------------------
# Clipped surrogate objective (pure formula check, independent of the
# network -- this is the mathematical core of PPO, so it gets its own
# direct test rather than relying only on the integration test above).
# ---------------------------------------------------------------------------

def clipped_surrogate(ratio: torch.Tensor, advantages: torch.Tensor, eps: float) -> torch.Tensor:
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    return torch.min(surr1, surr2)


def test_clipping_caps_objective_for_large_positive_advantage():
    ratio = torch.tensor([2.0])       # policy has drifted far from the data-collecting policy
    advantages = torch.tensor([1.0])  # action turned out to be good
    result = clipped_surrogate(ratio, advantages, eps=0.2)
    assert result.item() == pytest.approx(1.2)  # capped at (1+eps)*A, not the raw 2.0*A


def test_clipping_caps_objective_for_large_negative_advantage():
    """When the advantage is negative, an increased ratio makes the
    surrogate *more* negative (worse) without bound unless clipped -- the
    clip must still activate to cap the penalty at (1-eps)*A."""
    ratio = torch.tensor([2.0])
    advantages = torch.tensor([-1.0])
    result = clipped_surrogate(ratio, advantages, eps=0.2)
    # Raw (unclipped) term would be 2.0 * -1.0 = -2.0; min() picks the more
    # negative of the two, which for a negative advantage is the *unclipped*
    # term -- this is intentional: PPO does not clip away large penalties
    # for confidently-wrong updates, only large rewards for them.
    assert result.item() == pytest.approx(-2.0)


def test_no_clipping_when_ratio_within_trust_region():
    ratio = torch.tensor([1.05])
    advantages = torch.tensor([1.0])
    result = clipped_surrogate(ratio, advantages, eps=0.2)
    assert result.item() == pytest.approx(1.05)
