import numpy as np
import pytest
import torch

from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.replay import ReplayBuffer
from wildfire_crmab.agents.network import bellman_targets


SHAPES = {"map": (6, 12, 12), "arms": (18, 10), "context": (4,)}


def observation():
    return {key: np.zeros(shape, np.float32) for key, shape in SHAPES.items()}


def test_bellman_target_masks_invalid_high_q_and_handles_terminal_empty_mask():
    targets = bellman_targets(
        torch.tensor([1., -3.]), torch.tensor([False, True]),
        torch.tensor([[2., 10000., 4.], [float("nan"), float("nan"), float("nan")]]),
        torch.tensor([[True, False, True], [False, False, False]]), 0.5,
    )
    torch.testing.assert_close(targets, torch.tensor([3., -3.]))
    with pytest.raises(ValueError, match="no feasible"):
        bellman_targets(torch.tensor([0.]), torch.tensor([False]), torch.zeros(1, 3),
                        torch.zeros(1, 3, dtype=torch.bool), .99)


def test_both_exploration_and_greedy_only_select_feasible_allocations():
    agent = DQNAgent(SHAPES, 5, 123)
    with torch.no_grad():
        agent.online.head[-1].weight.zero_()
        agent.online.head[-1].bias.copy_(torch.tensor([10000., 3., 4., 5., 6.]))
    mask = np.array([False, True, False, True, False])
    assert agent.act(observation(), mask) == 3
    assert {agent.act(observation(), mask, epsilon=1) for _ in range(50)} == {1, 3}
    with pytest.raises(ValueError):
        agent.act(observation(), np.zeros(5, bool))


def test_replay_rejects_infeasible_experience_without_mutating_buffer():
    replay = ReplayBuffer(5, SHAPES, 3)
    mask = np.array([True, False, True])
    with pytest.raises(ValueError, match="infeasible"):
        replay.add(observation(), 1, 0., observation(), False, mask, mask)
    with pytest.raises(ValueError, match="feasible"):
        replay.add(observation(), 0, 0., observation(), False, mask, np.zeros(3, bool))
    with pytest.raises(ValueError, match="schema"):
        replay.add({"map": np.zeros((6, 12, 12))}, 0, 0., observation(), False, mask, mask)
    assert replay.size == replay.position == 0


def test_replay_copies_observations_and_preserves_terminal_masks():
    replay = ReplayBuffer(2, SHAPES, 3)
    obs = observation()
    replay.add(obs, 0, -1., obs, True, np.ones(3, bool), np.zeros(3, bool))
    obs["arms"][:] = 1
    batch = replay.sample(1)
    assert not batch["observations"]["arms"].any()
    assert batch["done"].item()
    assert not batch["next_masks"].any()


def test_optimizer_updates_online_then_target_and_checkpoint_roundtrip(tmp_path):
    agent = DQNAgent(SHAPES, 3, 11)
    replay = ReplayBuffer(4, SHAPES, 3)
    for action in (0, 2):
        replay.add(observation(), action, -1., observation(), False,
                   np.ones(3, bool), np.array([True, False, False]))
    original = [p.detach().clone() for p in agent.online.parameters()]
    loss = agent.learn(replay.sample(2))
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(original, agent.online.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(original, agent.target.parameters()))
    agent.update_target()
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))
    path = tmp_path / "policy.pt"
    metadata = {"observe_wind": True, "allocation_signature": "test-schema", "step": 2}
    agent.save(path, metadata)
    restored, saved = DQNAgent.load(path)
    assert saved == metadata
    with torch.no_grad():
        batch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in observation().items()}
        torch.testing.assert_close(restored.online(batch), agent.online(batch), rtol=0, atol=0)


def test_observation_action_compatibility_with_real_environment():
    from wildfire_crmab.environment.env import WildfireEnv

    env = WildfireEnv()
    obs, info = env.reset(seed=16)
    agent = DQNAgent(env.observation_space, env.action_space.n, 16)
    replay = ReplayBuffer(4, env.observation_space, env.action_space.n)
    for _ in range(3):
        action = agent.act(obs, info["action_mask"], epsilon=1)
        assert info["action_mask"][action]
        successor, reward, terminated, truncated, next_info = env.step(action)
        replay.add(obs, action, reward, successor, terminated or truncated,
                   info["action_mask"], next_info["action_mask"])
        obs, info = successor, next_info
        if terminated or truncated:
            break
    assert np.isfinite(agent.learn(replay.sample(2)))
