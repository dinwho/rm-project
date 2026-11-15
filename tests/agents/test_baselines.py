from dataclasses import replace

import numpy as np
import pytest

from wildfire_crmab.agents.baselines import BASELINE_NAMES, NoIntervention, RandomPlacement, ThreatHeuristic, make_baseline
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.environment.scenario import Scenario


def empty_observation(n=12):
    layers = np.zeros((6, n, n), dtype=np.float32)
    layers[0] = 1
    return {"map": layers, "context": np.array([0, 0.8, 1, 1], dtype=np.float32)}


@pytest.mark.parametrize("name", BASELINE_NAMES)
@pytest.mark.parametrize("budget", [None, 0, 1, 6])
def test_policies_share_environment_feasibility(name, budget):
    config = EnvironmentConfig(max_steps=12)
    env = WildfireEnv(config)
    agent = make_baseline(name, config, seed=8)
    observation, info = env.reset(options={"scenario": Scenario(seed=25, budget=budget)})
    for _ in range(config.max_steps):
        action = agent.act(observation, info["action_mask"])
        assert info["action_mask"][action]
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    if budget is not None:
        assert info["budget_used"] <= budget
    assert info["budget_used"] <= info["steps"] * config.capacity * config.resource_cost


def test_random_is_reproducible_and_samples_whole_feasible_catalogue():
    mask = np.array([True, False, True, False, True])
    left, right = RandomPlacement(18), RandomPlacement(18)
    a = [left.act({}, mask) for _ in range(100)]
    b = [right.act({}, mask) for _ in range(100)]
    assert a == b
    assert set(a) == {0, 2, 4}


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_no_policy_can_return_invalid_action_from_empty_mask(name):
    agent = make_baseline(name, EnvironmentConfig())
    with pytest.raises(ValueError):
        agent.act(empty_observation(), np.zeros(1, dtype=bool))


def test_no_intervention_requires_empty_allocation():
    with pytest.raises(ValueError):
        NoIntervention().act({}, np.array([False, True]))


def test_heuristics_preserve_material_when_no_candidate_threatened():
    for use_wind in (True, False):
        heuristic = ThreatHeuristic(EnvironmentConfig(), use_wind)
        mask = np.ones(heuristic.allocations.count, dtype=bool)
        assert heuristic.act(empty_observation(), mask) == 0


def test_heuristic_can_allocate_two_simultaneous_segments():
    heuristic = ThreatHeuristic(EnvironmentConfig(
        placement_mode="segments", segment_length=3, candidate_stride=4,
        capacity=2, initial_budget=6, include_settlement=True,
        settlement_weight=10), use_wind=False)
    # Find disjoint, well-separated segments and put a fire beside each.
    pair = next(subset for subset in heuristic.allocations.subsets
                if len(subset) == 2 and
                np.linalg.norm(np.array(heuristic.arms.cells(subset[0])).mean(axis=1)
                               - np.array(heuristic.arms.cells(subset[1])).mean(axis=1)) >= 5)
    obs = empty_observation()
    protected = set()
    for arm in pair:
        rr, cc = heuristic.arms.cells(arm)
        protected.update(zip(rr.tolist(), cc.tolist()))
    for arm in pair:
        rr, cc = heuristic.arms.cells(arm)
        neighbours = [(int(r + dr), int(c + dc)) for r, c in zip(rr, cc)
                      for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
        r, c = next((r, c) for r, c in neighbours if 0 <= r < 12 and 0 <= c < 12
                    and (r, c) not in protected)
        obs["map"][1, r, c] = 1
        obs["map"][3, r, c] = 1
    mask = np.array([set(subset) <= set(pair) for subset in heuristic.allocations.subsets])
    action = heuristic.act(obs, mask)
    assert heuristic.allocations.subsets[action] == pair
    assert (heuristic.arm_scores(obs)[list(pair)] > 0).all()


def test_wind_heuristic_only_uses_public_wind_when_enabled():
    config = EnvironmentConfig()
    blind = ThreatHeuristic(config, use_wind=False)
    aware = ThreatHeuristic(config, use_wind=True)
    obs = empty_observation()
    # Locate a source beside a catalogue segment so wind changes a candidate score.
    rr, cc = blind.arms.cells(0)
    r, c = int(rr[0]), int(cc[0])
    source = (r + 1, c) if r + 1 < config.grid_size else (r - 1, c)
    obs["map"][1, source[0], source[1]] = 1
    obs["map"][3, source[0], source[1]] = 1
    obs["context"][:2] = (0.8, 0.6)
    blind_score = blind.arm_scores(obs)
    aware_score = aware.arm_scores(obs)
    obs["context"][:2] *= -1
    np.testing.assert_array_equal(blind_score, blind.arm_scores(obs))
    assert not np.allclose(aware_score, aware.arm_scores(obs))
    no_physical_wind = ThreatHeuristic(replace(config, wind_physics=False), use_wind=True)
    np.testing.assert_array_equal(blind_score, no_physical_wind.arm_scores(obs))
