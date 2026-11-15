from dataclasses import replace
from itertools import combinations

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from wildfire_crmab.environment.allocations import AllocationCatalogue
from wildfire_crmab.environment.arms import CandidateArms
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.environment.scenario import Scenario, generate_world
from wildfire_crmab.environment.spread import ignition_probability, shifted


def reset_env(config=None, budget=6, seed=3, observe_wind=True):
    env = WildfireEnv(config, observe_wind=observe_wind)
    env.reset(options={"scenario": Scenario(seed, budget, (0.0, 0.8))})
    return env


def first_pair(env):
    return next(int(i) for i in np.flatnonzero(env.action_mask())
                if len(env.allocations.decode(int(i))) == 2)


def corridor(env):
    """A controlled physical corridor separates spread from stochastic sampling."""
    env.fuel[:] = 0
    env.fuel[2, :] = 1
    env.settlement[:] = False
    env.settlement[2, -1] = True
    env.vegetation = (env.fuel > 0) & ~env.settlement
    env.fire_age[:] = 0
    env.fire_age[2, 0] = env.config.burn_duration
    env.ever_burned[:] = False
    env.ever_burned[2, 0] = True
    env.world.spread_draws[:] = 0


def test_gym_contract_and_observation_contains_explicit_arms():
    # Gym's checker samples without a feasibility mask. Check the general return
    # contract in the always-feasible wait-only configuration; default feasibility
    # and strict invalid-call semantics are exercised separately below.
    check_env(WildfireEnv(replace(EnvironmentConfig(), capacity=0)), skip_render_check=True)
    env = WildfireEnv()
    obs, info = env.reset(seed=2)
    assert env.observation_space.contains(obs)
    assert obs["arms"].shape == (18, 10)
    assert env.allocations.count == 163
    np.testing.assert_array_equal(obs["arms"][:, 0], env.built)
    np.testing.assert_array_equal(obs["arms"][:, 1], env.eligibility())
    assert info["action_mask"][0]


def test_both_segments_block_spread_before_the_single_transition():
    env = reset_env(replace(EnvironmentConfig(), base_spread=1, burn_duration=10))
    corridor(env)
    env.fire_age[:] = 0
    env.fire_age[2, 4] = 10
    env.ever_burned[:] = False
    env.ever_burned[2, 4] = True
    # Arm 0 covers columns 1..3; arm 2 covers 5..7 on the same corridor.
    _, _, _, _, info = env.step(env.allocations.encode((0, 2)))
    assert info["newly_ignited_vegetation"] == 0
    assert info["newly_cleared"] == 6
    assert env.fire_age[2, 4] == 9 and env.step_number == 1


def test_catalogue_is_exact_disjoint_subsets_with_capacity_limit():
    arms = CandidateArms(EnvironmentConfig())
    catalogue = AllocationCatalogue(arms)
    expected = {()}
    for size in (1, 2):
        for subset in combinations(range(arms.count), size):
            cells = [set(arms.segments[i].cells) for i in subset]
            if size == 1 or cells[0].isdisjoint(cells[1]):
                expected.add(subset)
    assert set(catalogue.subsets) == expected
    assert catalogue.decode(catalogue.encode((0, 2))) == (0, 2)
    for invalid in ((0, 0), (0, 1), (0, 2, 4)):
        with pytest.raises(ValueError):
            catalogue.encode(invalid)
    with pytest.raises(ValueError, match="bound"):
        AllocationCatalogue(arms, capacity=4, max_allocations=100)


def test_passive_and_unavailable_arms_continue_evolving():
    env = reset_env(replace(EnvironmentConfig(), base_spread=1, burn_duration=10))
    corridor(env)
    before = env.arm_states()
    env.step(0)
    after = env.arm_states()
    assert env.fire_age[2, 1] == 10
    assert env.fire_age[2, 0] == 9
    assert not np.array_equal(before[0].fire_age, after[0].fire_age)
    assert not after[0].built and not after[0].eligible
    env.step(0)
    assert not np.array_equal(after[0].fire_age, env.arm_states()[0].fire_age)
    assert not env.built.any()


def test_active_construction_persists_and_cannot_be_built_again():
    env = reset_env(replace(EnvironmentConfig(), base_spread=1, burn_duration=10))
    corridor(env)
    action = env.allocations.encode((0,))
    env.step(action)
    rows, cols = env.arms.cells(0)
    assert env.built[0] and env.firebreak[rows, cols].all()
    assert not env.fuel[rows, cols].any()
    assert not env.action_mask()[action]
    before = env.arm_states()[0]
    for _ in range(3):
        env.step(0)
    assert env.built[0] and env.firebreak[rows, cols].all()
    assert not env.ever_burned[rows, cols].any()
    assert not np.array_equal(before.fire_age, env.arm_states()[0].fire_age)


def test_joint_allocation_builds_both_before_exactly_one_fire_tick():
    env = reset_env(replace(EnvironmentConfig(), base_spread=0, burn_duration=10))
    action = first_pair(env)
    subset = env.allocations.decode(action)
    original = env.fire_age.copy()
    obs, _, _, _, info = env.step(action)
    assert env.step_number == 1
    assert env.budget == 4
    assert info["constructed_count"] == 2 and info["resources_spent"] == 2
    assert info["newly_cleared"] == 6
    np.testing.assert_array_equal(env.fire_age, np.maximum(original-1, 0))
    for i in subset:
        r, c = env.arms.cells(i)
        assert env.firebreak[r, c].all() and env.built[i]
        assert obs["arms"][i, 0] == 1


def test_construction_capacity_renews_but_material_budget_does_not():
    env = reset_env(replace(EnvironmentConfig(), base_spread=0, burn_duration=10), budget=5)
    env.step(first_pair(env))
    assert env.budget == 3
    env.step(first_pair(env))
    assert env.budget == 1 and env.built.sum() == 4
    assert all(len(env.allocations.decode(int(a))) <= 1 for a in np.flatnonzero(env.action_mask()))
    action = int(np.flatnonzero(env.action_mask())[1])
    env.step(action)
    assert env.budget == 0 and not env.finished
    assert np.flatnonzero(env.action_mask()).tolist() == [0]
    age = env.fire_age.copy()
    env.step(0)
    np.testing.assert_array_equal(env.fire_age, np.maximum(age-1, 0))


def test_resource_units_affordability_and_atomic_invalid_rejection():
    env = reset_env(replace(EnvironmentConfig(), resource_cost=2), budget=3)
    pair = next(i for i, s in enumerate(env.allocations.subsets)
                if len(s) == 2 and env.eligibility()[list(s)].all())
    before = env.observation()
    for invalid in (pair, -1, env.allocations.count, True, 1.5):
        with pytest.raises(ValueError):
            env.step(invalid)
        assert env.budget == 3 and env.step_number == 0
        for key in before:
            np.testing.assert_array_equal(before[key], env.observation()[key])
    action = int(np.flatnonzero(env.action_mask())[1])
    env.step(action)
    assert env.budget == 1
    assert np.flatnonzero(env.action_mask()).tolist() == [0]


@pytest.mark.parametrize("reason", ["built", "fuel", "burned", "cleared", "asset"])
def test_physical_eligibility_rejects_invalid_sites(reason):
    env = reset_env()
    arm = int(np.flatnonzero(env.eligibility())[0])
    row, col = env.arms.segments[arm].cells[0]
    if reason == "built":
        env.built[arm] = True
    elif reason == "fuel":
        env.fuel[row, col] = 0
    elif reason == "burned":
        env.ever_burned[row, col] = True
    elif reason == "cleared":
        env.firebreak[row, col] = True
    else:
        env.settlement[row, col] = True
    assert not env.eligibility()[arm]
    with pytest.raises(ValueError):
        env.step(env.allocations.encode((arm,)))


def test_overlapping_site_is_ineligible_without_falsely_marking_it_built():
    env = reset_env(replace(EnvironmentConfig(), base_spread=0))
    assert env.eligibility()[0] and env.eligibility()[1]
    env.step(env.allocations.encode((0,)))
    assert env.built[0] and not env.built[1]
    assert not env.eligibility()[1]


def test_building_changes_another_passive_arms_transition():
    config = replace(EnvironmentConfig(), base_spread=1, burn_duration=20)
    active, passive = reset_env(config), reset_env(config)
    corridor(active)
    corridor(passive)
    active.step(active.allocations.encode((0,)))
    passive.step(0)
    for _ in range(5):
        active.step(0)
        passive.step(0)
    # Arm 2 is a separate horizontal segment at columns 5..7; never selected.
    assert not active.built[2] and not passive.built[2]
    assert active.arm_states()[2].eligible
    assert not passive.arm_states()[2].eligible
    assert not active.ever_burned[2, 6] and passive.ever_burned[2, 6]


def test_new_ignitions_do_not_cascade_within_one_interval():
    env = reset_env(replace(EnvironmentConfig(), base_spread=1, burn_duration=10))
    corridor(env)
    env.step(0)
    assert env.ever_burned[2, 1]
    assert not env.ever_burned[2, 2]


def test_reward_charges_new_damage_and_clearing_once():
    env = reset_env(replace(EnvironmentConfig(), base_spread=0, burn_duration=10))
    count = int(env.vegetation.sum())
    _, reward, _, _, _ = env.step(first_pair(env))
    assert reward == pytest.approx(-6/count)
    _, reward, _, _, _ = env.step(0)
    assert reward == 0
    assert env.info()["affected_fraction"] >= env.info()["burned_fraction"]


def test_asset_reward_is_charged_on_first_ignition_only():
    env = reset_env(replace(EnvironmentConfig(), base_spread=1, burn_duration=10))
    corridor(env)
    env.settlement[:] = False
    env.settlement[2, 1] = True
    env.vegetation = (env.fuel > 0) & ~env.settlement
    _, reward, _, _, _ = env.step(0)
    assert reward == -env.config.settlement_weight
    _, reward, _, _, info = env.step(0)
    assert info["newly_ignited_assets"] == 0
    assert reward > -env.config.settlement_weight


def test_wind_is_fixed_and_observation_ablation_leaves_physics_identical():
    aware, blind = reset_env(), reset_env(observe_wind=False)
    wind = aware.wind.copy()
    for _ in range(10):
        if aware.finished:
            break
        action = int(np.flatnonzero(aware.action_mask())[-1])
        a, r, term, trunc, _ = aware.step(action)
        b, rb, termb, truncb, _ = blind.step(action)
        np.testing.assert_array_equal(a["map"], b["map"])
        np.testing.assert_array_equal(a["arms"], b["arms"])
        np.testing.assert_array_equal(aware.wind, wind)
        assert not b["context"][:2].any()
        assert (r, term, trunc) == (rb, termb, truncb)


def test_removing_wind_physics_is_a_distinct_intervention():
    driven = reset_env()
    calm = reset_env(replace(EnvironmentConfig(), wind_physics=False))
    for env in (driven, calm):
        corridor(env)
        env.world.spread_draws[:] = 0.5
    driven.step(0)
    calm.step(0)
    assert driven.ever_burned[2, 1] and not calm.ever_burned[2, 1]


def test_common_randomness_is_independent_of_budget_and_physics_ablation():
    config = EnvironmentConfig()
    a = generate_world(config, Scenario(42, 1))
    b = generate_world(replace(config, wind_physics=False), Scenario(42, 9))
    for field in ("fuel", "settlement", "wind", "spread_draws"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert a.ignition == b.ignition
    other = generate_world(config, Scenario(43, 1))
    assert not np.array_equal(a.wind, other.wind)


def test_no_spread_extinguishes_and_finite_horizon_ends_episode():
    env = reset_env(replace(EnvironmentConfig(), base_spread=0))
    for _ in range(env.config.burn_duration):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(0)
    env = reset_env(replace(EnvironmentConfig(), max_steps=1))
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_spread_has_no_wrap_and_favors_downwind():
    burning = np.zeros((7, 7), dtype=bool)
    burning[3, 3] = True
    p = ignition_probability(burning, np.ones((7, 7)), np.array([0, 0.8]), 0.2, 1.2)
    assert p[3, 4] > p[3, 2]
    corner = np.zeros((7, 7))
    corner[0, 0] = 1
    assert shifted(corner, -1, 0).sum() == 0


def test_zero_capacity_and_zero_budget_preserve_passive_action():
    env = reset_env(replace(EnvironmentConfig(), capacity=0), budget=0)
    assert env.action_space.n == 1 and env.eligibility().any()
    assert env.action_mask().tolist() == [True]
    env.step(0)
    assert env.step_number == 1


@pytest.mark.parametrize("kwargs", [{"grid_size": 5}, {"segment_length": 2},
                                    {"capacity": -1}, {"resource_cost": 0},
                                    {"base_spread": float("nan")}, {"capacity": 1.5}])
def test_configuration_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        EnvironmentConfig(**kwargs)
