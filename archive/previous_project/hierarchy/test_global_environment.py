from dataclasses import replace
from math import comb
import numpy as np
import pytest

from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.environment.scenario import Scenario
from wildfire_crmab.hierarchy.environment import GlobalWildfireEnv
from wildfire_crmab.hierarchy.baselines import AllocationBaseline


class FirstEligible:
    def act(self, observation, mask):
        return int(np.flatnonzero(mask)[1]) if mask.sum() > 1 else 0


class WaitingPolicy:
    def act(self, observation, mask):
        return 0


def config(**kwargs):
    return EnvironmentConfig(grid_size=6, base_spread=0, burn_duration=8, max_steps=12, **kwargs)


def make_env(local_config=None, policy=None, capacity=2):
    env = GlobalWildfireEnv(local_config or config(), policy or FirstEligible(), team_capacity=capacity)
    obs, info = env.reset(seed=123)
    return env, obs, info


@pytest.mark.parametrize("capacity", [0, 1, 2, 3])
def test_stable_catalogue_with_exact_capacity_masks(capacity):
    env, obs, info = make_env(capacity=capacity)
    assert env.observation_space.contains(obs)
    assert env.action_space.n == 16
    assert info["action_mask"].sum() == sum(comb(4, k) for k in range(capacity+1))
    for i in np.flatnonzero(info["action_mask"]):
        assert len(env.subsets[i]) <= capacity
    assert obs["context"][0] == capacity/4
    if capacity < 4:
        before = [e.observation() for e in env.incidents]
        with pytest.raises(ValueError, match="capacity"):
            env.step(env.encode(tuple(range(capacity+1))))
        assert env.step_number == 0
        for prior, incident in zip(before, env.incidents):
            for key in prior:
                np.testing.assert_array_equal(prior[key], incident.observation()[key])


def test_all_construction_precedes_all_fire_updates_and_unassigned_evolve():
    env, _, _ = make_env()
    before = [e.fire_age.copy() for e in env.incidents]
    for incident in env.incidents:
        advance = incident.advance_fire
        def checked_advance(advance=advance):
            assert env.incidents[0].built.sum() == env.incidents[2].built.sum() == 1
            assert not env.incidents[1].built.any() and not env.incidents[3].built.any()
            return advance()
        incident.advance_fire = checked_advance
    _, reward, terminated, truncated, info = env.step(env.encode((0, 2)))
    assert not terminated and not truncated
    assert info["incident_ticks"] == [1, 1, 1, 1]
    assert info["selected_incidents"] == (0, 2)
    assert info["constructed_this_step"] == info["cleared_cells"] == 2
    assert reward == pytest.approx(-2/(4*6*6))
    assert info["team_utilization"] == info["construction_utilization"] == 1
    for age, incident in zip(before, env.incidents):
        np.testing.assert_array_equal(incident.fire_age, np.maximum(age-1, 0))


def test_policy_calls_see_preconstruction_state_and_invalid_call_is_atomic():
    class BadSecond:
        def __init__(self):
            self.calls = 0
        def act(self, obs, mask):
            assert not obs["map"][4].any()
            self.calls += 1
            return int(np.flatnonzero(mask)[1]) if self.calls == 1 else -1
    policy = BadSecond()
    env, _, _ = make_env(policy=policy)
    before = [e.observation() for e in env.incidents]
    with pytest.raises(ValueError):
        env.step(env.encode((0, 1)))
    assert policy.calls == 2
    assert env.step_number == env.total_assignments == 0
    for prior, incident in zip(before, env.incidents):
        assert incident._pending_construction is None
        for key in prior:
            np.testing.assert_array_equal(prior[key], incident.observation()[key])


def test_changing_one_incident_allocation_does_not_change_another_transition():
    cfg = EnvironmentConfig(grid_size=6, fixed_wind=(0.2, 0.5))
    left, _, _ = make_env(cfg)
    right, _, _ = make_env(cfg)
    standalone = WildfireEnv(cfg)
    standalone.reset(options={"scenario": Scenario(left.incident_seeds[0], None)})
    action = FirstEligible().act(standalone.observation(), standalone.action_mask())
    left.step(left.encode((0,)))
    right.step(right.encode((0, 1)))
    standalone.step(action)
    for key, value in standalone.observation().items():
        np.testing.assert_array_equal(value, left.incidents[0].observation()[key])
        np.testing.assert_array_equal(value, right.incidents[0].observation()[key])
    assert not np.array_equal(left.incidents[0].world.spread_draws, left.incidents[1].world.spread_draws)


def test_zero_teams_has_no_local_calls_but_fire_evolves_to_extinction():
    class ForbiddenCall:
        def act(self, obs, mask):
            raise AssertionError("No team means no constructor call")
    env, _, _ = make_env(policy=ForbiddenCall(), capacity=0)
    while not env.finished:
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0 and not truncated
    assert terminated and info["steps"] == 8
    assert info["cleared_cells"] == info["assigned_team_intervals"] == info["team_utilization"] == 0
    assert info["contained_count"] == info["extinguished_count"] == 4
    with pytest.raises(RuntimeError):
        env.step(0)


def test_local_wait_is_allowed_but_team_assignment_still_counts():
    env, _, _ = make_env(policy=WaitingPolicy())
    _, _, _, _, info = env.step(env.encode((1, 3)))
    assert info["team_utilization"] == 1
    assert info["construction_utilization"] == 0
    assert info["local_actions"] == (0, 0, 0, 0)
    assert info["incident_ticks"] == [1]*4


def test_extinguished_incident_is_absorbing_and_unavailable():
    env, _, _ = make_env()
    incident = env.incidents[1]
    incident.fire_age[:] = 0
    incident.finished = True
    prior = incident.observation()
    assert all(1 not in env.subsets[a] for a in np.flatnonzero(env.action_mask()))
    env.step(env.encode((0, 2)))
    assert incident.step_number == 0
    for key in prior:
        np.testing.assert_array_equal(prior[key], incident.observation()[key])
    assert env.info()["local_actions"][1] is None


def test_full_burnout_is_not_counted_as_containment_and_horizon_not_extinction():
    env, _, _ = make_env(replace(config(), max_steps=1))
    _, _, terminated, truncated, info = env.step(0)
    assert truncated and not terminated and info["contained_count"] == 0
    for incident in env.incidents:
        incident.fire_age[:] = 0
        incident.ever_burned[:] = True
        incident.fuel[:] = 0
    info = env.info()
    assert info["extinguished_count"] == 4 and info["contained_count"] == 0


def test_staged_local_transition_matches_ordinary_step_and_guards_double_advance():
    cfg = EnvironmentConfig(grid_size=6)
    a, b = WildfireEnv(cfg), WildfireEnv(cfg)
    for env in (a, b):
        env.reset(options={"scenario": Scenario(5, None)})
    action = FirstEligible().act(a.observation(), a.action_mask())
    expected = a.step(action)
    b.construct(action)
    assert b.step_number == 0 and b.built.sum() == 1
    with pytest.raises(RuntimeError):
        b.construct(0)
    actual = b.advance_fire()
    for key in expected[0]:
        np.testing.assert_array_equal(expected[0][key], actual[0][key])
    assert actual[1:4] == expected[1:4]
    with pytest.raises(RuntimeError):
        b.advance_fire()


@pytest.mark.parametrize("name", ["random", "severity", "risk", "no_allocation"])
@pytest.mark.parametrize("capacity", [0, 1, 2, 3])
def test_allocation_baselines_obey_same_masks(name, capacity):
    env, obs, info = make_env(EnvironmentConfig(grid_size=6), capacity=capacity)
    policy = AllocationBaseline(name, env.subsets, env.local_config, seed=7)
    while not env.finished:
        action = policy.act(obs, info["action_mask"])
        assert info["action_mask"][action]
        obs, _, _, _, info = env.step(action)
    assert info["team_utilization"] <= 1


def test_reset_capacity_and_incident_draws_are_reproducible():
    env, first, _ = make_env()
    seeds = env.incident_seeds
    second, _ = env.reset(seed=123, options={"team_capacity": 3})
    assert env.incident_seeds == seeds
    np.testing.assert_array_equal(first["maps"], second["maps"])
    assert second["context"][0] == .75
    with pytest.raises(ValueError):
        env.reset(seed=123, options={"incident_seeds": [1, 1, 1, 1]})
