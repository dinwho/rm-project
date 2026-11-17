"""Lower-level learning, masks, finite horizon and frozen checkpoint contract."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from sb3_contrib import MaskablePPO
from wildfire_crmab import EnvironmentConfig, Scenario, WildfireEnv
from wildfire_crmab.hierarchy.local.policy import FrozenLocalPolicy, LocalConstructionEnv
from wildfire_crmab.hierarchy.local.training import train_local, evaluate_local


def test_service_outage_forces_wait_but_fire_evolves():
    config = EnvironmentConfig(grid_size=6, base_spread=0, burn_duration=10)
    env = LocalConstructionEnv(config, service_probability=0)
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    assert env.action_masks().sum() == 1 and not obs["eligible"].any()
    with pytest.raises(ValueError):
        env.step(int(np.flatnonzero(env.base.action_mask())[1]))
    assert env.base.step_number == 0
    before = env.base.fire_age.copy()
    env.step(0)
    np.testing.assert_array_equal(env.base.fire_age, np.maximum(before-1, 0))
    assert env.base.step_number == 1 and not env.base.firebreak.any()


def test_finite_horizon_is_terminal_for_ppo_without_timeout_bootstrap():
    config = EnvironmentConfig(grid_size=6, max_steps=1, burn_duration=3, base_spread=0)
    env = LocalConstructionEnv(config, service_probability=1)
    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["finite_horizon_reached"]
    assert env.base.fire_age.any()


def test_trained_local_checkpoint_freezes_valid_policy_and_rejects_mismatch(tmp_path):
    config = EnvironmentConfig(grid_size=6, max_steps=5)
    policy = train_local(tmp_path/"local", config, steps=128, rollout_steps=64, n_epochs=1,
                         validation_every=64, validation_seeds=[1_100_000_000])
    initial = MaskablePPO.load(tmp_path/"local"/"initial.zip", device="cpu")
    last = MaskablePPO.load(tmp_path/"local"/"last.zip", device="cpu")
    assert any(not torch.equal(a, b) for a, b in zip(initial.policy.parameters(), last.policy.parameters()))
    assert not any(p.requires_grad for p in policy.model.policy.parameters())
    assert policy.sha256 == policy.checkpoint_sha256
    snapshots = [p.detach().clone() for p in policy.model.policy.parameters()]
    env = WildfireEnv(config)
    obs, info = env.reset(options={"scenario": Scenario(99, None)})
    while not env.finished:
        action = policy.act(obs, info["action_mask"])
        assert info["action_mask"][action]
        assert action == policy.act(obs, info["action_mask"])
        obs, _, _, _, info = env.step(action)
    assert all(torch.equal(a, b) for a, b in zip(snapshots, policy.model.policy.parameters()))
    with pytest.raises(ValueError, match="mismatch"):
        FrozenLocalPolicy.load(tmp_path/"local"/"best.zip", replace(config, wind_bias=1.))
    rows = evaluate_local(tmp_path/"eval", policy, config, seeds=[2_100_000_000], service_probabilities=(1.,))
    assert len(rows) == 5 and all(r["cleared_cells"] <= r["steps"] for r in rows)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_local(tmp_path/"bad", policy, config, seeds=[1_100_000_000])
    metadata = tmp_path/"local"/"best.json"
    saved = json.loads(metadata.read_text())
    saved["sha256"] = "bad"
    metadata.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum"):
        FrozenLocalPolicy.load(metadata.with_suffix(".zip"), config)
