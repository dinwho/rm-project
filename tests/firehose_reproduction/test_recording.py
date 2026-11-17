"""Regression checks for real simulator provenance and recording conventions."""
import numpy as np
import pytest
from firehose_reproduction.recording import record
from firehose_reproduction.paths import BINARY


@pytest.mark.skipif(not BINARY.exists(), reason="Run firehose setup to compile Cell2Fire")
def test_recorded_network_actions_are_real_unique_treatments_and_reproducible():
    first = record("ppo")
    second = record("ppo")
    assert first["checkpoint_sha256"] and first["training_steps"] == 5013504
    assert first["frames"] == second["frames"]
    actions = [frame["action"] for frame in first["frames"][1:]]
    assert len(actions) == len(set(actions))
    assert first["frames"][0]["action"] is None
    assert first["frames"][-1]["harvested"] == len(actions)
    assert first["metrics"]["reward"] == pytest.approx(sum(f["reward"] for f in first["frames"]))
    assert np.asarray(first["frames"][0]["rgb"]).shape == (20, 20, 3)


@pytest.mark.skipif(not BINARY.exists(), reason="Run firehose setup")
def test_second_map_uses_matching_policy_geometry_and_original_timing():
    episode = record("ppo", checkpoint_name="40x40-fixed-ppo-maskable.zip")
    assert episode["map"] == "Sub40x40" and episode["ignition"] == 909
    assert episode["warmup_steps"] == 25 and episode["physics_steps_per_action"] == 5
    assert episode["training_steps"] == 10027008
    assert np.asarray(episode["frames"][0]["rgb"]).shape == (40, 40, 3)
    actions = [frame["action"] for frame in episode["frames"][1:]]
    assert all(0 <= action < 1600 for action in actions)
    assert len(set(actions)) == len(actions)
