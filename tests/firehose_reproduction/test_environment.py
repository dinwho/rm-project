"""Integration checks against actual pinned Cell2Fire, not mock spread physics."""
import numpy as np
import pytest
from firehose_reproduction.environment import make_env
from firehose_reproduction.paths import BINARY

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason="Run firehose setup to compile Cell2Fire")


def test_original_reset_timing_wait_harvest_and_repeat_mask():
    env = make_env()
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == (3, 20, 20) and obs.dtype == np.uint8
        assert env.observation_space.contains(obs)
        assert env.legacy.steps_before_sim == 20 and env.legacy.steps_per_action == 8
        # Preserve the upstream reset observation behavior even after warmup.
        assert not env.legacy.state.any() and info["step"] == 0
        obs, reward, term, trunc, info = env.step(-1)
        assert info["fire_affected_cells"] > 0 and info["step"] == 1
        assert reward == pytest.approx(-10*np.count_nonzero(env.legacy.state > 0)/400)
        action = int(np.flatnonzero(env.legacy.state.ravel() > 0)[0])
        env.step(action)
        assert env.legacy.state.ravel()[action] == -1
        assert not env.action_masks()[action]
        # Upstream permits duplicate actions; only masked policies suppress them.
        env.step(action)
        assert env.legacy.state.ravel()[action] == -1
        process = env.legacy.fire_process.process
        input_folder = env.legacy.helper.tmp_input_folder
    finally:
        env.close()
    from pathlib import Path
    assert process.poll() is not None and not Path(input_folder).exists()


def test_matched_fixed_scenarios_reproduce_original_transition():
    outputs = []
    for _ in range(2):
        env = make_env()
        try:
            env.reset(seed=19)
            obs, reward, _, _, info = env.step(-1)
            outputs.append((obs.copy(), reward, info["fire_affected_cells"]))
        finally:
            env.close()
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1:] == outputs[1][1:]


def test_bounds_rejected_before_original_step():
    env = make_env()
    try:
        env.reset()
        for action in (-2,400,True,1.5):
            with pytest.raises(ValueError):
                env.step(action)
        assert env.legacy.iter == 0
    finally:
        env.close()


def test_original_priority_map_reward_uses_protected_cells():
    env = make_env(map_name="mit_m", ignition=372)
    try:
        env.reset(seed=42)
        _, reward, _, _, info = env.step(-1)
        assert env.legacy.steps_before_sim == 25 and env.legacy.steps_per_action == 5
        assert info["protected_cells"] > 0
        assert reward == pytest.approx(-10*info["protected_affected_cells"]/400)
    finally:
        env.close()


def test_random_ignition_stream_changes_across_resets_and_reproduces():
    sequences = []
    for _ in range(2):
        env = make_env(ignition=None)
        try:
            _, first = env.reset(seed=73)
            _, second = env.reset()
            sequences.append((first["ignition_index"], second["ignition_index"]))
        finally:
            env.close()
    assert sequences[0] == sequences[1]
    assert sequences[0][0] != sequences[0][1]
