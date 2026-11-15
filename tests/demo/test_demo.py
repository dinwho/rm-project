"""Demo replays expose real feasible cell construction without legacy model fallback."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("demo_engine", Path(__file__).resolve().parents[2]/"demo"/"engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


@pytest.mark.parametrize("budget", [None, 6])
def test_demo_replay_tracks_atomic_allocations_and_serializes(budget):
    data = engine.rollout("random", seed=42, budget=budget)
    json.dumps(data, allow_nan=False)
    assert data["action_count"] == data["config"]["grid_size"]**2 + 1
    for previous, successor in zip(data["frames"], data["frames"][1:]):
        chosen = previous["allocation"]
        assert len(chosen) <= 1
        assert all(previous["eligible"][i] for i in chosen)
        assert successor["t"] == previous["t"] + 1
        assert successor["budget"] == (None if budget is None else previous["budget"] - len(chosen))
        assert successor["resources_used"] == previous["resources_used"] + len(chosen)
        assert all(successor["built"][i] for i in chosen)
    assert data["frames"][-1]["finished"]


def test_demo_rejects_invalid_inputs():
    for kwargs in ({"policy_name": "invented"}, {"seed": -1}, {"budget": 9},
                   {"policy_name": "dqn", "budget": 2}):
        with pytest.raises(ValueError):
            engine.rollout(**kwargs)


def test_demo_saved_q_values_match_the_selected_feasible_allocation():
    if not (engine.RUN/"models"/"dqn"/"seed_11"/"best.pt").exists():
        pytest.skip("Optional saved-policy demo requires continuous_fixed weights")
    data = engine.rollout("dqn", 42)
    json.dumps(data, allow_nan=False)
    assert data["model"]["parameter_count"] > 0
    for frame in data["frames"][:-1]:
        assert frame["q_values"][0]["action"] == frame["action"]
        assert all(frame["eligible"][i] for i in frame["allocation"])
