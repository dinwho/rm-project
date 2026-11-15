from dataclasses import asdict
import json

import numpy as np
import pytest

from wildfire_crmab.config import load_config
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.evaluation.scenarios import scenario_suite
from wildfire_crmab.evaluation.statistics import summarize, paired_differences


def test_seed_splits_do_not_overlap_and_worlds_repeat_across_budgets():
    validation = scenario_suite("validation", 3, [2, 4])
    test = scenario_suite("test", 3, [2, 4])
    assert not {s.seed for s in validation} & {s.seed for s in test}
    assert len({s.seed for s in validation}) == 3
    assert len({s.scenario_id for s in validation}) == 6
    assert min(s.seed for s in validation) >= 1_000_000_000


def test_budget_repetitions_are_clustered_not_independent_worlds():
    rows = [dict(method="policy", training_seed=s, world_seed=w, budget=b,
                 condition="weaker", value=w+b) for s in [1, 2] for w in [10, 20] for b in [2, 4]]
    all_row = next(r for r in summarize(rows, 100, metrics=("value",)) if r["group"] == "all")
    assert all_row["n_worlds"] == all_row["n_training_seeds"] == 2
    assert all_row["mean"] == 18
    with pytest.raises(ValueError, match="Incomplete"):
        summarize([r for r in rows if not (r["training_seed"] == 1 and r["world_seed"] == 10)], metrics=("value",))


def test_paired_comparison_uses_same_scenario_and_learning_seed():
    from wildfire_crmab.evaluation.statistics import METRICS
    rows = []
    for method in ("no_intervention", "random", "greedy", "wind_heuristic", "dqn", "dqn_no_wind"):
        for seed in ([11, 22] if method.startswith("dqn") else [-1]):
            for world in [1, 2, 3]:
                value = world + (5 if method == "dqn" else 0)
                rows.append(dict(method=method, training_seed=seed, world_seed=world,
                                 scenario_id=str(world), condition="weaker", budget=2,
                                 **{m: value for m in METRICS}))
    comparisons = paired_differences(rows, 100)
    diff = next(r for r in comparisons if r["method"] == "dqn minus dqn_no_wind" and
                r["group"] == "all" and r["metric"] == "discounted_return")
    assert [diff[k] for k in ("mean", "ci_low", "ci_high")] == pytest.approx([5, 5, 5])


def test_config_roundtrip_and_reject_unknown_keys(tmp_path):
    config = load_config("configs/smoke.json")
    assert config["environment"] == asdict(EnvironmentConfig())
    path = tmp_path/"config.json"
    path.write_text(json.dumps(config))
    assert load_config(path) == config
    config["training"]["typo"] = 3
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)
