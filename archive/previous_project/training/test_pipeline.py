import json
import csv
import pytest

from wildfire_crmab.cli import run
from wildfire_crmab.config import load_config
from wildfire_crmab.evaluation.compare import evaluate


@pytest.mark.parametrize("budget", [None, 2])
def test_training_checkpoint_selection_and_matched_evaluation(tmp_path, budget):
    config = load_config("configs/fixed.json" if budget is None else "configs/smoke.json")
    config["environment"]["max_steps"] = 4
    config["training"].update(steps=16, seeds=[11], batch_size=2, learning_starts=2, train_every=2,
                              validation_every=8, validation_worlds=1, budgets=[budget])
    config["evaluation"].update(worlds=2, bootstrap_samples=20)
    folder = tmp_path/"run"
    run(config, folder)
    for variant in ("dqn", "dqn_no_wind"):
        summary = json.loads((folder/"models"/variant/"seed_11"/"training_summary.json").read_text())
        assert summary["optimizer_updates"] == 8
        assert summary["best_step"] in [8, 16]
        assert summary["validation_improvement_from_initial"] == pytest.approx(
            summary["best_validation_return"] - summary["initial_validation_return"])
        model_folder = folder/"models"/variant/"seed_11"
        assert (model_folder/"initial.pt").is_file()
        with (model_folder/"validation.csv").open() as handle:
            assert [int(row["step"]) for row in csv.DictReader(handle)] == [0, 8, 16]
    assert (folder/"RESULTS.md").is_file()
    if budget is None:
        assert "fixed-scenario learning diagnostic" in (folder/"RESULTS.md").read_text()
    assert (folder/"source"/"src"/"wildfire_crmab"/"environment"/"arms.py").is_file()
    rows = evaluate(config, folder)
    assert len(rows) == 12
    assert all(r["invalid_actions"] == 0 for r in rows)
    assert all(r["budget"] == budget for r in rows)
    assert all(r["multi_construction_intervals"] == 0 for r in rows)


def test_single_dqn_variant_writes_aggregated_learning_curve(tmp_path):
    config = load_config("configs/firehose.json")
    config["environment"]["max_steps"] = 4
    config["training"].update(steps=8, seeds=[11, 22], batch_size=2, learning_starts=2,
                              train_every=2, validation_every=4, validation_worlds=1,
                              variants=["dqn"])
    config["evaluation"].update(worlds=2, bootstrap_samples=20)
    folder = tmp_path/"firehose"
    run(config, folder)
    assert (folder/"models"/"dqn"/"seed_11"/"best.pt").is_file()
    assert not (folder/"models"/"dqn_no_wind").exists()
    with (folder/"evaluation"/"learning_curve.csv").open() as handle:
        curve = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in curve] == [0, 4, 8]
    assert all(int(row["n_seeds"]) == 2 for row in curve)
    assert (folder/"learning_curve.png").is_file()
    report = (folder/"RESULTS.md").read_text()
    assert "dqn_no_wind" not in report
    assert "bootstrap over those seeds" in report
