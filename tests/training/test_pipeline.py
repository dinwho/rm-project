import json

from wildfire_crmab.cli import run
from wildfire_crmab.config import load_config
from wildfire_crmab.evaluation.compare import evaluate


def test_training_checkpoint_selection_and_matched_evaluation(tmp_path):
    config = load_config("configs/smoke.json")
    config["environment"]["max_steps"] = 4
    config["training"].update(steps=16, batch_size=2, learning_starts=2, train_every=2,
                              validation_every=8, validation_worlds=1, budgets=[2])
    config["evaluation"].update(worlds=2, bootstrap_samples=20)
    folder = tmp_path/"run"
    run(config, folder)
    for variant in ("dqn", "dqn_no_wind"):
        summary = json.loads((folder/"models"/variant/"seed_11"/"training_summary.json").read_text())
        assert summary["optimizer_updates"] == 8
        assert summary["best_step"] in [8, 16]
    assert (folder/"RESULTS.md").is_file()
    assert (folder/"source"/"src"/"wildfire_crmab"/"environment"/"arms.py").is_file()
    rows = evaluate(config, folder)
    assert len(rows) == 12
    assert all(r["invalid_actions"] == 0 for r in rows)
