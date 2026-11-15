"""Validate experiment configuration before starting a run."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
from .environment.config import EnvironmentConfig

def load_config(path):
    data = json.loads(Path(path).read_text())
    if set(data) != {"name", "environment", "training", "evaluation"}:
        raise ValueError("Expected name/environment/training/evaluation configuration")
    data["environment"] = asdict(EnvironmentConfig(**data["environment"]))
    t, e = data["training"], data["evaluation"]
    expected = {"steps", "seeds", "budgets", "batch_size", "replay_capacity", "learning_starts",
                "train_every", "target_every", "validation_every", "validation_worlds", "gamma", "learning_rate"}
    if set(t) != expected or set(e) != {"worlds", "bootstrap_samples"}:
        raise ValueError("Unknown or missing experiment settings")
    for key in expected - {"seeds", "budgets", "gamma", "learning_rate", "learning_starts"}:
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seeds", "budgets"):
        if not t[key] or len(set(t[key])) != len(t[key]) or any(type(v) is not int or v < 0 for v in t[key]):
            raise ValueError(f"{key} must be distinct nonnegative integers")
    if not 0 < t["gamma"] <= 1 or not np.isfinite(t["learning_rate"]) or t["learning_rate"] <= 0:
        raise ValueError("Invalid discount or learning rate")
    if type(t["learning_starts"]) is not int or t["learning_starts"] < 0 or t["batch_size"] > t["replay_capacity"]:
        raise ValueError("Invalid replay settings")
    if any(type(v) is not int or v < 1 for v in e.values()):
        raise ValueError("Positive evaluation counts required")
    return data

