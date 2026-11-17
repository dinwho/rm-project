"""Command entry points and reproducible experiment orchestration."""
import argparse
from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import platform
import torch
from .config import load_config
from .training.train import train_one
from .evaluation.compare import evaluate
from .evaluation.scenarios import scenario_suite
from .utils.io import write_json

def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", config)
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root/"src").rglob("*.py")) + sorted((root/"docs").glob("*.md"))
    write_json(output/"manifest.json", dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(),
        packages={p: version(p) for p in ("numpy", "torch", "gymnasium", "matplotlib")},
        source_hashes={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        source_snapshot="source/", seed_split="train < 1e9; validation 1e9+; test 2e9+",
        checkpoint_selection="discounted validation return", wind_physics=config["environment"]["wind_physics"]))
    for path in sources:
        snapshot = output/"source"/path.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    write_json(output/"validation_scenarios.json", [asdict(s) for s in scenario_suite(
        "validation", config["training"]["validation_worlds"], config["training"]["budgets"])])
    for variant in config["training"]["variants"]:
        for seed in config["training"]["seeds"]:
            train_one(config, output, seed, variant == "dqn")
    evaluate(config, output)
    print(f"Completed: {output/'RESULTS.md'}", flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run")
    train.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    train.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    test = commands.add_parser("evaluate")
    test.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "run":
        run(load_config(args.config), args.output)
    else:
        evaluate(load_config(args.run_dir/"config.json"), args.run_dir)

