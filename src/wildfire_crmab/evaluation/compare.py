"""Compare frozen validation-selected checkpoints and matched baselines."""
from dataclasses import asdict
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.baselines import make_baseline
from wildfire_crmab.agents.checkpoints import catalogue_metadata
from wildfire_crmab.config import budget_scale
from wildfire_crmab.utils.io import write_json, write_csv
from .scenarios import scenario_suite
from .runner import run_episode
from .statistics import summarize, paired_differences
from .report import report

def evaluate(config, output):
    t = config["training"]
    scale = budget_scale(config)
    suite = scenario_suite("test", config["evaluation"]["worlds"], t["budgets"])
    write_json(output/"evaluation"/"scenarios.json", [asdict(s) for s in suite])
    rows = []
    for method in ("no_intervention", "random", "greedy", "wind_heuristic"):
        env = WildfireEnv(EnvironmentConfig(**config["environment"]), method == "wind_heuristic", scale)
        for scenario in suite:
            policy = make_baseline(method, env.config, seed=scenario.seed+123)
            rows.append(dict(method=method, training_seed=-1, **run_episode(env, policy, scenario, t["gamma"])))
    for method in ("dqn", "dqn_no_wind"):
        for seed in t["seeds"]:
            agent, metadata = DQNAgent.load(output/"models"/method/f"seed_{seed}"/"best.pt")
            env = WildfireEnv(EnvironmentConfig(**config["environment"]), method == "dqn", scale)
            expected = catalogue_metadata(env)
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("Checkpoint environment/allocation/observation mismatch")
            for scenario in suite:
                rows.append(dict(method=method, training_seed=seed, **run_episode(env, agent, scenario, t["gamma"])))
    samples = config["evaluation"]["bootstrap_samples"]
    summary = summarize(rows, samples)
    paired = paired_differences(rows, samples)
    write_csv(output/"evaluation"/"episodes.csv", rows)
    write_csv(output/"evaluation"/"summary.csv", summary)
    write_csv(output/"evaluation"/"paired_differences.csv", paired)
    report(config, output, rows, summary, paired)
    return rows

