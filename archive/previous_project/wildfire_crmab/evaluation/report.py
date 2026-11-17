"""Tables and exportable scientific comparison figures from measured outcomes."""
import csv
import json
import numpy as np
from wildfire_crmab.utils.io import write_csv


def learning_curve(config, output):
    training = config["training"]
    curves = []
    for method in training["variants"]:
        for seed in training["seeds"]:
            path = output/"models"/method/f"seed_{seed}"/"validation.csv"
            if not path.is_file():
                continue
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    curves.append(dict(method=method, seed=seed, step=int(row["step"]),
                                       discounted_return=float(row["discounted_return"])))
    if not curves:
        return []
    grouped = {}
    for method in training["variants"]:
        method_rows = [row for row in curves if row["method"] == method]
        steps = sorted({row["step"] for row in method_rows})
        seeds = sorted({row["seed"] for row in method_rows})
        matrix = np.array([[next(row["discounted_return"] for row in method_rows
                                 if row["seed"] == seed and row["step"] == step)
                            for seed in seeds] for step in steps])
        if matrix.shape[1] != len(training["seeds"]):
            raise ValueError(f"Incomplete learning curves for {method}")
        rng = np.random.default_rng(2026)
        draws = rng.integers(0, matrix.shape[1], size=(5000, matrix.shape[1]))
        bootstrap_means = matrix[np.arange(matrix.shape[0])[:, None, None], draws[None, :, :]].mean(axis=2)
        for index, step in enumerate(steps):
            low, high = np.quantile(bootstrap_means[index], [.025, .975])
            grouped[method, step] = dict(method=method, step=step,
                                         mean=float(matrix[index].mean()),
                                         ci_low=float(low), ci_high=float(high),
                                         n_seeds=matrix.shape[1])
    result = list(grouped.values())
    write_csv(output/"evaluation"/"learning_curve.csv", result)
    return result

def report(config, output, rows, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["no_intervention", "random", "greedy", "wind_heuristic", *config["training"]["variants"]]
    learned_methods = config["training"]["variants"]
    curves = learning_curve(config, output)
    def cell(method, metric):
        return next(r for r in summary if r["method"] == method and r["group"] == "all" and r["metric"] == metric)
    environment = config["environment"]
    fixed = all(environment.get(key) is not None for key in
                ("fixed_layout_seed", "fixed_ignition", "fixed_wind"))
    budgets = ", ".join("unlimited" if b is None else str(b) for b in config["training"]["budgets"])
    unit = "spread realizations of the same fixed world" if fixed else "worlds"
    lines = ["# Measured CRMAB firebreak allocation results", "",
             f"Run: **{config['name']}**. {len(config['training']['seeds'])} learning seeds; "
             f"{config['training']['steps']:,} physical transitions per seed and DQN variant; "
             f"{config['evaluation']['worlds']} held-out {unit}; episode budgets: {budgets}.", "",
             "Each DQN action is a simultaneous feasible construction subset. Wind is constant within each world. "
             + ("Wind-driven physics is enabled. The wind-aware policy observes the world wind."
              if environment["wind_physics"] else "Wind physics is disabled: this is a separate physics ablation."), "",
             f"Placement mode: `{environment['placement_mode']}`. At most {environment['capacity']} "
             "construction sites are built before one synchronous fire update. Clearing and burning both reduce vegetation.", ""]
    if fixed:
        lines += ["This is a fixed-scenario learning diagnostic. Layout, ignition and wind stay fixed; training, "
                  "validation and test use disjoint spread random fields. It does not measure generalization to "
                  "new ignition sites or winds.", ""]
    lines += ["| Method | Discounted return (95% interval) | Burned | Burned or cleared | Asset survival | Resources used |",
             "| --- | --- | --- | --- | --- | --- |"]
    for method in methods:
        r = cell(method, "discounted_return")
        others = [cell(method, m)["mean"] for m in ("burned_fraction", "affected_fraction", "settlement_survived", "budget_used")]
        asset_survival = f"{others[2]:.3f}" if environment["include_settlement"] else "N/A"
        lines.append(f"| {method} | {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"{others[0]:.3f} | {others[1]:.3f} | {asset_survival} | {others[3]:.2f} |")
    lines += ["", "Paired discounted-return differences (positive favours the first method):", ""]
    for row in paired:
        if row["group"] == "all" and row["metric"] == "discounted_return" and row["method"].startswith("dqn minus"):
            lines.append(f"- {row['method']}: {row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}].")
    lines += ["", "Learning diagnostic on validation fields (positive change means improvement from the untrained network):", "",
              "| Variant | Seed | Initial return | Selected return | Change |", "| --- | --- | --- | --- | --- |"]
    for method in learned_methods:
        for seed in config["training"]["seeds"]:
            path = output/"models"/method/f"seed_{seed}"/"training_summary.json"
            if path.is_file():
                training = json.loads(path.read_text())
                if "initial_validation_return" in training:
                    lines.append(f"| {method} | {seed} | {training['initial_validation_return']:.3f} | "
                                 f"{training['best_validation_return']:.3f} | "
                                 f"{training['validation_improvement_from_initial']:+.3f} |")
    lines += ["", "The initial checkpoint is recorded separately. Checkpoint selection uses the best validation "
              "return among trained checkpoints; test results above are never used to select a checkpoint. "
              "Validation improvement is a diagnostic, not an unbiased estimate of generalization."]
    if curves:
        lines += ["", "Validation learning curves report the mean across independent training seeds; brackets are "
                  "percentile intervals from a bootstrap over those seeds (descriptive with few seeds). See "
                  "`evaluation/learning_curve.csv` and `learning_curve.png`."]
    lines += ["", "Intervals use a crossed bootstrap of training seeds and test worlds, keeping repeated budgets together. "
              "Baselines have one policy per world and are not replicated across artificial learning seeds. "
              "The same exogenous random fields are used across methods. Intervals are descriptive and unadjusted for multiple comparisons.", "",
              f"Observed invalid allocations: {sum(r['invalid_actions'] for r in rows)}. "
              f"Intervals constructing multiple sites: {sum(r['multi_construction_intervals'] for r in rows)}. "
              f"Time-limited episodes: {sum(r['truncated'] for r in rows)}/{len(rows)}.", "",
              "Short diagnostic/pilot results establish execution and preliminary comparisons, not convergence or DQN superiority. "
              "These synthetic transitions are uncalibrated. Connected placements are possible in cell mode, "
              "but one construction per interval can still be too slow to contain a spreading fire. "
              "The coupled CRMAB extension has no inherited independence or indexability guarantee. "
              "CRMAB here is a spatially coupled mathematical model, not the full Firehose/Cell2Fire simulator.", "",
              "Raw outcomes and per-budget/per-wind summaries are in `evaluation/`. Learning logs and selected "
              "checkpoints are in `models/`. `manifest.json` records source hashes and package versions."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    third_metric = "settlement_survived" if environment["include_settlement"] else "budget_used"
    for ax, metric in zip(axes, ("discounted_return", "affected_fraction", third_metric)):
        values = [cell(m, metric) for m in methods]
        means = np.array([v["mean"] for v in values])
        errors = np.maximum(0, [means-np.array([v["ci_low"] for v in values]), np.array([v["ci_high"] for v in values])-means])
        ax.bar(np.arange(len(methods)), means, yerr=errors, capsize=3)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=65, ha="right")
        ax.set_title(metric.replace("_", " "))
    fig.savefig(output/"comparison.png", dpi=160)
    fig.savefig(output/"comparison.pdf")
    plt.close(fig)
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for method in learned_methods:
            points = [row for row in curves if row["method"] == method]
            steps = np.array([row["step"] for row in points])
            means = np.array([row["mean"] for row in points])
            low = np.array([row["ci_low"] for row in points])
            high = np.array([row["ci_high"] for row in points])
            ax.plot(steps, means, marker="o", label=method)
            ax.fill_between(steps, low, high, alpha=.2)
        ax.set_xlabel("Training fire intervals")
        ax.set_ylabel("Mean validation discounted return")
        ax.set_title("DQN learning across training seeds")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output/"learning_curve.png", dpi=160)
        fig.savefig(output/"learning_curve.pdf")
        plt.close(fig)

