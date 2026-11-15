"""Tables and exportable scientific comparison figures from measured outcomes."""
import json
import numpy as np

def report(config, output, rows, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["no_intervention", "random", "greedy", "wind_heuristic", "dqn", "dqn_no_wind"]
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
             "Every DQN action is a simultaneous feasible construction subset. Wind is constant within each world. "
             "Both DQNs use identical wind-driven physics; the unaware variant withholds only wind observations.", "",
             f"Placement mode: `{environment['placement_mode']}`. At most {environment['capacity']} "
             "construction sites are built before one synchronous fire update. Clearing and burning both reduce vegetation.", ""]
    if fixed:
        lines += ["This is a fixed-scenario learning diagnostic. Layout, ignition and wind stay fixed; training, "
                  "validation and test use disjoint spread random fields. It does not measure generalization to "
                  "new ignition sites or winds. A constant wind can be learned implicitly even when its observation "
                  "is withheld, so this run alone cannot establish the value of wind observations.", ""]
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
    for method in ("dqn", "dqn_no_wind"):
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
              "The wind-unaware feed-forward policy does not maintain a belief over hidden wind.", "",
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

