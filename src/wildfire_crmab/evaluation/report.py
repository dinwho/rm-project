"""Tables and exportable scientific comparison figures from measured outcomes."""
import numpy as np

def report(config, output, rows, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["no_intervention", "random", "greedy", "wind_heuristic", "dqn", "dqn_no_wind"]
    def cell(method, metric):
        return next(r for r in summary if r["method"] == method and r["group"] == "all" and r["metric"] == metric)
    lines = ["# Measured CRMAB firebreak allocation results", "",
             f"Run: **{config['name']}**. {len(config['training']['seeds'])} learning seeds; "
             f"{config['training']['steps']:,} physical transitions per seed and DQN variant; "
             f"{config['evaluation']['worlds']} held-out worlds, each at budgets {config['training']['budgets']}.", "",
             "Every DQN action is a simultaneous feasible construction subset. Wind is constant within each world. "
             "Both DQNs use identical wind-driven physics; the unaware variant withholds only wind observations.", "",
             "| Method | Discounted return (95% interval) | Burned | Burned or cleared | Asset survival | Budget used |",
             "| --- | --- | --- | --- | --- | --- |"]
    for method in methods:
        r = cell(method, "discounted_return")
        others = [cell(method, m)["mean"] for m in ("burned_fraction", "affected_fraction", "settlement_survived", "budget_used")]
        lines.append(f"| {method} | {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"{others[0]:.3f} | {others[1]:.3f} | {others[2]:.3f} | {others[3]:.2f} |")
    lines += ["", "Paired discounted-return differences (positive favours the first method):", ""]
    for row in paired:
        if row["group"] == "all" and row["metric"] == "discounted_return" and row["method"].startswith("dqn minus"):
            lines.append(f"- {row['method']}: {row['mean']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}].")
    lines += ["", "Intervals use a crossed bootstrap of training seeds and test worlds, keeping repeated budgets together. "
              "Baselines have one policy per world and are not replicated across artificial learning seeds. "
              "The same exogenous random fields are used across methods. Intervals are descriptive and unadjusted for multiple comparisons.", "",
              f"Observed invalid allocations: {sum(r['invalid_actions'] for r in rows)}. "
              f"Intervals constructing multiple segments: {sum(r['multi_construction_intervals'] for r in rows)}. "
              f"Time-limited episodes: {sum(r['truncated'] for r in rows)}/{len(rows)}.", "",
              "Smoke/pilot results establish execution and preliminary comparisons, not convergence or DQN superiority. "
              "These synthetic transitions are uncalibrated. Sparse candidate geometry limits available firebreaks. "
              "The coupled CRMAB extension has no inherited independence or indexability guarantee. "
              "The wind-unaware feed-forward policy does not maintain a belief over hidden wind.", "",
              "Raw outcomes and per-budget/per-wind summaries are in `evaluation/`. Learning logs and selected "
              "checkpoints are in `models/`. `manifest.json` records source hashes and package versions."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for ax, metric in zip(axes, ("discounted_return", "affected_fraction", "settlement_survived")):
        values = [cell(m, metric) for m in methods]
        means = np.array([v["mean"] for v in values])
        errors = np.maximum(0, [means-np.array([v["ci_low"] for v in values]), np.array([v["ci_high"] for v in values])-means])
        ax.bar(np.arange(len(methods)), means, yerr=errors, capsize=3)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=65, ha="right")
        ax.set_title(metric.replace("_", " "))
    fig.savefig(output/"comparison.png", dpi=160)
    fig.savefig(output/"comparison.pdf")
    plt.close(fig)

