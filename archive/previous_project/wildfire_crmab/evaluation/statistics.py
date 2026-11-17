"""Crossed training-seed/world bootstrap; budgets stay within world clusters."""

from collections import defaultdict
import numpy as np

METRICS = ("discounted_return", "burned_fraction", "affected_fraction",
           "settlement_survived", "budget_used", "truncated", "active_fire_cells")


def summarize(rows, samples=1000, seed=2026, metrics=METRICS):
    rng = np.random.default_rng(seed)
    output = []
    for method in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method]
        groups = {"all": subset}
        for budget in sorted({r["budget"] for r in subset}, key=lambda b: (b is None, b or 0)):
            label = "unlimited" if budget is None else str(budget)
            groups[f"budget={label}"] = [r for r in subset if r["budget"] == budget]
        for condition in sorted({r["condition"] for r in subset}):
            groups[f"wind={condition}"] = [r for r in subset if r["condition"] == condition]
        for group, records in groups.items():
            seeds = sorted({r["training_seed"] for r in records})
            worlds = sorted({r["world_seed"] for r in records})
            cells = defaultdict(list)
            for row in records:
                cells[row["training_seed"], row["world_seed"]].append(row)
            # Baselines have one actual policy seed per world, not duplicated DQN seeds.
            sw = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)), samples)/len(seeds)
            ww = rng.multinomial(len(worlds), np.full(len(worlds), 1/len(worlds)), samples)/len(worlds)
            for metric in metrics:
                if any(not cells[s, w] for s in seeds for w in worlds):
                    raise ValueError("Incomplete crossed evaluation matrix")
                matrix = np.array([[np.mean([r[metric] for r in cells[s, w]]) for w in worlds] for s in seeds])
                if not np.isfinite(matrix).all():
                    raise ValueError("Nonfinite evaluation metric")
                estimates = np.einsum("bi,ij,bj->b", sw, matrix, ww)
                low, high = np.quantile(estimates, [.025, .975])
                output.append(dict(method=method, group=group, metric=metric,
                                   mean=float(matrix.mean()), ci_low=float(low), ci_high=float(high),
                                   n_training_seeds=len(seeds), n_worlds=len(worlds)))
    return output


def paired_differences(rows, samples=1000):
    index = {(r["method"], r["training_seed"], r["scenario_id"]): r for r in rows}
    differences = []
    learned_methods = {r["method"] for r in rows if r["method"] in ("dqn", "dqn_no_wind")}
    for row in rows:
        if row["method"] not in learned_methods:
            continue
        refs = ["no_intervention", "random", "greedy", "wind_heuristic"]
        refs.extend(method for method in learned_methods if method != row["method"])
        for ref in refs:
            ref_seed = row["training_seed"] if ref.startswith("dqn") else -1
            other = index[ref, ref_seed, row["scenario_id"]]
            difference = {**row, "method": row["method"] + " minus " + ref}
            for metric in METRICS:
                difference[metric] = row[metric] - other[metric]
            differences.append(difference)
    return summarize(differences, samples, metrics=METRICS)
