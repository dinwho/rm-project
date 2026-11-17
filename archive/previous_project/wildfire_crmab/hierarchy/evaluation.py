"""Matched allocation comparisons with exactly the same frozen local constructor."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from wildfire_crmab.utils.io import write_json, write_csv
from .environment import GlobalWildfireEnv
from .baselines import AllocationBaseline
from .dqn import GlobalDQN, checkpoint_metadata
from .training import world_seeds, run_global_episode

METRICS = ('discounted_return', 'episode_return', 'burned_cells', 'cleared_cells',
           'affected_fraction', 'contained_fraction', 'team_utilization', 'construction_utilization', 'truncated')


def summarize(rows, samples=500, seed=2026):
    """Crossed learning-seed/world bootstrap, separately for each team count."""
    if samples < 1:
        raise ValueError('Positive bootstrap sample count required')
    rng = np.random.default_rng(seed)
    output = []
    for method in sorted({r['method'] for r in rows}):
        for capacity in sorted({r['team_capacity'] for r in rows}):
            records = [r for r in rows if r['method'] == method and r['team_capacity'] == capacity]
            seeds = sorted({r['training_seed'] for r in records})
            worlds = sorted({r['world_seed'] for r in records})
            by_key = {(r['training_seed'], r['world_seed']): r for r in records}
            if len(by_key) != len(records) or len(by_key) != len(seeds)*len(worlds):
                raise ValueError('Incomplete or duplicated crossed comparison')
            sw = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)), samples)/len(seeds)
            ww = rng.multinomial(len(worlds), np.full(len(worlds), 1/len(worlds)), samples)/len(worlds)
            for metric in METRICS:
                matrix = np.array([[by_key[s,w][metric] for w in worlds] for s in seeds])
                if not np.isfinite(matrix).all():
                    raise ValueError('Nonfinite global evaluation metric')
                estimates = np.einsum('bi,ij,bj->b', sw, matrix, ww)
                low, high = np.quantile(estimates, [.025, .975])
                output.append(dict(method=method, team_capacity=capacity, metric=metric,
                    mean=float(matrix.mean()), ci_low=float(low), ci_high=float(high),
                    learning_seeds=len(seeds), worlds=len(worlds)))
    return output


def paired_comparisons(rows, samples=500):
    refs = {(r['method'], r['scenario_id']): r for r in rows if r['method'] != 'dqn'}
    differences = []
    for row in rows:
        if row['method'] == 'dqn':
            for method in ('no_allocation', 'random', 'severity', 'risk'):
                other = refs[method, row['scenario_id']]
                diff = {**row, 'method': 'dqn minus '+method}
                diff.update({metric: row[metric]-other[metric] for metric in METRICS})
                differences.append(diff)
    return summarize(differences, samples)


def evaluate_global(output, config, local_policy, checkpoints, *, worlds=24,
                    team_counts=(0, 1, 2, 3), incident_count=4, bootstrap_samples=500, gamma=.99):
    if not checkpoints or not team_counts or len(set(team_counts)) != len(team_counts):
        raise ValueError('Provide checkpoints and distinct team counts')
    output = Path(output)
    folder = output/'global_evaluation'
    env = GlobalWildfireEnv(config, local_policy, incident_count, team_counts[0])
    metadata = checkpoint_metadata(env, local_policy)
    suite = world_seeds('test', worlds)
    write_json(folder/'protocol.json', dict(world_seeds=suite, team_counts=list(team_counts),
               incident_count=incident_count, gamma=gamma, **metadata))
    rows = []
    for method in ('no_allocation', 'random', 'severity', 'risk'):
        for world in suite:
            for capacity in team_counts:
                policy = AllocationBaseline(method, env.subsets, config, seed=world+123)
                rows.append(dict(method=method, training_seed=-1,
                    local_checkpoint_sha256=metadata['local_checkpoint_sha256'],
                    **run_global_episode(env, policy, world, capacity, gamma)))
    seen = set()
    for path in checkpoints:
        policy, saved = GlobalDQN.load(path, metadata)
        if policy.seed in seen:
            raise ValueError('Duplicate global training seed')
        if policy.gamma != gamma:
            raise ValueError('Evaluation gamma must match training objective')
        seen.add(policy.seed)
        for world in suite:
            for capacity in team_counts:
                rows.append(dict(method='dqn', training_seed=policy.seed,
                    local_checkpoint_sha256=metadata['local_checkpoint_sha256'],
                    **run_global_episode(env, policy, world, capacity, gamma)))
    summary = summarize(rows, bootstrap_samples)
    paired = paired_comparisons(rows, bootstrap_samples)
    write_csv(folder/'episodes.csv', rows)
    write_csv(folder/'summary.csv', summary)
    write_csv(folder/'paired_differences.csv', paired)
    lines = ['# Global team-allocation results', '',
        f'{len(seen)} DQN learning seeds, {worlds} held-out independent multi-incident worlds per team count. '
        f'Team counts: {list(team_counts)}. Every strategy uses exactly the same frozen local policy '
        f'`{metadata["local_checkpoint_sha256"]}`.', '',
        'The global DQN is trained with observed team capacity sampled across its configured counts. '
        'The local construction weights remain frozen. These comparisons isolate allocation policy, '
        'not a change of local constructor. Local placement results must be reported separately.', '',
        '| Teams | Allocator | Return (95% interval) | Burned cells | Cleared cells | Contained incidents | Team utilization |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    lookup = {(r['method'], r['team_capacity'], r['metric']): r for r in summary}
    for k in team_counts:
        for method in ('no_allocation', 'random', 'severity', 'risk', 'dqn'):
            def cell(metric): return lookup[method,k,metric]
            ret = cell('discounted_return')
            lines.append(f'| {k} | {method} | {ret["mean"]:.3f} [{ret["ci_low"]:.3f}, {ret["ci_high"]:.3f}] | '
                         f'{cell("burned_cells")["mean"]:.2f} | {cell("cleared_cells")["mean"]:.2f} | '
                         f'{cell("contained_fraction")["mean"]:.1%} | {cell("team_utilization")["mean"]:.1%} |')
    lines += ['', 'Containment means extinction with at least one unburned, uncleared vegetation cell remaining; '
              'it is an operational outcome, not a proven closed perimeter. Full burnout is not containment. '
              'Utilization counts assigned team-intervals divided by K × elapsed intervals; reported as zero for K=0. '
              'Construction utilization separately counts actual cells built, since the frozen policy may wait.', '',
              f'Invalid allocations: {sum(r["invalid_actions"] for r in rows)}; '
              f'horizon endings: {sum(r["truncated"] for r in rows)}/{len(rows)}.', '',
              'Confidence intervals resample worlds and learning seeds; baseline policies are not duplicated '
              'as artificial training seeds. Paired DQN-minus-baseline differences are in '
              '`global_evaluation/paired_differences.csv`. Higher return is better; lower damage is better. '
              'The intervals are descriptive, not corrected for multiple comparisons.', '',
              'This is a short synthetic research diagnostic, not an optimality claim. Global performance '
              'is conditional on this one local checkpoint and its training seed. No uncertainty over '
              'independently retrained local constructors is estimated.']
    (output/'GLOBAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return rows
