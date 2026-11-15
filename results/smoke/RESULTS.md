# Measured CRMAB firebreak allocation results

Run: **smoke**. 1 learning seeds; 600 physical transitions per seed and DQN variant; 6 held-out worlds, each at budgets [2, 4, 6].

Every DQN action is a simultaneous feasible construction subset. Wind is constant within each world. Both DQNs use identical wind-driven physics; the unaware variant withholds only wind observations.

| Method | Discounted return (95% interval) | Burned | Burned or cleared | Asset survival | Budget used |
| --- | --- | --- | --- | --- | --- |
| no_intervention | -7.009 [-9.553, -3.848] | 0.735 | 0.735 | 0.167 | 0.00 |
| random | -5.393 [-8.284, -2.234] | 0.525 | 0.609 | 0.389 | 3.89 |
| greedy | -3.603 [-6.643, -1.054] | 0.403 | 0.466 | 0.611 | 2.94 |
| wind_heuristic | -3.603 [-6.305, -1.024] | 0.403 | 0.466 | 0.611 | 2.94 |
| dqn | -6.467 [-9.307, -3.623] | 0.618 | 0.704 | 0.278 | 4.00 |
| dqn_no_wind | -5.060 [-8.064, -2.064] | 0.495 | 0.581 | 0.444 | 4.00 |

Paired discounted-return differences (positive favours the first method):

- dqn minus dqn_no_wind: -1.407 [-3.238, -0.260].
- dqn minus greedy: -2.864 [-5.707, -0.615].
- dqn minus no_intervention: 0.542 [0.022, 1.401].
- dqn minus random: -1.074 [-2.251, 0.112].
- dqn minus wind_heuristic: -2.864 [-5.292, -0.861].

Intervals use a crossed bootstrap of training seeds and test worlds, keeping repeated budgets together. Baselines have one policy per world and are not replicated across artificial learning seeds. The same exogenous random fields are used across methods. Intervals are descriptive and unadjusted for multiple comparisons.

Observed invalid allocations: 0. Intervals constructing multiple segments: 108. Time-limited episodes: 19/108.

Smoke/pilot results establish execution and preliminary comparisons, not convergence or DQN superiority. These synthetic transitions are uncalibrated. Sparse candidate geometry limits available firebreaks. The coupled CRMAB extension has no inherited independence or indexability guarantee. The wind-unaware feed-forward policy does not maintain a belief over hidden wind.

Raw outcomes and per-budget/per-wind summaries are in `evaluation/`. Learning logs and selected checkpoints are in `models/`. `manifest.json` records source hashes and package versions.

This compact snapshot omits weights and source copies; see `export_manifest.json` for the complete local run and checkpoint hashes.
