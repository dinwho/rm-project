# Measured CRMAB firebreak allocation results

Run: **pilot**. 3 learning seeds; 10,000 physical transitions per seed and DQN variant; 24 held-out worlds, each at budgets [2, 4, 6].

Every DQN action is a simultaneous feasible construction subset. Wind is constant within each world. Both DQNs use identical wind-driven physics; the unaware variant withholds only wind observations.

| Method | Discounted return (95% interval) | Burned | Burned or cleared | Asset survival | Budget used |
| --- | --- | --- | --- | --- | --- |
| no_intervention | -8.284 [-9.449, -6.833] | 0.833 | 0.833 | 0.125 | 0.00 |
| random | -7.356 [-8.806, -5.774] | 0.667 | 0.751 | 0.222 | 3.94 |
| greedy | -5.027 [-6.893, -3.179] | 0.495 | 0.557 | 0.486 | 2.90 |
| wind_heuristic | -5.031 [-6.860, -3.217] | 0.494 | 0.556 | 0.486 | 2.90 |
| dqn | -6.951 [-8.444, -5.337] | 0.644 | 0.730 | 0.264 | 4.00 |
| dqn_no_wind | -7.095 [-8.539, -5.328] | 0.651 | 0.736 | 0.250 | 3.99 |

Paired discounted-return differences (positive favours the first method):

- dqn minus dqn_no_wind: 0.144 [-0.668, 0.922].
- dqn minus greedy: -1.924 [-3.338, -0.716].
- dqn minus no_intervention: 1.333 [0.466, 2.295].
- dqn minus random: 0.405 [-0.075, 1.035].
- dqn minus wind_heuristic: -1.920 [-3.261, -0.714].

Intervals use a crossed bootstrap of training seeds and test worlds, keeping repeated budgets together. Baselines have one policy per world and are not replicated across artificial learning seeds. The same exogenous random fields are used across methods. Intervals are descriptive and unadjusted for multiple comparisons.

Observed invalid allocations: 0. Intervals constructing multiple segments: 996. Time-limited episodes: 108/720.

Smoke/pilot results establish execution and preliminary comparisons, not convergence or DQN superiority. These synthetic transitions are uncalibrated. Sparse candidate geometry limits available firebreaks. The coupled CRMAB extension has no inherited independence or indexability guarantee. The wind-unaware feed-forward policy does not maintain a belief over hidden wind.

Raw outcomes and per-budget/per-wind summaries are in `evaluation/`. Learning logs and selected checkpoints are in `models/`. `manifest.json` records source hashes and package versions.

This compact snapshot omits weights and source copies; see `export_manifest.json` for the complete local run and checkpoint hashes.
