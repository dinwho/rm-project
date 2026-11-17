# Measured CRMAB firebreak allocation results

Run: **cells_fixed_scenario**. 3 learning seeds; 10,000 physical transitions per seed and DQN variant; 32 held-out spread realizations of the same fixed world; episode budgets: unlimited.

Every DQN action is a simultaneous feasible construction subset. Wind is constant within each world. Both DQNs use identical wind-driven physics; the unaware variant withholds only wind observations.

Placement mode: `cells`. At most 1 construction sites are built before one synchronous fire update. Clearing and burning both reduce vegetation.

This is a fixed-scenario learning diagnostic. Layout, ignition and wind stay fixed; training, validation and test use disjoint spread random fields. It does not measure generalization to new ignition sites or winds. A constant wind can be learned implicitly even when its observation is withheld, so this run alone cannot establish the value of wind observations.

| Method | Discounted return (95% interval) | Burned | Burned or cleared | Asset survival | Resources used |
| --- | --- | --- | --- | --- | --- |
| no_intervention | -0.860 [-0.898, -0.799] | 0.943 | 0.943 | N/A | 0.00 |
| random | -0.816 [-0.892, -0.713] | 0.667 | 0.892 | N/A | 14.44 |
| greedy | -0.445 [-0.581, -0.320] | 0.341 | 0.494 | N/A | 9.78 |
| wind_heuristic | -0.289 [-0.416, -0.173] | 0.209 | 0.326 | N/A | 7.47 |
| dqn | -0.634 [-0.771, -0.472] | 0.515 | 0.699 | N/A | 11.79 |
| dqn_no_wind | -0.605 [-0.717, -0.481] | 0.498 | 0.671 | N/A | 11.04 |

Paired discounted-return differences (positive favours the first method):

- dqn minus dqn_no_wind: -0.029 [-0.190, 0.142].
- dqn minus greedy: -0.189 [-0.373, 0.003].
- dqn minus no_intervention: 0.226 [0.104, 0.368].
- dqn minus random: 0.182 [0.036, 0.343].
- dqn minus wind_heuristic: -0.344 [-0.519, -0.172].

Learning diagnostic on validation fields (positive change means improvement from the untrained network):

| Variant | Seed | Initial return | Selected return | Change |
| --- | --- | --- | --- | --- |
| dqn | 11 | -0.791 | -0.566 | +0.225 |
| dqn | 22 | -0.780 | -0.516 | +0.264 |
| dqn | 33 | -0.683 | -0.715 | -0.032 |
| dqn_no_wind | 11 | -0.811 | -0.369 | +0.442 |
| dqn_no_wind | 22 | -0.785 | -0.576 | +0.209 |
| dqn_no_wind | 33 | -0.684 | -0.443 | +0.241 |

The initial checkpoint is recorded separately. Checkpoint selection uses the best validation return among trained checkpoints; test results above are never used to select a checkpoint. Validation improvement is a diagnostic, not an unbiased estimate of generalization.

Intervals use a crossed bootstrap of training seeds and test worlds, keeping repeated budgets together. Baselines have one policy per world and are not replicated across artificial learning seeds. The same exogenous random fields are used across methods. Intervals are descriptive and unadjusted for multiple comparisons.

Observed invalid allocations: 0. Intervals constructing multiple sites: 0. Time-limited episodes: 0/320.

Short diagnostic/pilot results establish execution and preliminary comparisons, not convergence or DQN superiority. These synthetic transitions are uncalibrated. Connected placements are possible in cell mode, but one construction per interval can still be too slow to contain a spreading fire. The coupled CRMAB extension has no inherited independence or indexability guarantee. The wind-unaware feed-forward policy does not maintain a belief over hidden wind.

Raw outcomes and per-budget/per-wind summaries are in `evaluation/`. Learning logs and selected checkpoints are in `models/`. `manifest.json` records source hashes and package versions.
