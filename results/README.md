# Recorded original-Firehose example

All five runs use Sub20x20, fixed one-based ignition372 and simulator seed123.
They use the original20-step warm-up and8 Cell2Fire steps per decision.
The neural policy uses the authors' `20x20-fixed-ppo-maskable.zip` checkpoint,
trained for5,013,504 steps. Its weights were not trained in this workspace.

| Policy | Original cumulative reward (higher better) | Final fire-affected state cells | Treated cells |
| --- | ---: | ---: | ---: |
| Published Maskable PPO | -102.400 |96|58|
| No intervention | -224.075 |217|0|
| Random | -201.750 |186|52|
| Min-L2 | -153.975 |124|57|
| Max-L2 | -176.525 |157|57|

This is one recorded scenario, not a reproduction of the paper's statistical
comparison. Original C++ output marks burning and burned cells with the same value1.
Consequently the red-cell count is called fire-affected, not active burning.
Treatment of an already affected cell can remove it from that displayed count;
these numbers do not establish exact preserved vegetation or a containment rate.

Raw frames/actions/rewards: `../firehose_demo/data.json`.
Compact metrics and provenance: `recorded_metrics.json`.
The checkpoint SHA256 and pinned source commit are included in those files.

## Additional original map: Sub40x40

Fixed ignition909, seed123,25 warm-up steps and5 simulator steps per decision.
Matching published checkpoint: `40x40-fixed-ppo-maskable.zip` (10,027,008 training steps).

| Policy | Return | Fire-affected | Treated |
| --- | ---: | ---: | ---: |
| Published Maskable PPO | -56.519 | 149 | 79 |
| No Intervention | -126.837 | 394 | 0 |
| Random | -122.981 | 376 | 77 |
| Min L2 | -88.256 | 232 | 78 |
| Max L2 | -78.469 | 218 | 78 |
