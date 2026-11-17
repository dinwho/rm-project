# Original Firehose reproduction

Reference: [aidan-curtis/firehose](https://github.com/aidan-curtis/firehose), pinned
commit `e49a52a9a987a1585d25ddca2f711222ca500ffb`. Runtime source and original data
are downloaded into `artifacts/references/firehose`. A writable copy is built in
`artifacts/firehose_runtime`; no C++ physics source changes are applied.

## Preserved implementation

- Original Cell2Fire C++ executable and Sub20x20 map/weather/vegetation data.
- Original Python `FireEnv`, harvest action, reward and repeated-action mask.
- Treatment can permanently prevent ignition and extinguish burning cells.
- Original20-step warm-up and8 simulator steps per policy action.
- Original RGB observation, including its initially empty fire overlay after reset.
- Original `FireSizeReward`: negative positive-state-cell fraction times10.
- Original `PaddedNatureCNN`:8×8/stride4,4×4/stride2,1×1/stride1 convolutions,
  followed by512 features; PPO policy/value heads have64,64 hidden units.
- Published fixed-ignition PPO weights trained for5,013,504 steps, gamma.9.

## Compatibility work

The original project uses old Gym/SB3 APIs. A thin Gymnasium adapter transposes
RGB to CHW, converts reset/step results, adds pipe timeouts and reliably kills
subprocesses on close. Simulator seed123 remains the default. Other seeds are
explicitly passed to the C++ command; reset seeds control random ignition sampling.
No new wind inputs, reward shaping, forced barriers or heuristic policy actions
are inserted into the neural model.

Published ZIP policy tensors are loaded with `weights_only=True`. The loader
verifies archive SHA256, map, dimensions and every tensor. Modern SB3 registers
shared feature extractors under additional aliases; these refer to identical
original tensors. No serialized upstream Python code is executed to load weights.
Original weights can be used only on their compatible map/observation/action setup.

The original random baseline may repeat a cell; PPO masks repeated actions.
Min-L2 and Max-L2 choose the nearest/farthest positive-state cell relative to
ignition, matching the upstream baseline rules. The random seed is fixed for the
recorded example. All five policies share map, ignition and simulator seed.

## Interpretation of original output

C++ outputGrid writes both burning and burned cells as1; harvested cells are-1.
Thus original reward and the red overlay conflate burning and burned cells.
The demo calls this **fire-affected**. It cannot establish current active fire,
unique vegetation loss or containment from the red count alone. Treating an
already affected cell can reduce that displayed count. The initial observation
also omits warm-up fire, exactly as in upstream reset; this is retained deliberately.

The included runs show the published checkpoint working on one scenario. They do
not reproduce the paper's complete results, random-ignition comparisons or MIT
priority-region experiment. The MIT maps/reward masks remain available in upstream
source, but the four published checkpoints do not include an MIT-trained model.
The user removed the requirement to recreate those letters.

## Training and validation

`firehose train --output artifacts/my-training --steps 5000000` trains a fresh
Maskable PPO CNN with original defaults (2048 rollout steps, batch64,10 epochs,
gamma.9,learning rate.0003). `--random-ignition` samples ignition locations.
Training rounds requested steps to complete rollouts; the actual count is stored.
This command does not overwrite the published model or demo. Changing maps,
rewards or observations requires retraining. The demo uses published weights;
a32-step test only verifies that fresh training and checkpoint loading work.

Tests exercise real simulator transitions, treatment persistence, original masks,
priority reward handling, reproducibility, checkpoint tensors and a fresh optimizer
run. Browser checks cover live/offline playback and mobile layout. Measured example
outcomes are in [results](../results/README.md). No optimality or exact statistical
replication is claimed.
