# Firehose implementation audit and local-policy adaptation

Reference inspected: [aidan-curtis/firehose](https://github.com/aidan-curtis/firehose),
commit **e49a52a9a987a1585d25ddca2f711222ca500ffb**. The read-only checkout is
`artifacts/references/firehose/`. This is a source-level inspection, not an assertion
that the original Cell2Fire executable was built or reproduced here. No upstream
policy weights are used by this project.

## What the actual repository implements

- [CNN source](https://github.com/aidan-curtis/firehose/blob/e49a52a9a987a1585d25ddca2f711222ca500ffb/cell2fire/firehose/models.py):
  `PaddedNatureCNN` receives RGB images. Its convolutions are 32 channels/kernel8/stride4,
  64/kernel4/stride2, and 64/kernel1/stride1, followed by a 512-feature dense layer.
  Despite its name these convolution definitions have no explicit padding. Applying
  these kernels directly to a6×6 grid is invalid.
- [Training setup](https://github.com/aidan-curtis/firehose/blob/e49a52a9a987a1585d25ddca2f711222ca500ffb/cell2fire/rl_experiment_vectorized.py):
  the CNN uses `forest_rgb` observations and the custom feature extractor. Algorithm
  selection resolves `ppo-maskable` to `sb3_contrib.MaskablePPO` through the evaluation
  module. Training supports vectorized environments. The repository requirements
  point at William Shen's sb3-contrib fork, not a reproducible pinned release.
- [Environment](https://github.com/aidan-curtis/firehose/blob/e49a52a9a987a1585d25ddca2f711222ca500ffb/cell2fire/gym_env.py):
  discrete actions use flattened row-major cell IDs. `action_masks()` excludes
  previously selected cells; it is not our stricter vegetation-only eligibility.
  `step()` applies treatment to the Cell2Fire process before obtaining its next state.
  Treatment also allows active suppression; our constructor deliberately does not.
  The code supports larger action footprints as well as single-cell actions.
- [Timing presets](https://github.com/aidan-curtis/firehose/blob/e49a52a9a987a1585d25ddca2f711222ca500ffb/cell2fire/evaluate_model.py):
  Sub20x20 has20 initial simulator steps before intervention and8 simulator steps
  per policy action; Sub40x40 uses25 and5. “One action per decision” must not be
  confused with “one Cell2Fire physics step.” Our hierarchy explicitly uses one
  construction opportunity per assigned team per shared simulation interval.
- [Reward source](https://github.com/aidan-curtis/firehose/blob/e49a52a9a987a1585d25ddca2f711222ca500ffb/cell2fire/firehose/rewards.py):
  `FireSizeReward` is negative current burning-cell fraction times10.
  `CellsBurnedReward` and a distance-penalized alternative also exist. None is our
  exact incremental burned-plus-cleared vegetation objective.

## Pretrained compatibility inspection

The four original ZIP archives were inspected using `zipfile` and JSON `data`
only; serialized Python objects were not unpickled or executed. Their human-readable
metadata reports:

| Archive | Observation | Actions | Recorded training steps | Gamma |
| --- | --- | --- | --- | --- |
|20x20-fixed-ppo-maskable.zip|uint8 RGB3×20×20|400|5,013,504|.9|
|20x20-random-20m-ppo-maskable.zip|uint8 RGB3×20×20|400|20,021,248|.9|
|40x40-fixed-ppo-maskable.zip|uint8 RGB3×40×40|1600|10,027,008|.9|
|40x40-random-ppo-maskable.zip|uint8 RGB3×40×40|1600|5,013,504|.9|

The action head, observation encoding, grid shape, intervention semantics, simulator
and timing differ. These weights are **incompatible**, even if a deserializer could
load them. We train a fresh Maskable PPO policy. No resizing or silent substitution
is used.

## Adaptation implemented here

`hierarchy/local/policy.py` supplies `LocalConstructionEnv`, `SmallFireCNN` and
`FrozenLocalPolicy`. The CNN sees six physical maps plus an eligibility channel;
wind, normalized remaining budget and remaining time are supplied as context.
Two padded3×3 convolutions retain cell locations and feed a128-feature dense layer.
Maskable PPO has separate policy/value heads and outputs one cell choice plus wait.
The initial6×6 experiment therefore has37 local actions.

The local simulator retains persistent, fuel-zero firebreaks, one-cell construction
capacity and wind-driven eight-neighbour spread. Burning and burned cells are
invalid. Adjacent selections can form perimeters, but connectivity is not hardcoded
or guaranteed by the network. There is no heuristic inside the learned policy.

Training uses normalized incremental vegetation loss, multiplied by10 solely for
optimization scale. There is no perimeter-completion bonus or expert target. Both
clearing and burning remain losses; protective sequences must improve future loss.
The local finite-horizon boundary is terminal for PPO, avoiding timeout value
bootstrapping beyond the stated objective.

During local training an independent Bernoulli(.5) service process sometimes
withholds the team: those intervals permit only wait, while fire advances normally.
This exposes the network to states after interrupted construction. The eligibility
input and PPO sampling/log-probabilities use that same mask. Local holdouts report
both continuous service and this intermittent-service setting. During global use
only the allocator determines service; there are no extra random outages.

`training.py` selects the best trained checkpoint using disjoint validation seeds,
records initialization separately, and evaluates held-out placement baselines.
`FrozenLocalPolicy.load()` verifies a JSON schema/environment signature and the ZIP's
SHA-256, then disables gradients and uses deterministic masked inference. Global
checkpoints record this exact constructor hash. Global strategies all use that
same fixed constructor. Old single-level DQN weights have no role in the hierarchy.

This is a Firehose-inspired policy design and training algorithm on our synthetic
simulator, not a reproduction of Firehose results or Cell2Fire physics. A trained
checkpoint and successful integration do not establish learned containment.
