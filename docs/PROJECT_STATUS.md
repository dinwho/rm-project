# Current project scope

The active project is an original-Firehose reproduction with a modular Python
interface and browser replay. The previous CRMAB/DQN and hierarchical designs
were stopped at the user's request and archived under `archive/previous_project`.
They are not imported by the active package or used to generate its demo.

The public Firehose checkpoint contains a trained Maskable PPO CNN. We load its
learned tensors with strict checks, using the authors' original maps, Cell2Fire
physics, harvesting semantics and reward. This is inference from published weights;
we do not claim to have trained those weights locally.

A cell treatment can both prevent future ignition and extinguish existing fire.
Therefore this project now includes active suppression, as explicitly requested.
The original implementation's timestep, observations and reward differ from the
former synthetic grid. See FIREHOSE_REPRODUCTION.md for exact compatibility notes.
