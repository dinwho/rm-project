# Deep Reinforcement Learning for Constrained Wildfire Resource Allocation

**CRMAB defines the allocation problem → the simulator implements it → DQN learns a feasible policy.**

An arm is a **cell where a firebreak can be constructed**. Selecting an eligible
arm permanently clears that cell. All other arms remain passive while the fire
continues spreading. Adjacent placements can form connected barriers and closed
perimeters; placement is not restricted to separated segments.

The default allows **one construction before each fire update, with no episode-wide
material cap**. Burning, previously burned and already cleared cells are ineligible.
Construction sacrifices vegetation, so both burning and clearing count as damage.
There is no extinguishing action. Firebreaks work by removing fuel from actual
spread paths.

This is a **spatially coupled, single-construction CRMAB extension**, equivalently
a constrained factored MDP. Nearby arms interact through fire spread. We do not
claim independent-arm transitions, indexability or guaranteed DQN optimality.

## Start here

- [Beginner's guide](docs/BEGINNERS_GUIDE.md): RL, CRMAB, DQN and the code.
- [Mathematical formulation](docs/FORMULATION.md): states, constraints, physics,
  rewards and the precise Firehose placement adaptation.
- [Code map](docs/ARCHITECTURE.md) and [change audit](docs/AUDIT.md).
- [Validation](docs/VALIDATION.md): barrier checks, tests and measured learning.
- [Simple demo](demo/SIMPLE_DEMO.md): a short Python script producing a GIF.

The initial learning diagnostic uses an **8×8** forest with fixed ignition and
wind, with independent stochastic spread fields for training, validation and test.
The general default is **12×12** with wind and ignition varied between episodes.
No protected settlement is included by default; the older asset scenario remains
an optional configuration.

## Run from this directory

```bash
source .venv/bin/activate
python -m pytest -q
python examples/one_allocation.py
python examples/connected_barrier.py
python demo/simple_demo.py --policy wind_heuristic

# Fixed scenario first: 3 learning seeds, 10,000 transitions per DQN variant/seed.
wildfire-crmab run --config configs/fixed.json --output artifacts/my-fixed-run

# Broader variable-wind protocol; review fixed-scenario learning before running.
wildfire-crmab run --config configs/pilot.json --output artifacts/my-cell-pilot
```

A run requires a new output directory. It saves configuration, source snapshots,
checkpoints, training/validation logs, held-out outcomes and a comparison report.
Checkpoints support inference, not interrupted-training resume. For a fresh setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

The existing workspace has the dependencies installed. `requirements-cpu.lock.txt`
records the previously verified core package versions.

## How the decisions work

For grid side `n`, arm ID is `row*n + column`. With capacity one, action 0 means
wait and action `1 + arm_id` constructs that cell. Thus the default 12×12 problem
has **144 arms and 145 allocation actions**; the 8×8 diagnostic has 64 and 65.
A feasibility mask is applied to DQN exploration, greedy selection and learning
targets. The environment also rejects invalid actions before changing anything.

Optional capacity greater than one uses exact simultaneous subsets: every selected
cell is cleared before one spread update. Oversized catalogues are rejected,
never silently replaced with actions that advance fire between selections.
Optional finite episode budgets are also supported; JSON `null` means no cap.

Eight-neighbour spread can pass diagonally through a missing corner. Connected
placements are possible, but a line must actually separate spread paths and must
be built before the fire arrives. [The barrier example](examples/connected_barrier.py)
compares a manually chosen sequence with waiting under identical random draws.
It demonstrates geometry and timing, not a learned or universally successful rule.

## Components

```text
configs/                 fixed diagnostic, smoke, variable-wind runs; legacy/ settings
src/wildfire_crmab/
  environment/           arm geometry, eligibility, allocations, scenarios, spread
  agents/                DQN network, replay, masks, checkpoints, placement baselines
  training/              learning loop and validation-only checkpoint selection
  evaluation/            matched scenarios, policy rollouts, statistics, reports
  config.py, cli.py      experiment validation and commands
examples/                short executable examples
demo/                   simple animation and optional plain browser replay
docs/                   formulation, explanations, code map and checks
tests/                  environment, agents, learning, evaluation and demo checks
results/                compact measured runs; older segment runs clearly labelled
artifacts/              full local runs with weights and source snapshots
```

Baselines are no construction, random feasible placement, immediate fire-threat
placement and a wind-aware threat rule. They use observations and public model
parameters, never future spread draws. Wind-aware and wind-unaware DQN run in the
**same wind-driven simulator**; only the explicit wind observation is withheld.
Removing wind from the physics is a separate ablation.

The old separated-segment results in `results/pilot/` and `results/smoke/` are
historical. They do not measure the new cell model. Explicit replay configurations
are under `configs/legacy/`. This simulator is synthetic and uncalibrated;
containment tests and a working training loop do not establish DQN superiority.
