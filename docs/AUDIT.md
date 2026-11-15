# Audit and implementation map

Work belongs in `/home/d1nu/LEBRON/BLOND/research`. It initially contained no files.
The sibling `research-project` was inspected read-only as prior context. Its
placement geometry encoded one patch action per step, no explicit construction-arm
state, no simultaneous allocation capacity, and invalid placement advanced time
as wait. Its results do not validate the new allocation problem.

Useful numerical primitives may be retained with explicit attribution: safe grid
shifts and eight-neighbour ignition probabilities; a compact CNN; masked DQN,
replay and seed/world bootstrap concepts. These are implementation utilities, not
the problem definition. The new arm catalogue, arm state, simultaneous subset
allocation, strict feasibility, observation schema, experiments and tests are
derived from FORMULATION.md. No sibling project files or results are modified.

| Mathematics | Implementation | Reuse/change |
| --- | --- | --- |
| P_i, H_i and arm identifiers | `environment/arms.py`: `CandidateArms`, `Segment`, `ArmState` | New explicit candidate and local-state model |
| s_i, b_i, Y_i | `environment/arms.py`: `states`, `eligible`, `features`; `environment/env.py`: `built` | New persistent candidate identity, snapshots and eligibility |
| a in feasible subset set | `environment/allocations.py`: `AllocationCatalogue.mask`, `decode` | Replaces single-patch placement encoding |
| K and c sum a <= B | `environment/allocations.py` plus `environment/env.py`: `step` | New simultaneous capacity; strict atomic rejection |
| Persistent C, zero cleared fuel | `environment/env.py`: `step` construction phase | Physical clearing retained; driven by subset arms |
| G, passive and active transitions | `environment/env.py`: `step`; `environment/spread.py`: `ignition_probability` | Shared synchronous physics, refreshed arm states for all candidates |
| q_uv, p_v, U_(t,v), w | `environment/spread.py`; `environment/scenario.py`: `generate_world` | Spread primitive retained; fixed episode wind, indexed draws |
| r_t, finite objective | `environment/env.py`; `evaluation/runner.py`; `agents/network.py` | Burn/clear/asset loss accounting retained and tested |
| o_t | `environment/env.py`: `observation`; `agents/network.py`: `QNetwork` | Maps plus explicit arm features and budget/time/wind |
| Q(o, entire subset) | `agents/dqn.py`: `DQNAgent`; `environment/allocations.py` | DQN head now values simultaneous allocations |
| Feasible Bellman target | `agents/network.py`: `bellman_targets`; `agents/replay.py`: `add` | Explicit current and successor mask checks |
| Matched validation/test comparisons | `training/train.py`; `evaluation/` | Fresh protocol, checkpoints and results; no imported old scores |

The detailed equations and assumptions in FORMULATION.md are the implementation
specification. VALIDATION.md records measured checks and experiment limitations.
