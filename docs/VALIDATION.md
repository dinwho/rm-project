# Validation record

Validated locally on 16 November 2026 using Python 3.14.8, NumPy 2.5.4,
PyTorch 2.15.0+cpu, Gymnasium 1.4.0 and pytest 9.2.0. Exact core dependencies
are in `requirements-cpu.lock.txt`. The sibling project was not modified.

## Executed checks

`python -m pytest -q`: **65 passed** (10.81 seconds on this machine, including demo tests).

| Area | Evidence |
| --- | --- |
| Restless passive evolution | Unselected and unavailable arms change as fire spreads and ages; built-arm neighbourhoods also evolve |
| Active effect and persistence | Selected cells become permanent fuel-zero firebreaks; build bits persist and reconstruction is rejected |
| Capacity and budget | Two-segment allocations, renewable interval capacity, depleting episode materials, nonunit costs and zero-budget continuing fire |
| Eligibility | Burned, cleared, asset-containing, fuel-zero and already-built sites rejected; overlap invalidates neighbours without falsely marking them built |
| Atomic allocation timing | Invalid actions do not mutate state/time; two selected segments block spread before the same single fire tick |
| Physical dynamics | No wraparound, wind direction affects ignition, no within-tick cascades, only new damage rewarded |
| Spatial coupling | Building one segment changes a separate passive candidate's subsequent fire state and eligibility in a controlled corridor |
| Wind experiment | Hiding wind leaves all maps, arm features and rewards identical under identical actions; removing physical wind changes transitions |
| DQN interface | Environment observations fit network/replay; exploration and greedy choices obey masks; large invalid next-Q values cannot enter targets |
| Finite objective | Terminal targets equal reward even for empty successor masks; budget exhaustion continues dynamics, while extinction/horizon end episodes |
| Learning and checkpoints | Parameters change after optimization, target copying works, saved/reloaded outputs agree |
| Experiment protocol | Disjoint seed splits, matched scenario differences, repeated-budget clustering, configuration validation |
| Complete pipeline | Real optimization, validation checkpoint selection, reload, baseline/DQN evaluation and report generation in a tiny integration run |

The Gymnasium checker uses the capacity-zero configuration because it samples
actions without consulting feasibility. Separate tests validate the normal
capacity-two environment. This is a masked environment; generic unmasked callers
must be adapted, not silently treated as wait.

`python examples/one_allocation.py` also ran successfully: 18 arms, 163 whole
allocations, two constructions in one interval, material use 2, remaining budget
4, persistent build flags and continuing passive fire evolution.

## Experiment interpretation

The smoke configuration ran both DQN variants for 600 physical intervals each,
then evaluated six methods on six held-out worlds at three budgets: 108 episodes,
zero invalid allocations and 108 intervals constructing multiple segments.
This is pipeline evidence, not a performance claim.

The pilot configuration specifies three independent learning seeds, 10,000
physical intervals per variant/seed, eight validation worlds and 24 test worlds,
with each world evaluated at budgets 2, 4 and 6. The selected checkpoint maximizes
discounted validation return; test performance never selects a checkpoint.
Detailed measured outputs are in [the pilot report](../results/pilot/RESULTS.md).

The completed pilot evaluated **720 episodes** with **zero invalid allocations**,
996 intervals constructing multiple segments, and 108 horizon-limited endings.
Mean discounted returns were -8.284 (no intervention), -7.356 (random), -5.027
(near-fire heuristic), -5.031 (wind heuristic), -6.951 (DQN), and -7.095
(wind-unaware DQN). The paired wind-aware minus wind-unaware return difference
was 0.144, with a 95% interval [-0.668, 0.922]. This pilot does not resolve a wind
observation advantage. Both placement heuristics outperformed DQN; no DQN
superiority is claimed. No settings were changed in response to test results.

Full artifacts include best/last weights, per-episode training logs, validation
logs, raw test outcomes, paired comparisons, source snapshots and hashes. Compact
snapshots in `results/` omit weights and source copies; their export manifest points
to the complete run. The larger 200,000-step/five-seed protocol is provided but is
not reported as executed.

## Interactive demo

`demo/` imports the same environment and saved policies. It exposes actual
allocation Q-values and a complete state trajectory; browser playback is inference
replay, not browser training. The standalone HTML embeds world 42/budget 6 and all
six policies. The local server accepts other seeds and budgets.

Live and offline versions were tested in Chromium: six policy choices, next/back,
autoplay, arm selection, hidden wind display, architecture, measured results and
390-pixel mobile layout, with no JavaScript errors. `demo/check_browser.py` records
the repeatable flow. Demo regression tests verify JSON serialization and that
displayed allocations, budget accounting and Q rankings match the actual simulator.

## Limits of the claims

The tests establish consistency with the stated coupled CRMAB formulation and
the specified discrete simulator, not calibration to real wildfire data.
Independent-arm factorization and indexability do not hold here. Eligibility is
observable and irreversible for this incident; there is no inferred stochastic
logistics process. Fully observed fire conditions are a stated modelling choice.

The fixed catalogue restricts the placements DQN and baselines can make. Exact
subset enumeration is suitable for this small instance and deliberately refuses
oversized cases. Single-layer firebreak segments may be bypassed diagonally.
There is no terrain, spotting, construction time, safety-distance constraint,
multiple ignition, changing intra-episode wind or field-calibrated fire model.

Short training and few seeds can give unstable policy rankings. Confidence
intervals resample training seeds and worlds, preserving budget clusters; they
are descriptive and not corrected for multiple comparisons. The wind-unaware
feed-forward DQN does not maintain a hidden-wind belief or recurrent history.
Time-limited survival does not imply eventual containment. Report active fire,
asset damage and burned-or-cleared vegetation alongside return.
