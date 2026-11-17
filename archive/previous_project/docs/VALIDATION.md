# Cell-construction validation

Validated on 16 November 2026. Full suite: **101 passed**. This supersedes the
old segment validation count; original segment results remain in results/pilot/.

Checks cover every-cell geometry and boundary anchors, persistent fuel removal,
burning/repeated-action rejection, passive evolution, exact spending with both
unlimited and finite episode materials, capacity, atomic timing, synchronous
spread, wind physics versus hidden observations, feasible DQN selection/targets,
checkpoint roundtrips and complete training/evaluation integration.

A closed square ring is constructed through 16 separate legal capacity-one steps.
Controlled random draws delay spread during construction, then force every
positive-probability neighbour transmission. The complete ring protects outside
vegetation; an equal-cost construction with one corner missing leaks diagonally.
This is a geometry stress test, not a typical containment-rate experiment.

`examples/connected_barrier.py` also uses unchanged stochastic physics: seed0,
12×12, corner ignition, wind(0,.8). Three legal sequential placements leave one
cell burned and three cleared; waiting on the same draws burns141 cells. This
fixed hand-built illustration establishes an effect, not learned generalization.

## Learning diagnostic

[Measured results](../results/continuous_fixed/RESULTS.md): 8×8, fixed corner
ignition and wind(0,.5), one cell per interval, no episode budget, three learning
seeds and 10,000 transitions per DQN variant/seed. Spread fields are independent
across train/validation/test; all policies share the32 held-out realizations.
320 evaluation episodes completed with zero invalid actions and zero truncations.

| Policy | Mean burned-or-cleared fraction |
| --- | --- |
| Wait | .943 |
| Random | .892 |
| Near-fire heuristic | .494 |
| Wind-aware heuristic | .326 |
| DQN | .699 |
| Wind-unaware DQN | .671 |

Five of six trained policies improved on their initialization on validation;
one wind-aware seed regressed. Wind-aware DQN beats random on paired test return
(difference .182, descriptive95% interval[.036,.343]), but trails the wind-aware
heuristic (-.344, interval[-.519,-.172]). The wind-aware versus unaware difference
is unresolved (-.029, interval[-.190,.142]). Fixed wind makes this diagnostic
unsuitable for estimating the benefit of observations across different winds.

Connected construction is now possible; reliable learned containment is not
established. Delayed protective benefits, immediate clearing costs, stochastic
spread, large action choices and one-cell construction capacity are plausible
learning difficulties, not experimentally isolated causal explanations. Broad
variable-wind training has not been run for the new model.

The simulator remains synthetic, spatially coupled and uncalibrated. There is
no spotting, terrain or guaranteed independence/indexability. A single successful
barrier or a passing test suite does not prove a good learned policy.
