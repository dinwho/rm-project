# Historical model — replaced by cell construction

This document records the separated-segment experiment only. The current specification is [FORMULATION.md](../FORMULATION.md).

# Deep Reinforcement Learning for Constrained Wildfire Resource Allocation

## Specification, fixed before implementation

The problem is allocation of limited **firebreak construction** resources. CRMAB
defines these decisions; a physical wildfire simulator realizes their transitions;
DQN learns a policy over feasible simultaneous allocations. No extra water,
regional suppression, or abstract two-state danger process is used.

Reference: Kaza et al., *Constrained Restless Bandits for Dynamic Scheduling in
Cyber-Physical Systems*, IEEE Access 12 (2024), 182274–182295,
[DOI](https://doi.org/10.1109/ACCESS.2024.3510558),
[published full text](https://nrc-publications.canada.ca/eng/view/ft/?id=2ff6f915-4c05-4073-8dfa-e17a5e464c43).
Sections II–III motivate availability-constrained binary arm decisions, restless
evolution, and long-term allocation. Section VIII is a different application,
not the intervention model implemented here. The paper studies index and rollout
methods; DQN is this project's solver, not an algorithm attributed to that paper.

## 1. Arms and sufficient system state

Fix a finite catalogue of N candidate segments P_i before an episode. Every P_i
is an in-bounds, horizontal or vertical, L-cell contiguous segment; catalogue
geometry and identifiers remain fixed across training and evaluation. Arms are
these construction opportunities, not individual grid cells. The initial default
is a 12 by 12 grid, length-three segments on a sparse fixed lattice, one ignition,
uniform vegetation and one protected settlement. Candidate neighbourhoods may
overlap. Segment orientation describes its geometry, not wind direction.

Let G_t=(F_t,A_t,E_t,C_t,S) contain cell fuel, remaining fire lifetime, ever-ignited
indicator, persistent cleared/firebreak indicator, and protected-asset indicator.
Let b_i(t) indicate that candidate i was actually constructed. A different built
segment overlapping P_i does not make b_i true: it instead makes i ineligible.
Arm i has explicit state

    s_i(t) = (b_i(t), F_t|H_i, A_t|H_i, E_t|H_i, C_t|H_i, Y_i(t)),

where H_i is the segment's bounding rectangle expanded by one cell. The sufficient
global state is Z_t=(G_t,b(t),B_t,w,t), including fire outside the local patches.
Local arm summaries alone are not assumed Markov. The simulator stores Z_t and
derives every arm state from it after every physical transition, including passive
and unavailable arms. Fuel is initially one everywhere and is not a hidden state.

## 2. Actions, persistence and eligibility

Each allocation is a binary vector a(t) in {0,1}^N, chosen simultaneously.
Active a_i=1 constructs P_i at the start of the interval. Passive a_i=0 leaves
its construction status unchanged. Construction is instantaneous in this initial
model; no crew travel or multi-interval construction is assumed.

    b_i(t+1) = max(b_i(t), a_i(t))
    C_t^+ = C_t union (union of P_i with a_i=1)
    F_t^+(v) = 0 for v in newly constructed segments; F_t(v) otherwise.

Physical eligibility is fully observed:

    Y_i(t)=1 iff b_i(t)=0 and every v in P_i has positive fuel,
             E_t(v)=0, C_t(v)=0, and S(v)=0.

Thus burning, previously burned, already cleared, asset-containing, and previously
built segments cannot be constructed. Fire can make an unbuilt arm unavailable.
In this no-regrowth model, unavailable construction sites do not recover. This is
a one-construction opportunity per arm, not the paper's recoverable logistics
availability process. Eligibility and affordability are distinct: a site can be
physically eligible when the remaining budget is zero.

## 3. Hard allocation constraints

One fixed-size segment costs c integer material units. For every interval:

    a_i <= Y_i(t),   sum_i a_i <= K,
    c sum_i a_i <= B_t,   B_(t+1) = B_t - c sum_i a_i.

Selected segments must also be pairwise disjoint: overlapping simultaneous
constructions are invalid, avoiding duplicate work and ambiguous material costs.
K is renewable construction capacity **per decision interval**. B_0 is a
consumable **whole-episode** budget. Unused capacity does not carry forward;
unspent material does. Zero budget does not stop fire evolution. Empty allocation
is always feasible. Invalid allocations raise before any mutation or time advance.

## 4. Physical transition and wind

After the entire allocation is constructed, all fire spread uses the same
post-construction snapshot. For each burning source u and susceptible eight-neighbour
target v, d=v-u:

    q_uv = clip(p0 F_t^+(v) exp(eta * dot(w,d)/||d||) / ||d||, 0, 0.95)
    p_v = 1 - product_(burning u adjacent v) (1-q_uv)
    I_v = 1{F_t^+(v)>0, E_t(v)=0, U_(t,v)<p_v}.

Wind w=(row velocity,column velocity) points downwind, is fixed within an episode,
and varies in direction and strength between episodes. Construction removes fuel
before q is evaluated. Firebreak cells cannot ignite. Newly ignited cells cannot
spread again in the same interval. Existing fire ages decrement once; new fire
gets lifetime D. Spent vegetation becomes zero fuel. E is the union of past and
new ignitions. Diagonal bypass around short firebreaks is allowed; there is no
spotting or crossing directly over a cleared cell to a non-neighbour cell.

Active arms change their own and other arms' future neighbourhoods by changing
spread paths. Passive arms still see spreading, ageing, and fuel loss. Built arms
retain b_i=1 while their surrounding fire continues evolving. There is no special
passive-freeze operation.

## 5. Reward and objective

Let V_0 be initial non-asset vegetation count, S_0 the asset cell count,
n_V newly ignited vegetation cells, n_S newly ignited asset cells, and n_C newly
cleared vegetation cells. The interval reward is

    r_t = -(n_V + lambda_C n_C)/V_0 - lambda_S n_S/S_0.
    maximize E_pi[sum_(t=0)^(T-1) gamma^t r_t], 0 < gamma <= 1.

Default lambda_C=1 accounts for vegetation sacrificed by firebreaks and prevents
clearing from being reported as cost-free preservation. Resource feasibility is
enforced separately, not through a penalty. Initial ignition is an action-independent
constant omitted from rewards and included in outcome metrics. No vegetation or
asset is charged for ignition twice. The finite incident ends when all fire dies
or at observed horizon T. Bootstrapping stops at either boundary because this is
a finite-horizon objective. Remaining active fire and time-limit endings are reported.

## 6. Exact scope of the CRMAB extension

This is a **spatially coupled, single-construction CRMAB extension** with persistent
interventions, observed state-dependent eligibility, renewable capacity and a
consumable budget. Its product binary action vector and feasible allocation set
are defined by construction arms; the grid supplies their shared transition field.

The paper's independent-arm factorization

    P(s'|s,a) = product_i P_i(s_i'|s_i,a_i)

does **not** hold. A neighbour's action can change s_i', local patches overlap,
and outside fire can enter H_i. The smallest necessary extension is to retain the
shared Markov field G and use joint P(Z'|Z,a); no independent local transition
approximation is used. Adding budget B and persistent construction bits makes
these effects and constraints Markov in the augmented state. There is no inherited
Whittle indexability, threshold-policy result, optimality theorem, or reproduction
claim. DQN is an approximate joint-allocation solver for this extension.

Equivalently, this extension is a constrained factored Markov decision process
with binary construction-arm decisions and a shared spatial field. The CRMAB
interpretation describes the allocation structure; it is not a claim to have
reduced the coupled process to classical independent bandits. Reward is accounted
over unique physical cells, not summed over overlapping arm patches (which would
double-count damage). No additive independent arm-reward decomposition is assumed.

## 7. Learning interface and simultaneous allocation

For small N,K, enumerate every pairwise-disjoint subset of size at most K once,
including empty. Q_theta(o,j) is the discounted value of executing the **entire
allocation** catalogue[j], then following the policy. It is not an arm index.
The number is at most sum_(k=0)^K binomial(N,k); a configured hard catalogue cap
must reject oversized problems before enumeration. This is an exact small-instance
representation, not a scalable solution for arbitrary N or K. No internal
sequential construction decisions are used and time advances once per subset.

The observation includes six physical maps (fuel, fire, normalized fire lifetime,
ever ignited, cleared, assets), explicit per-arm construction/eligibility/local
condition/geometry features, remaining budget and time, and current wind. Arm
construction bits are necessary: different construction histories can create the
same cleared union. The complete grid is supplied because local summaries are not
Markov. The catalogue geometry is fixed and known to every policy.

Wind-aware and wind-unaware DQN are trained separately with identical wind-driven
physics, maps, constraints, rewards and scenario splits. Only explicit wind inputs
are zeroed in the unaware observation; arm summaries and masks do not encode wind.
The unaware agent can infer some wind from fire trajectories, but a feed-forward
DQN does not maintain an exact belief: this variant is an observation ablation of
a partially observed problem. Removing wind physics is a separate simulator
setting and is never labelled the observation ablation. Other than this ablation,
partial fire observations from the paper are not inherited by this application.

The same feasible mask applies to epsilon exploration, greedy selection, replay
validation, and the next-state maximum in the Bellman target. Terminal targets
equal r, avoiding 0 times negative-infinity. Repeated or overbudget actions cannot
be selected through these interfaces; direct invalid callers get a strict error.

## 8. Experiment protocol

Compare no construction, random feasible allocations, a near-fire placement rule,
and a wind-aware placement rule, plus the two DQNs. Heuristics score whole feasible
subsets, not one intervention per step. Baselines see only current observations,
the same public geometry and physical parameters, and feasibility masks. Report
which rules receive wind. No policy sees future draws. A no-wind-physics experiment
is available separately for sensitivity, not substituted for the wind ablation.

Use disjoint train/validation/test seeds, fixed wind per episode, matched worlds
and random spread fields indexed by (time,row,column) across methods and budgets,
validation-only checkpoint choice, independent learning seeds, raw episode tables,
paired differences and uncertainty intervals. Report both vegetation burned and
burned-or-cleared, asset loss, budget use, unfinished incidents, and discounted
return. Smoke runs verify integration; preliminary pilots do not establish DQN
superiority or a calibrated wildfire forecast.
