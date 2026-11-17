"""Observation-only policies over complete feasible construction allocations.

The threat heuristics are fixed placement rules, not Whittle indices or an oracle.
They sum immediate ignition risk on candidate segments, weighted by proximity to
protected assets. They do not estimate downstream path blocking or the long-term
cost of spending materials now. All choices use the same allocation mask as DQN.
"""

from __future__ import annotations

import numpy as np

from wildfire_crmab.environment.allocations import AllocationCatalogue
from wildfire_crmab.environment.arms import CandidateArms
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.spread import ignition_probability


BASELINE_NAMES = ("no_intervention", "random", "greedy", "wind_heuristic")


def _valid_indices(action_mask: np.ndarray, expected_count: int | None = None) -> np.ndarray:
    mask = np.asarray(action_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("Action mask must be a one-dimensional boolean array")
    if expected_count is not None and len(mask) != expected_count:
        raise ValueError("Action mask does not match the allocation catalogue")
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise ValueError("No feasible allocation is available")
    return valid


class NoIntervention:
    """Leave every candidate passive at this decision interval."""

    def act(self, observation: dict, action_mask: np.ndarray) -> int:
        valid = _valid_indices(action_mask)
        if valid[0] != 0:
            raise ValueError("The empty allocation must be feasible")
        return 0


class RandomPlacement:
    """Uniform over feasible allocation subsets, including the empty subset.

    This is not uniform over subset sizes: sizes represented by more feasible
    subsets receive more probability. Every sampled subset is applied at once.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: dict, action_mask: np.ndarray) -> int:
        return int(self.rng.choice(_valid_indices(action_mask)))


class ThreatHeuristic:
    """Prioritize eligible segments threatened by current fire near assets.

    ``use_wind=False`` computes risk under zero wind as an information-limited
    baseline. The wind-aware version consumes only the observation's wind vector;
    if wind is hidden, it receives zero just like a wind-unaware DQN. Neither
    version sees simulator draws, future fire, or private environment state.
    """

    def __init__(self, config: EnvironmentConfig, use_wind: bool):
        self.config = config
        self.use_wind = use_wind
        self.arms = CandidateArms(config)
        self.allocations = AllocationCatalogue(
            self.arms,
            capacity=config.capacity,
            resource_cost=config.resource_cost,
            max_allocations=config.max_allocations,
        )
        self.subset_sizes = np.array([len(subset) for subset in self.allocations.subsets])
        self.incidence = np.zeros((self.allocations.count, len(self.arms.segments)), dtype=float)
        for index, subset in enumerate(self.allocations.subsets):
            self.incidence[index, list(subset)] = 1

    def arm_scores(self, observation: dict) -> np.ndarray:
        """Return immediate local placement scores, not long-term Q values."""
        layers = np.asarray(observation["map"])
        fuel, burning, ever_ignited, assets = layers[0], layers[1] > 0, layers[3] > 0, layers[5] > 0
        wind = np.asarray(observation["context"][:2]) if self.use_wind else np.zeros(2)
        # Disabling physical wind is a separate environment ablation.
        if not self.config.wind_physics:
            wind = np.zeros(2)
        risk = ignition_probability(burning, fuel, wind, self.config.base_spread, self.config.wind_bias)
        risk *= (fuel > 0) & ~ever_ignited
        if assets.any():
            rows, cols = np.indices(fuel.shape)
            # Distance to the closest protected cell avoids privileging its centroid.
            distances = np.stack([np.hypot(rows - r, cols - c) for r, c in np.argwhere(assets)])
            risk *= 1 + self.config.settlement_weight / (1 + distances.min(axis=0))
        return np.asarray([float(risk[self.arms.cells(i)].sum()) for i in range(len(self.arms.segments))])

    def act(self, observation: dict, action_mask: np.ndarray) -> int:
        valid = _valid_indices(action_mask, self.allocations.count)
        scores = self.incidence @ self.arm_scores(observation)
        # Maximum total threat, then fewest constructions, then stable catalogue ID.
        # Zero-risk additions therefore never spend extra materials on a score tie.
        order = np.lexsort((valid, self.subset_sizes[valid], -scores[valid]))
        return int(valid[order[0]])


def make_baseline(name: str, config: EnvironmentConfig, seed: int = 0):
    if name == "no_intervention":
        return NoIntervention()
    if name == "random":
        return RandomPlacement(seed)
    if name in {"greedy", "wind_heuristic"}:
        return ThreatHeuristic(config, use_wind=name == "wind_heuristic")
    raise ValueError(f"Unknown baseline: {name!r}; expected one of {BASELINE_NAMES}")
