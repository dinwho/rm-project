"""Exact small-instance catalogue of simultaneous feasible construction subsets."""

from itertools import combinations
from math import comb
import numpy as np

from .arms import CandidateArms


class AllocationCatalogue:
    def __init__(self, arms: CandidateArms, capacity: int | None = None,
                 resource_cost: int | None = None, max_allocations: int | None = None):
        config = arms.config
        self.capacity = config.capacity if capacity is None else capacity
        self.resource_cost = config.resource_cost if resource_cost is None else resource_cost
        limit = config.max_allocations if max_allocations is None else max_allocations
        if self.capacity < 0 or self.resource_cost < 1 or limit < 1:
            raise ValueError("Invalid allocation capacity, cost or limit")
        # Conservative bound, checked before generating any combinations.
        upper = sum(comb(arms.count, k) for k in range(min(self.capacity, arms.count)+1))
        if upper > limit:
            raise ValueError(f"Allocation bound {upper} exceeds max_allocations={limit}; "
                             "reduce candidate density/capacity for exact subset DQN")
        cells = [frozenset(s.cells) for s in arms.segments]
        subsets = [()]
        for size in range(1, min(self.capacity, arms.count)+1):
            for subset in combinations(range(arms.count), size):
                if all(cells[i].isdisjoint(cells[j]) for i, j in combinations(subset, 2)):
                    subsets.append(subset)
        self.subsets = tuple(subsets)
        self.count = len(subsets)
        self._indices = {subset: index for index, subset in enumerate(subsets)}
        self.incidence = np.zeros((self.count, arms.count), dtype=bool)
        for index, subset in enumerate(subsets):
            self.incidence[index, list(subset)] = True
        self.costs = self.incidence.sum(axis=1) * self.resource_cost

    def decode(self, action: int) -> tuple[int, ...]:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < self.count:
            raise ValueError("Allocation index is outside the action catalogue")
        return self.subsets[int(action)]

    def encode(self, arm_ids) -> int:
        subset = tuple(sorted(arm_ids))
        if subset not in self._indices:
            raise ValueError("Allocation is duplicated, overlapping or exceeds capacity")
        return self._indices[subset]

    def mask(self, eligibility: np.ndarray, budget: int) -> np.ndarray:
        eligible = np.asarray(eligibility, dtype=bool)
        if eligible.shape != (self.incidence.shape[1],) or budget < 0:
            raise ValueError("Invalid eligibility shape or budget")
        return ~(self.incidence & ~eligible).any(axis=1) & (self.costs <= budget)
