"""Validated physical CRMAB parameters; units are cells and construction intervals."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EnvironmentConfig:
    grid_size: int = 12
    placement_mode: str = "cells"
    segment_length: int = 1
    candidate_stride: int = 1
    capacity: int = 1
    resource_cost: int = 1
    initial_budget: int | None = None
    max_steps: int = 40
    burn_duration: int = 3
    base_spread: float = 0.24
    wind_bias: float = 1.2
    settlement_weight: float = 0.0
    clearing_weight: float = 1.0
    max_allocations: int = 5000
    wind_physics: bool = True
    include_settlement: bool = False
    fixed_ignition: tuple[int, int] | None = None
    fixed_wind: tuple[float, float] | None = None
    fixed_layout_seed: int | None = None

    def __post_init__(self):
        integer_fields = ("grid_size", "segment_length", "candidate_stride", "capacity",
                          "resource_cost", "max_steps", "burn_duration", "max_allocations")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.grid_size < 6:
            raise ValueError("grid_size must be at least 6")
        if self.placement_mode not in ("cells", "segments"):
            raise ValueError("placement_mode must be 'cells' or 'segments'")
        if self.placement_mode == "cells" and (self.segment_length != 1 or self.candidate_stride != 1):
            raise ValueError("Cell placement requires segment_length=1 and candidate_stride=1")
        if not 1 <= self.segment_length <= self.grid_size - 2 or self.segment_length % 2 != 1:
            raise ValueError("segment_length must be positive, odd and at most grid_size-2")
        if min(self.candidate_stride, self.resource_cost, self.max_steps,
               self.burn_duration, self.max_allocations) < 1:
            raise ValueError("Stride, cost, durations and catalogue limit must be positive")
        if self.capacity < 0:
            raise ValueError("Capacity must be nonnegative")
        if self.initial_budget is not None and (isinstance(self.initial_budget, bool)
                or not isinstance(self.initial_budget, int) or self.initial_budget < 0):
            raise ValueError("initial_budget must be a nonnegative integer or None")
        if not math.isfinite(self.base_spread) or not 0 <= self.base_spread <= 1:
            raise ValueError("base_spread must lie in [0,1]")
        for name in ("wind_bias", "settlement_weight", "clearing_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.wind_physics, bool):
            raise ValueError("wind_physics must be boolean")
        if not isinstance(self.include_settlement, bool):
            raise ValueError("include_settlement must be boolean")
        if self.fixed_ignition is not None:
            if (not isinstance(self.fixed_ignition, (tuple, list)) or len(self.fixed_ignition) != 2
                    or any(isinstance(x, bool) or not isinstance(x, int)
                    or not 0 <= x < self.grid_size for x in self.fixed_ignition)):
                raise ValueError("fixed_ignition must contain two in-bounds integer coordinates")
            object.__setattr__(self, "fixed_ignition", tuple(self.fixed_ignition))
        if self.fixed_wind is not None:
            if (not isinstance(self.fixed_wind, (tuple, list)) or len(self.fixed_wind) != 2
                    or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in self.fixed_wind)
                    or math.hypot(*self.fixed_wind) > 1.000001):
                raise ValueError("fixed_wind must be a finite two-vector of norm at most one")
            object.__setattr__(self, "fixed_wind", tuple(float(x) for x in self.fixed_wind))
        if self.fixed_layout_seed is not None and (isinstance(self.fixed_layout_seed, bool)
                or not isinstance(self.fixed_layout_seed, int) or self.fixed_layout_seed < 0):
            raise ValueError("fixed_layout_seed must be a nonnegative integer or None")
