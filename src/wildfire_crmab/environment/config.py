"""Validated physical CRMAB parameters; units are cells and construction intervals."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EnvironmentConfig:
    grid_size: int = 12
    segment_length: int = 3
    candidate_stride: int = 4
    capacity: int = 2
    resource_cost: int = 1
    initial_budget: int = 6
    max_steps: int = 40
    burn_duration: int = 3
    base_spread: float = 0.24
    wind_bias: float = 1.2
    settlement_weight: float = 10.0
    clearing_weight: float = 1.0
    max_allocations: int = 5000
    wind_physics: bool = True

    def __post_init__(self):
        integer_fields = ("grid_size", "segment_length", "candidate_stride", "capacity",
                          "resource_cost", "initial_budget", "max_steps", "burn_duration", "max_allocations")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.grid_size < 6:
            raise ValueError("grid_size must be at least 6")
        if not 1 <= self.segment_length <= self.grid_size - 2 or self.segment_length % 2 != 1:
            raise ValueError("segment_length must be positive, odd and at most grid_size-2")
        if min(self.candidate_stride, self.resource_cost, self.max_steps,
               self.burn_duration, self.max_allocations) < 1:
            raise ValueError("Stride, cost, durations and catalogue limit must be positive")
        if self.capacity < 0 or self.initial_budget < 0:
            raise ValueError("Capacity and budget must be nonnegative")
        if not math.isfinite(self.base_spread) or not 0 <= self.base_spread <= 1:
            raise ValueError("base_spread must lie in [0,1]")
        for name in ("wind_bias", "settlement_weight", "clearing_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.wind_physics, bool):
            raise ValueError("wind_physics must be boolean")
