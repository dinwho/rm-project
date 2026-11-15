"""Simple vegetation, one ignition, constant wind and policy-independent draws."""

from dataclasses import dataclass
import numpy as np

from .config import EnvironmentConfig


@dataclass(frozen=True)
class Scenario:
    seed: int
    budget: int | None
    wind: tuple[float, float] | None = None

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("Scenario seed must be a nonnegative integer")
        if self.budget is not None and (isinstance(self.budget, bool)
                or not isinstance(self.budget, (int, np.integer)) or self.budget < 0):
            raise ValueError("Scenario budget must be a nonnegative integer or None")
        if self.wind is not None:
            w = np.asarray(self.wind, dtype=float)
            if w.shape != (2,) or not np.isfinite(w).all() or np.linalg.norm(w) > 1.000001:
                raise ValueError("Wind must be a finite two-vector of norm at most one")

    @property
    def scenario_id(self):
        return f"s{self.seed}-b{self.budget}" + (f"-w{self.wind}" if self.wind is not None else "")


@dataclass
class World:
    fuel: np.ndarray
    settlement: np.ndarray
    ignition: tuple[int, int]
    wind: np.ndarray
    spread_draws: np.ndarray


def generate_world(config: EnvironmentConfig, scenario: Scenario) -> World:
    layout_seed, wind_seed, spread_seed = np.random.SeedSequence(scenario.seed).spawn(3)
    rng = np.random.default_rng(layout_seed if config.fixed_layout_seed is None else config.fixed_layout_seed)
    n = config.grid_size
    fuel = np.ones((n, n), dtype=np.float32)
    settlement = np.zeros((n, n), dtype=bool)
    row, col = rng.integers(n//3, 2*n//3, size=2)
    # Still draw the legacy layout coordinates so disabling assets does not
    # silently change the paired ignition or random spread field.
    if config.include_settlement:
        settlement[row:row+2, col:col+2] = True
    along = int(rng.integers(1, n-1))
    edge = int(rng.integers(4))
    ignition = ((0, along), (n-1, along), (along, 0), (along, n-1))[edge]
    if config.fixed_ignition is not None:
        ignition = tuple(config.fixed_ignition)
        if settlement[ignition]:
            # The fixed diagnostic ignition is vegetation, never an asset. This
            # branch does not alter generation of any legacy random scenario.
            settlement[:] = False
            candidates = [(r, c) for r in range(n-1) for c in range(n-1)
                          if not (r <= ignition[0] < r+2 and c <= ignition[1] < c+2)]
            row, col = candidates[0]
            settlement[row:row+2, col:col+2] = True
    weather_rng = np.random.default_rng(wind_seed)
    angle, speed = weather_rng.uniform(0, 2*np.pi), weather_rng.uniform(0.3, 0.9)
    specified_wind = scenario.wind if scenario.wind is not None else config.fixed_wind
    wind = (speed * np.array([np.sin(angle), np.cos(angle)]) if specified_wind is None
            else np.asarray(specified_wind)).astype(np.float32)
    draws = np.random.default_rng(spread_seed).random((config.max_steps, n, n), dtype=np.float32)
    return World(fuel, settlement, ignition, wind, draws)
