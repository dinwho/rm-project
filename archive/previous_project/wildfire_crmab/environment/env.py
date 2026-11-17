"""Coupled CRMAB construction environment: one complete allocation per fire tick."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .allocations import AllocationCatalogue
from .arms import CandidateArms
from .config import EnvironmentConfig
from .scenario import Scenario, generate_world
from .spread import ignition_probability


class WildfireEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 6}

    def __init__(self, config: EnvironmentConfig | None = None, observe_wind: bool = True,
                 budget_scale: int = 9, render_mode: str | None = None):
        super().__init__()
        self.config = config or EnvironmentConfig()
        self.observe_wind = observe_wind
        self.budget_scale = max(1, budget_scale, self.config.initial_budget or 0)
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.render_mode = render_mode
        self.arms = CandidateArms(self.config)
        self.allocations = AllocationCatalogue(self.arms)
        self.action_space = spaces.Discrete(self.allocations.count)
        n = self.config.grid_size
        self.observation_space = spaces.Dict({
            "map": spaces.Box(0, 1, (6, n, n), np.float32),
            "arms": spaces.Box(0, 1, (self.arms.count, self.arms.feature_count), np.float32),
            "context": spaces.Box(np.array([-1, -1, -1, 0], np.float32), np.ones(4, np.float32)),
        })
        self.finished = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            scenario = Scenario(int(self.np_random.integers(0, 2**31)), self.config.initial_budget)
        if not isinstance(scenario, Scenario) or (scenario.budget is not None and scenario.budget > self.budget_scale):
            raise ValueError("Scenario budget must fit the fixed observation budget_scale")
        self.scenario = scenario
        self.world = generate_world(self.config, scenario)
        self.fuel = self.world.fuel.copy()
        self.settlement = self.world.settlement.copy()
        self.vegetation = (self.fuel > 0) & ~self.settlement
        self.fire_age = np.zeros_like(self.fuel, dtype=np.int32)
        self.fire_age[self.world.ignition] = self.config.burn_duration
        self.ever_burned = self.fire_age > 0
        self.firebreak = np.zeros_like(self.settlement)
        self.built = np.zeros(self.arms.count, dtype=bool)
        self.budget = scenario.budget
        self.budget_used = 0
        self.step_number = 0
        self.total_reward = 0.0
        self.finished = False
        self._pending_construction = None
        return self.observation(), self.info()

    @property
    def wind(self):
        return self.world.wind if self.config.wind_physics else np.zeros(2, dtype=np.float32)

    def eligibility(self):
        return self.arms.eligible(self.fuel, self.ever_burned, self.firebreak,
                                  self.settlement, self.built)

    def arm_states(self):
        return self.arms.states(self.fuel, self.fire_age, self.ever_burned,
                                self.firebreak, self.settlement, self.built)

    def action_mask(self):
        return self.allocations.mask(self.eligibility(), self.budget)

    def observation(self):
        maps = np.stack((self.fuel, self.fire_age > 0,
                         self.fire_age / self.config.burn_duration,
                         self.ever_burned, self.firebreak, self.settlement)).astype(np.float32)
        wind = self.wind if self.observe_wind else np.zeros(2, dtype=np.float32)
        budget_fraction = -1.0 if self.budget is None else self.budget/self.budget_scale
        context = np.array([*wind, budget_fraction,
                            1-self.step_number/self.config.max_steps], dtype=np.float32)
        features = self.arms.features(self.fuel, self.fire_age, self.ever_burned,
                                     self.firebreak, self.settlement, self.built)
        return {"map": maps, "arms": features, "context": context}

    def validate_action(self, action: int):
        """Check an action without mutating state, for atomic multi-incident use."""
        if self.finished:
            raise RuntimeError("Reset before stepping a finished episode")
        if self._pending_construction is not None:
            raise RuntimeError("Advance fire before another construction phase")
        subset = self.allocations.decode(action)
        if not self.action_mask()[action]:
            raise ValueError("Allocation violates eligibility or remaining material budget")
        return subset

    def construct(self, action: int):
        """Apply one validated allocation; physical time does not advance here.

        The only supported next phase is ``advance_fire``. Global orchestration
        uses this to construct at every assigned incident before any fire moves.
        """
        subset = self.validate_action(action)
        # Complete validation precedes mutation: simultaneous construction phase.
        cleared = 0
        for arm_id in subset:
            rows, cols = self.arms.cells(arm_id)
            self.firebreak[rows, cols] = True
            self.fuel[rows, cols] = 0
            self.built[arm_id] = True
            cleared += len(rows)
        spent = len(subset) * self.config.resource_cost
        if self.budget is not None:
            self.budget -= spent
        self.budget_used += spent
        self._pending_construction = (subset, cleared, spent)

    def advance_fire(self):
        """Advance exactly once after a construction phase, including a wait."""
        if self.finished or self._pending_construction is None:
            raise RuntimeError("Construct or wait before advancing fire")
        subset, cleared, spent = self._pending_construction
        # One synchronous shared-field transition, irrespective of subset size.
        probability = ignition_probability(self.fire_age > 0, self.fuel, self.wind,
                                             self.config.base_spread, self.config.wind_bias)
        susceptible = (self.fuel > 0) & ~self.ever_burned & ~self.firebreak
        ignited = susceptible & (self.world.spread_draws[self.step_number] < probability)
        self.fire_age = np.maximum(self.fire_age - 1, 0)
        self.fire_age[ignited] = self.config.burn_duration
        self.ever_burned |= ignited
        self.fuel[self.ever_burned & (self.fire_age == 0)] = 0
        land_damage = int(np.count_nonzero(ignited & self.vegetation))
        settlement_damage = int(np.count_nonzero(ignited & self.settlement))
        reward = -(land_damage + self.config.clearing_weight * cleared) / int(self.vegetation.sum())
        reward -= self.config.settlement_weight * settlement_damage / max(1, int(self.settlement.sum()))
        self.total_reward += float(reward)
        self.step_number += 1
        terminated = not bool(np.any(self.fire_age > 0))
        truncated = self.step_number >= self.config.max_steps and not terminated
        self.finished = terminated or truncated
        self._pending_construction = None
        info = self.info()
        info.update(allocation=subset, resources_spent=spent, constructed_count=len(subset),
                    newly_ignited_vegetation=land_damage, newly_ignited_assets=settlement_damage,
                    newly_cleared=cleared, invalid_action=False)
        return self.observation(), float(reward), terminated, truncated, info

    def step(self, action: int):
        """One construction phase followed by exactly one physical interval."""
        self.construct(action)
        return self.advance_fire()

    def info(self):
        vegetation_count = int(self.vegetation.sum())
        return {
            "action_mask": self.action_mask(),
            "scenario_id": self.scenario.scenario_id,
            "budget_remaining": self.budget,
            "budget_used": self.budget_used,
            "burned_fraction": float(np.count_nonzero(self.ever_burned & self.vegetation)/vegetation_count),
            "affected_fraction": float(np.count_nonzero((self.ever_burned | self.firebreak) & self.vegetation)/vegetation_count),
            "settlement_survived": float(not np.any(self.ever_burned & self.settlement)),
            "settlement_damage": float(np.count_nonzero(self.ever_burned & self.settlement)/max(1, int(self.settlement.sum()))),
            "steps": self.step_number,
            "active_fire_cells": int(np.count_nonzero(self.fire_age)),
            "constructed_segments": int(self.built.sum()),
            "episode_return": self.total_reward,
        }

    def render(self):
        image = np.full((*self.fuel.shape, 3), [224, 220, 200], dtype=np.uint8)
        image[(self.fuel > 0) & ~self.settlement] = [65, 135, 75]
        image[self.settlement] = [72, 125, 210]
        image[self.ever_burned] = [65, 60, 55]
        image[self.firebreak] = [207, 173, 117]
        image[self.fire_age > 0] = [244, 83, 32]
        return image
