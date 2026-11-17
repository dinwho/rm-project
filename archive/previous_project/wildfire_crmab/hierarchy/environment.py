"""Independent wildfire incidents with synchronous constrained team allocation.

One active incident receives one team and at most one construction. Local
construction is selected by a shared, frozen policy. Extinguished incidents are
absorbing; every remaining incident advances once even when unassigned.
"""
from itertools import combinations
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.environment.scenario import Scenario


class GlobalWildfireEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, local_config: EnvironmentConfig, local_policy,
                 incident_count: int = 4, team_capacity: int = 2):
        super().__init__()
        if type(incident_count) is not int or not 1 <= incident_count <= 10:
            raise ValueError("incident_count must be an integer from 1 to 10")
        if (local_config.placement_mode != "cells" or local_config.capacity != 1
                or local_config.initial_budget is not None):
            raise ValueError("Hierarchy requires cell construction, local capacity one, and no episode material cap")
        if not callable(getattr(local_policy, "act", None)):
            raise ValueError("local_policy must implement act(observation, action_mask)")
        self.local_config = local_config
        self.local_policy = local_policy
        self.incident_count = incident_count
        self._check_capacity(team_capacity)
        self.team_capacity = team_capacity
        self.incidents = [WildfireEnv(local_config) for _ in range(incident_count)]
        self.subsets = tuple(subset for size in range(incident_count+1)
                             for subset in combinations(range(incident_count), size))
        self._indices = {subset: i for i, subset in enumerate(self.subsets)}
        self.action_space = spaces.Discrete(len(self.subsets))
        self.incidence = np.zeros((len(self.subsets), incident_count), dtype=bool)
        for i, subset in enumerate(self.subsets):
            self.incidence[i, list(subset)] = True
        n = local_config.grid_size
        self.observation_space = spaces.Dict({
            "maps": spaces.Box(0, 1, (incident_count, 6, n, n), np.float32),
            "local_context": spaces.Box(np.tile(np.array([-1, -1, -1, 0], np.float32), (incident_count, 1)),
                                        np.ones((incident_count, 4), np.float32)),
            "eligibility": spaces.Box(0, 1, (incident_count, n*n), np.float32),
            "context": spaces.Box(0, 1, (2,), np.float32),
        })
        self.finished = True

    def _check_capacity(self, capacity):
        if type(capacity) is not int or not 0 <= capacity <= self.incident_count:
            raise ValueError("team_capacity must be an integer between zero and incident_count")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        capacity = options.get("team_capacity", self.team_capacity)
        self._check_capacity(capacity)
        episode_seed = int(self.np_random.integers(0, 2**32)) if seed is None else int(seed)
        incident_seeds = options.get("incident_seeds")
        if incident_seeds is None:
            incident_seeds = [int(child.generate_state(1)[0])
                              for child in np.random.SeedSequence(episode_seed).spawn(self.incident_count)]
        if (len(incident_seeds) != self.incident_count
                or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0
                       for s in incident_seeds)
                or len(set(incident_seeds)) != self.incident_count):
            raise ValueError("Provide one distinct nonnegative incident seed per incident")
        self.team_capacity = capacity
        self.episode_seed = episode_seed
        self.incident_seeds = tuple(int(s) for s in incident_seeds)
        for env, incident_seed in zip(self.incidents, self.incident_seeds):
            env.reset(options={"scenario": Scenario(incident_seed, None)})
        self.step_number = 0
        self.total_reward = 0.0
        self.total_assignments = 0
        self.total_constructions = 0
        self.last_selected = ()
        self.last_local_actions = (None,) * self.incident_count
        self.last_incident_rewards = (0.0,) * self.incident_count
        self.finished = False
        return self.observation(), self.info()

    def encode(self, incident_ids):
        subset = tuple(sorted(incident_ids))
        if subset not in self._indices:
            raise ValueError("Unknown or duplicated incident identifiers")
        return self._indices[subset]

    def decode(self, action):
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.subsets):
            raise ValueError("Unknown global allocation action")
        return self.subsets[int(action)]

    def available_incidents(self):
        return np.asarray([not env.finished and env.action_mask()[1:].any()
                           for env in self.incidents], dtype=bool)

    def action_mask(self):
        available = self.available_incidents()
        return (self.incidence.sum(axis=1) <= self.team_capacity) & ~(self.incidence & ~available).any(axis=1)

    def observation(self):
        observations = [env.observation() for env in self.incidents]
        return {
            "maps": np.stack([o["map"] for o in observations]),
            "local_context": np.stack([o["context"] for o in observations]),
            "eligibility": np.stack([env.eligibility() for env in self.incidents]).astype(np.float32),
            "context": np.array([self.team_capacity/self.incident_count,
                                  1-self.step_number/self.local_config.max_steps], dtype=np.float32),
        }

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset before stepping a finished global episode")
        selected = self.decode(action)
        if not self.action_mask()[action]:
            raise ValueError("Global allocation violates team capacity or incident availability")
        # Policies see the same pre-transition time. Validate every result before
        # mutating any incident, so an invalid late policy action is atomic.
        local_actions = [None] * self.incident_count
        for i, env in enumerate(self.incidents):
            if not env.finished:
                local_actions[i] = self.local_policy.act(env.observation(), env.action_mask()) if i in selected else 0
                env.validate_action(local_actions[i])
        for env, local_action in zip(self.incidents, local_actions):
            if local_action is not None:
                env.construct(local_action)
        rewards = np.zeros(self.incident_count, dtype=float)
        constructions = 0
        for i, env in enumerate(self.incidents):
            if local_actions[i] is not None:
                _, rewards[i], _, _, local_info = env.advance_fire()
                constructions += local_info["constructed_count"]
        self.step_number += 1
        reward = float(rewards.mean())
        self.total_reward += reward
        self.total_assignments += len(selected)
        self.total_constructions += constructions
        self.last_selected = selected
        self.last_local_actions = tuple(local_actions)
        self.last_incident_rewards = tuple(float(r) for r in rewards)
        terminated = not any(np.any(env.fire_age > 0) for env in self.incidents)
        truncated = self.step_number >= self.local_config.max_steps and not terminated
        self.finished = terminated or truncated
        info = self.info()
        info.update(terminated=terminated, truncated=truncated, constructed_this_step=constructions,
                    reward=reward, invalid_action=False)
        return self.observation(), reward, terminated, truncated, info

    def info(self):
        vegetation = sum(int(env.vegetation.sum()) for env in self.incidents)
        burned = sum(int(np.count_nonzero(env.ever_burned & env.vegetation)) for env in self.incidents)
        cleared = sum(int(np.count_nonzero(env.firebreak & env.vegetation)) for env in self.incidents)
        active = [int(np.count_nonzero(env.fire_age)) for env in self.incidents]
        preserved = [int(np.count_nonzero(env.vegetation & ~env.ever_burned & ~env.firebreak))
                     for env in self.incidents]
        extinct = [count == 0 for count in active]
        return {
            "action_mask": self.action_mask(), "episode_seed": self.episode_seed,
            "incident_seeds": self.incident_seeds, "team_capacity": self.team_capacity,
            "steps": self.step_number, "incident_ticks": [env.step_number for env in self.incidents],
            "burned_cells": burned, "cleared_cells": cleared,
            "burned_fraction": burned/vegetation, "cleared_fraction": cleared/vegetation,
            "affected_fraction": (burned+cleared)/vegetation,
            "extinguished_count": sum(extinct),
            # Operational containment: fire extinct AND some unburned uncleared
            # vegetation survives. Burning out the entire forest is not success.
            "contained_count": sum(e and p > 0 for e, p in zip(extinct, preserved)),
            "contained_fraction": sum(e and p > 0 for e, p in zip(extinct, preserved))/self.incident_count,
            "active_fire_cells": sum(active), "incident_active_fire": active,
            "assigned_team_intervals": self.total_assignments,
            "team_utilization": self.total_assignments/max(1, self.team_capacity*self.step_number),
            "construction_utilization": self.total_constructions/max(1, self.team_capacity*self.step_number),
            "selected_incidents": self.last_selected, "local_actions": self.last_local_actions,
            "incident_rewards": self.last_incident_rewards, "episode_return": self.total_reward,
        }

    def render(self):
        return np.stack([env.render() for env in self.incidents])
