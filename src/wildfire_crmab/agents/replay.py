"""Uniform replay storing observations and both allocation masks."""
import numpy as np
import torch
from .observations import observation_shapes, checked_observation, checked_mask

class ReplayBuffer:
    """Replay copies observations and retains both feasibility masks for auditing."""

    def __init__(self, capacity: int, observation_space, action_count: int, seed: int = 0):
        if capacity < 1 or action_count < 1:
            raise ValueError("Replay capacity and action_count must be positive")
        self.capacity, self.action_count = int(capacity), int(action_count)
        self.shapes = observation_shapes(observation_space)
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.observations = {key: np.empty((capacity, *shape), np.float32)
                             for key, shape in self.shapes.items()}
        self.next_observations = {key: np.empty_like(value)
                                  for key, value in self.observations.items()}
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)
        self.masks = np.empty((capacity, action_count), bool)
        self.next_masks = np.empty_like(self.masks)

    def add(self, observation, action: int, reward: float, next_observation,
            done: bool, current_mask, next_mask):
        current = checked_observation(observation, self.shapes)
        successor = checked_observation(next_observation, self.shapes)
        mask = checked_mask(current_mask, self.action_count)
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("done must be a boolean finite-horizon boundary")
        successor_mask = checked_mask(next_mask, self.action_count, allow_empty=bool(done))
        if (not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_))
                or not 0 <= action < self.action_count or not mask[action]):
            raise ValueError("Replay cannot contain an infeasible action")
        if not np.isfinite(reward):
            raise ValueError("Reward must be finite")
        index = self.position
        for key in self.shapes:
            self.observations[key][index] = current[key]
            self.next_observations[key][index] = successor[key]
        self.actions[index], self.rewards[index], self.done[index] = action, reward, done
        self.masks[index], self.next_masks[index] = mask, successor_mask
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device="cpu") -> dict:
        if batch_size < 1 or self.size == 0:
            raise ValueError("Cannot sample an empty replay or nonpositive batch")
        indices = self.rng.integers(self.size, size=batch_size)
        batch = {"observations": {}, "next_observations": {}}
        for group in ("observations", "next_observations"):
            batch[group] = {key: torch.as_tensor(value[indices], device=device)
                            for key, value in getattr(self, group).items()}
        for key in ("actions", "rewards", "done", "masks", "next_masks"):
            batch[key] = torch.as_tensor(getattr(self, key)[indices], device=device)
        return batch

