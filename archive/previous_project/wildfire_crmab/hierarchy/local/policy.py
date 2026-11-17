"""Masked PPO construction with a small spatial CNN and strict checkpoint identity."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wildfire_crmab import EnvironmentConfig, Scenario, WildfireEnv

FORMAT = "hierarchical-local-maskable-ppo-v1"
SCHEMA = "maps6-eligibility1-context4-cell-plus-wait-v1"


def canonical(value):
    return json.loads(json.dumps(value))


def local_observation(observation, mask):
    n = observation["map"].shape[-1]
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (n*n+1,) or not mask.any():
        raise ValueError("Local constructor requires a nonempty cell-plus-wait boolean mask")
    return {"map": np.asarray(observation["map"], dtype=np.float32),
            "eligible": mask[1:].reshape(1, n, n).astype(np.float32),
            "context": np.asarray(observation["context"], dtype=np.float32)}


class LocalConstructionEnv(gym.Env):
    """Train on active and passive intervals; unavailable service forces wait.

    The random service process exposes the constructor to interrupted construction.
    At inference the allocator decides service; no synthetic outage is applied.
    Horizon is terminal for the explicitly finite objective, avoiding SB3 timeout
    bootstrapping beyond the observed horizon.
    """
    def __init__(self, config, service_probability=0.5, reward_scale=10.0):
        super().__init__()
        if config.placement_mode != "cells" or config.capacity != 1:
            raise ValueError("Local constructor requires cell mode and capacity one")
        if not 0 <= service_probability <= 1 or not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("Invalid service probability or reward scale")
        self.base = WildfireEnv(config)
        self.config = config
        self.service_probability, self.reward_scale = service_probability, reward_scale
        self.action_space = self.base.action_space
        self.observation_space = spaces.Dict({
            "map": self.base.observation_space["map"],
            "eligible": spaces.Box(0, 1, (1, config.grid_size, config.grid_size), np.float32),
            "context": self.base.observation_space["context"],
        })
        self.available = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = (options or {}).get("scenario")
        if scenario is None:
            scenario = Scenario(int(self.np_random.integers(1_000_000_000)), self.config.initial_budget)
        obs, info = self.base.reset(options={"scenario": scenario})
        self.service_rng = np.random.default_rng(np.random.SeedSequence([scenario.seed, 835732]))
        self.available = bool(self.service_rng.random() < self.service_probability)
        return local_observation(obs, self.action_masks()), {**info, "service_available": self.available}

    def action_masks(self):
        mask = self.base.action_mask().copy()
        if not self.available:
            mask[1:] = False
        return mask

    def step(self, action):
        self.base.allocations.decode(action)
        if not self.action_masks()[action]:
            raise ValueError("Local action violates eligibility or service availability")
        obs, reward, terminated, truncated, info = self.base.step(action)
        available = self.available
        self.available = bool(self.service_rng.random() < self.service_probability)
        info = {**info, "unscaled_reward": reward, "finite_horizon_reached": truncated,
                "service_was_available": available, "service_available": self.available}
        return local_observation(obs, self.action_masks()), reward*self.reward_scale, terminated or truncated, False, info


class SmallFireCNN(BaseFeaturesExtractor):
    """Preserve spatial locations on small grids; include eligibility and wind/time."""
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n = observation_space["map"].shape[-1]
        self.spatial = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.combine = nn.Sequential(nn.Linear(32*n*n+4, features_dim), nn.ReLU())

    def forward(self, observations):
        maps = torch.cat((observations["map"], observations["eligible"]), dim=1)
        return self.combine(torch.cat((self.spatial(maps), observations["context"]), dim=1))


def save_checkpoint(model, path, config, metadata=None):
    path = Path(path).with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    result = {"format": FORMAT, "schema": SCHEMA, "environment": asdict(config),
              "action_count": config.grid_size**2+1, "training_steps": int(model.num_timesteps),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    return result


class FrozenLocalPolicy:
    """Immutable inference adapter. Only load this project's trusted local artifacts."""
    def __init__(self, model, metadata):
        self.model, self.metadata = model, metadata
        self.sha256 = self.checkpoint_sha256 = metadata["sha256"]
        self.model.policy.set_training_mode(False)
        for parameter in self.model.policy.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def load(cls, checkpoint, expected_config):
        path = Path(checkpoint).with_suffix(".zip")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if (metadata.get("format") != FORMAT or metadata.get("schema") != SCHEMA or
                metadata.get("environment") != canonical(asdict(expected_config)) or
                metadata.get("action_count") != expected_config.grid_size**2+1):
            raise ValueError("Local checkpoint environment/observation/action mismatch")
        if metadata.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Local checkpoint checksum mismatch")
        model = MaskablePPO.load(path, device="cpu")
        if model.action_space.n != metadata["action_count"]:
            raise ValueError("Local checkpoint action head mismatch")
        expected = LocalConstructionEnv(expected_config).observation_space
        if model.observation_space != expected:
            raise ValueError("Local checkpoint observation space mismatch")
        return cls(model, metadata)

    def act(self, observation, mask):
        inputs = local_observation(observation, mask)
        with torch.no_grad():
            action, _ = self.model.predict(inputs, action_masks=mask, deterministic=True)
        action = int(action)
        if not mask[action]:
            raise RuntimeError("Masked PPO selected an invalid construction")
        return action
