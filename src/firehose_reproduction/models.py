"""Original Firehose CNN and tensor-only conversion of published PPO checkpoints.

Architecture follows aidan-curtis/firehose, GPL-3.0, pinned in UPSTREAM_COMMIT.
No pickle/cloudpickle objects from upstream checkpoint metadata are executed.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch import nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

UPSTREAM_COMMIT = "e49a52a9a987a1585d25ddca2f711222ca500ffb"
PUBLIC_MODELS = {
    "20x20-fixed-ppo-maskable.zip": ("Sub20x20", "fixed", "ec9475239abb141ab627e6e3d07f14e49da2b7a71020699fed55504db0bf9a5c"),
    "20x20-random-20m-ppo-maskable.zip": ("Sub20x20", "random", "08cb56f17fdb99f2f17ac542008963324a75a724fe2d5cf032537e7381304dec"),
    "40x40-fixed-ppo-maskable.zip": ("Sub40x40", "fixed", "caaa82a68810ecdebfe48a4886ed215c76d090d7777b57bb1a6a3bacbe3dc1dc"),
    "40x40-random-ppo-maskable.zip": ("Sub40x40", "random", "dfc7debb8788f1cfba289481bf21ff220075776a61a2b9c8c1c85bbdba3729ae"),
}
FORMAT = "firehose-original-rgb-maskable-ppo-v1"


class PaddedNatureCNN(BaseFeaturesExtractor):
    """Exactly the published Firehose convolution and feature dimensions.

    The upstream class name says 'padded', but its convolutions have no padding.
    RGB normalization is performed once by Stable Baselines before this extractor.
    """
    def __init__(self, observation_space: spaces.Box, features_dim=512):
        if (observation_space.dtype != np.uint8 or len(observation_space.shape) != 3
                or observation_space.shape[0] != 3):
            raise ValueError("Original Firehose requires channels-first uint8 RGB")
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            count = self.cnn(torch.zeros((1, *observation_space.shape))).shape[1]
        self.linear = nn.Sequential(nn.Linear(count, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


def make_model(env, *, seed=1, n_steps=2048, batch_size=64, n_epochs=10, verbose=0):
    """Published CNN / Maskable PPO recipe with explicit upstream defaults."""
    torch.set_num_threads(1)
    return MaskablePPO(
        "CnnPolicy", env, learning_rate=3e-4, n_steps=n_steps, batch_size=batch_size,
        n_epochs=n_epochs, gamma=.9, gae_lambda=.95, clip_range=.2,
        ent_coef=0., vf_coef=.5, max_grad_norm=.5, seed=seed,
        policy_kwargs={"features_extractor_class": PaddedNatureCNN,
                       "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                       "normalize_images": True}, device="cpu", verbose=verbose)


def _state_from_zip(path):
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
        state = torch.load(io.BytesIO(archive.read("policy.pth")),
                           map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected tensor-only policy state")
    return data, state


def _load_policy_state(model, state):
    """Alias the shared extractor for SB3 versions that register it three times.

    This adds no learned parameters. Every incoming tensor is required, every
    target tensor shape is checked and duplicated aliases must be exactly equal.
    """
    target = model.policy.state_dict()
    mapped = {}
    used = set()
    for key, expected in target.items():
        source = key
        for prefix in ("pi_features_extractor.", "vf_features_extractor."):
            if key.startswith(prefix):
                source = "features_extractor." + key[len(prefix):]
        if source not in state or state[source].shape != expected.shape:
            raise ValueError(f"Checkpoint tensor missing or incompatible: {source}")
        if key in state and not torch.equal(state[key], state[source]):
            raise ValueError(f"Checkpoint shared feature alias disagrees: {key}")
        mapped[key] = state[source]
        used.add(source)
        if key in state:
            used.add(key)
    if set(state) != used:
        raise ValueError(f"Unexpected checkpoint tensors: {sorted(set(state)-used)}")
    model.policy.load_state_dict(mapped, strict=True)
    model.policy.set_training_mode(False)


def load_pretrained(path, env, *, expected_map=None):
    """Load a hash-identified published policy; reject changed maps/observations.

    This is inference conversion, not restoration of an optimizer or training RNG.
    """
    path = Path(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    matches = [(name, entry) for name, entry in PUBLIC_MODELS.items() if entry[2] == checksum]
    if not matches:
        raise ValueError("Checkpoint is not one of the pinned public Firehose artifacts")
    name, (map_name, ignition_mode, _) = matches[0]
    config = getattr(env, "config", None)
    actual_map = expected_map or getattr(config, "map_name", None) or getattr(env, "map_name", None)
    if actual_map != map_name:
        raise ValueError(f"Checkpoint was trained on {map_name}, not {actual_map}")
    expected_size = 20 if map_name == "Sub20x20" else 40
    if (env.observation_space.shape != (3, expected_size, expected_size)
            or env.observation_space.dtype != np.uint8
            or env.action_space.n != expected_size**2):
        raise ValueError("Original checkpoint observation or action space mismatch")
    if config is not None and getattr(config, "reward", "FireSizeReward") != "FireSizeReward":
        raise ValueError("Published policy does not implement a different reward task")
    data, state = _state_from_zip(path)
    if (data.get("observation_space", {}).get("_shape") != [3, expected_size, expected_size]
            or data.get("action_space", {}).get("n") != expected_size**2
            or data.get("gamma") != .9):
        raise ValueError("Public checkpoint metadata mismatch")
    model = make_model(env)
    _load_policy_state(model, state)
    model.num_timesteps = int(data["num_timesteps"])
    model.reproduction_metadata = {
        "format": FORMAT, "origin": "published_upstream_weights",
        "reference_commit": UPSTREAM_COMMIT, "checkpoint_sha256": checksum,
        "checkpoint_name": name, "map_name": map_name, "ignition_training": ignition_mode,
        "reward": "FireSizeReward", "training_steps": model.num_timesteps,
        "observation": "uint8-RGB-CHW", "action_count": expected_size**2,
        "gamma": .9, "conversion": "tensor-only; shared-extractor aliases verified"}
    return model


def save_model(model, path, task, *, metadata=None):
    path = Path(path).with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    record = {"format": FORMAT, "origin": "trained_here", "task": task,
              "reference_commit": UPSTREAM_COMMIT, "training_steps": int(model.num_timesteps),
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              **(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
    return path


def load_trained(path, env, *, expected_task):
    path = Path(path).with_suffix(".zip")
    record = json.loads(path.with_suffix(".json").read_text())
    if (record.get("format") != FORMAT or record.get("task") != expected_task
            or record.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Saved policy task or checksum mismatch")
    _, state = _state_from_zip(path)
    model = make_model(env)
    _load_policy_state(model, state)
    model.num_timesteps = int(record["training_steps"])
    model.reproduction_metadata = record
    return model
