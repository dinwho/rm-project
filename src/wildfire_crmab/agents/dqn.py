"""Masked vanilla DQN over complete simultaneous allocation actions."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .network import QNetwork, bellman_targets
from .observations import observation_shapes, checked_observation, checked_mask
CHECKPOINT_FORMAT = "wildfire-crmab-allocation-dqn-v1"

class DQNAgent:
    def __init__(self, observation_space, action_count: int, seed: int,
                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 device: str = "cpu", torch_threads: int = 1):
        if action_count < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
            raise ValueError("Invalid action_count or learning rate")
        if not 0 < gamma <= 1 or torch_threads < 1:
            raise ValueError("Invalid discount or thread count")
        self.shapes = observation_shapes(observation_space)
        self.action_count, self.seed = int(action_count), int(seed)
        self.gamma, self.learning_rate = float(gamma), float(learning_rate)
        self.device = torch.device(device)
        self.torch_threads = torch_threads
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(torch_threads)
        self.online = QNetwork(self.shapes, self.action_count).to(self.device)
        self.target = QNetwork(self.shapes, self.action_count).to(self.device)
        self.update_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    def act(self, observation, action_mask, epsilon: float = 0.0) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        arrays = checked_observation(observation, self.shapes)
        mask = checked_mask(action_mask, self.action_count)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            batch = {key: torch.as_tensor(value, device=self.device).unsqueeze(0)
                     for key, value in arrays.items()}
            q_values = self.online(batch)[0]
            feasible = torch.as_tensor(mask, device=self.device)
            if not torch.isfinite(q_values[feasible]).all():
                raise ValueError("Network produced nonfinite feasible Q values")
            return int(q_values.masked_fill(~feasible, -torch.inf).argmax().item())

    def learn(self, batch: dict) -> float:
        if not batch["masks"].gather(1, batch["actions"].unsqueeze(1)).all():
            raise ValueError("Learning batch contains an infeasible current action")
        values = self.online(batch["observations"])
        chosen = values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_values = self.target(batch["next_observations"])
            targets = bellman_targets(batch["rewards"], batch["done"], next_values,
                                      batch["next_masks"], self.gamma)
        loss = nn.functional.smooth_l1_loss(chosen, targets)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path, metadata: dict | None = None):
        """Save an inference checkpoint; optimizer/replay resume is not implemented.

        The caller should include environment config, catalogue identity, wind
        visibility and training step in metadata to identify the allocation task.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": CHECKPOINT_FORMAT, "shapes": self.shapes,
            "action_count": self.action_count, "seed": self.seed,
            "learning_rate": self.learning_rate, "gamma": self.gamma,
            "torch_threads": self.torch_threads,
            "state_dict": self.online.state_dict(), "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple["DQNAgent", dict]:
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Checkpoint is not a compatible CRMAB allocation DQN")
        agent = cls(saved["shapes"], saved["action_count"], saved["seed"],
                    saved["learning_rate"], saved["gamma"], device,
                    saved["torch_threads"])
        agent.online.load_state_dict(saved["state_dict"])
        agent.online.eval()
        agent.update_target()
        return agent, saved["metadata"]

