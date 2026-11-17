"""Joint-allocation value network and feasibility-aware Bellman targets."""
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    """Joint field, persistent arm identity, and resource/time context -> Q(o,A)."""

    def __init__(self, shapes: dict, action_count: int):
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
        )
        self.arm_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(int(np.prod(shapes["arms"])), 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 3 * 3 + 64 + 4, 128), nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(torch.cat([
            self.map_encoder(observations["map"]),
            self.arm_encoder(observations["arms"]), observations["context"],
        ], dim=1))



def bellman_targets(rewards: torch.Tensor, done: torch.Tensor,
                    next_q: torch.Tensor, next_masks: torch.Tensor,
                    gamma: float) -> torch.Tensor:
    """Finite-horizon targets; terminal rows need no feasible successor action.

    Only live rows are maximized. This remains finite even if a terminal mask is
    empty or the unused terminal Q values are NaN; there is no 0 * -inf operation.
    A nonterminal empty mask is a programming error, never interpreted as zero.
    """
    if done.dtype != torch.bool or next_masks.dtype != torch.bool:
        raise ValueError("done and next_masks must be boolean")
    if rewards.ndim != 1 or done.shape != rewards.shape:
        raise ValueError("Expected matching reward and done vectors")
    if next_q.ndim != 2 or next_q.shape != next_masks.shape or len(next_q) != len(rewards):
        raise ValueError("Successor Q and mask shapes do not match the batch")
    if not 0 < gamma <= 1 or not torch.isfinite(rewards).all():
        raise ValueError("Discount must be in (0,1] and rewards finite")
    result = rewards.clone()
    live = ~done
    if live.any():
        masks = next_masks[live]
        if not masks.any(dim=1).all():
            raise ValueError("Nonterminal successor has no feasible allocation")
        values = next_q[live].masked_fill(~masks, -torch.inf).max(dim=1).values
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite feasible successor Q value")
        result[live] += gamma * values
    return result

