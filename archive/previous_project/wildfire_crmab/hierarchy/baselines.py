"""Allocation strategies use the same frozen local constructor and observations."""
import numpy as np
from wildfire_crmab.environment.spread import ignition_probability


class AllocationBaseline:
    def __init__(self, name, subsets, local_config, seed=0):
        if name not in ("random", "severity", "risk", "no_allocation"):
            raise ValueError("Unknown allocation baseline")
        self.name, self.subsets, self.config = name, tuple(subsets), local_config
        self.rng = np.random.default_rng(seed)

    def act(self, observation, action_mask):
        mask = np.asarray(action_mask)
        if mask.dtype != np.bool_ or mask.shape != (len(self.subsets),) or not mask.any():
            raise ValueError("Invalid global action mask")
        valid = np.flatnonzero(mask)
        if self.name == "random":
            return int(self.rng.choice(valid))
        if self.name == "no_allocation":
            if not mask[0]:
                raise ValueError("Empty allocation must be feasible")
            return 0
        maps = observation["maps"]
        if self.name == "severity":
            scores = maps[:, 1].sum(axis=(1, 2))
        else:
            scores = []
            for field, context in zip(maps, observation["local_context"]):
                fuel, burning, burned = field[0], field[1] > 0, field[3] > 0
                wind = context[:2] if self.config.wind_physics else np.zeros(2)
                risk = ignition_probability(burning, fuel, wind, self.config.base_spread, self.config.wind_bias)
                scores.append(float((risk * ((fuel > 0) & ~burned)).sum()))
            scores = np.asarray(scores)
        # Maximum score; preserve a team on a zero-benefit tie, then stable ID.
        values = np.array([scores[list(self.subsets[a])].sum() for a in valid])
        sizes = np.array([len(self.subsets[a]) for a in valid])
        return int(valid[np.lexsort((valid, sizes, -values))[0]])
