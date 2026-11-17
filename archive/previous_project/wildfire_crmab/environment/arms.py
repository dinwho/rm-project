"""Construction-site arms and their evolving physical neighbourhoods."""

from dataclasses import dataclass
import numpy as np

from .config import EnvironmentConfig


@dataclass(frozen=True)
class Segment:
    arm_id: int
    row: int
    col: int
    vertical: bool
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ArmState:
    """Snapshot, not an independent Markov state: neighbours and outside fire matter."""
    arm_id: int
    built: bool
    eligible: bool
    fuel: np.ndarray
    fire_age: np.ndarray
    ever_burned: np.ndarray
    firebreak: np.ndarray


class CandidateArms:
    FEATURE_NAMES = ("built", "eligible", "local_fuel", "local_burning", "local_age",
                     "local_ever_burned", "local_cleared", "row", "col", "vertical")
    feature_count = len(FEATURE_NAMES)

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        half = config.segment_length // 2
        # Every physical cell is a default arm, including boundary anchors. The
        # sparse legacy geometry is explicit so old experiments remain replayable.
        centers = (range(config.grid_size) if config.placement_mode == "cells" else
                   range(half + 1, config.grid_size - half, config.candidate_stride))
        segments = []
        for row in centers:
            for col in centers:
                for vertical in (False, True):
                    cells = tuple((row + (offset if vertical else 0),
                                   col + (0 if vertical else offset))
                                  for offset in range(-half, half + 1))
                    # Length-one horizontal/vertical duplicates are the same opportunity.
                    if config.segment_length == 1 and vertical:
                        continue
                    segments.append(Segment(len(segments), row, col, vertical, cells))
        self.segments = tuple(segments)
        self.count = len(segments)
        self.patches = np.array([s.cells for s in segments], dtype=np.int64)
        self.neighbourhoods = tuple(self._neighbourhood(s) for s in segments)

    def _neighbourhood(self, segment: Segment):
        rows, cols = zip(*segment.cells)
        n = self.config.grid_size
        return (slice(max(0, min(rows)-1), min(n, max(rows)+2)),
                slice(max(0, min(cols)-1), min(n, max(cols)+2)))

    def cells(self, arm_id: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= arm_id < self.count:
            raise ValueError("Unknown arm id")
        return self.patches[arm_id, :, 0], self.patches[arm_id, :, 1]

    def eligible(self, fuel, ever_burned, firebreak, settlement, built) -> np.ndarray:
        buildable = (fuel > 0) & ~ever_burned & ~firebreak & ~settlement
        return ~built & buildable[self.patches[:, :, 0], self.patches[:, :, 1]].all(axis=1)

    def states(self, fuel, fire_age, ever_burned, firebreak, settlement, built) -> tuple[ArmState, ...]:
        eligible = self.eligible(fuel, ever_burned, firebreak, settlement, built)
        return tuple(ArmState(i, bool(built[i]), bool(eligible[i]),
                              fuel[h].copy(), fire_age[h].copy(),
                              ever_burned[h].copy(), firebreak[h].copy())
                     for i, h in enumerate(self.neighbourhoods))

    def features(self, fuel, fire_age, ever_burned, firebreak, settlement, built) -> np.ndarray:
        eligible = self.eligible(fuel, ever_burned, firebreak, settlement, built)
        rows = []
        for i, (segment, h) in enumerate(zip(self.segments, self.neighbourhoods)):
            rows.append((built[i], eligible[i], fuel[h].mean(), (fire_age[h] > 0).mean(),
                         fire_age[h].mean() / self.config.burn_duration,
                         ever_burned[h].mean(), firebreak[h].mean(),
                         segment.row/(self.config.grid_size-1),
                         segment.col/(self.config.grid_size-1), segment.vertical))
        return np.asarray(rows, dtype=np.float32)
