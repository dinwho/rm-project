"""Synchronous, eight-neighbor stochastic spread without spotting or wraparound."""

import numpy as np


NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0))


def shifted(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    result = np.zeros_like(array)
    n, m = array.shape
    source_rows = slice(max(0, -dr), min(n, n - dr))
    source_cols = slice(max(0, -dc), min(m, m - dc))
    target_rows = slice(max(0, dr), min(n, n + dr))
    target_cols = slice(max(0, dc), min(m, m + dc))
    result[target_rows, target_cols] = array[source_rows, source_cols]
    return result


def ignition_probability(burning: np.ndarray, fuel: np.ndarray, wind: np.ndarray,
                         base_spread: float, wind_bias: float) -> np.ndarray:
    survival = np.ones_like(fuel, dtype=np.float32)
    for dr, dc in NEIGHBORS:
        distance = np.hypot(dr, dc)
        alignment = (dr * wind[0] + dc * wind[1]) / distance
        probability = np.clip(base_spread * fuel * np.exp(wind_bias * alignment) / distance, 0, 0.95)
        survival *= 1.0 - shifted(burning, dr, dc) * probability
    return 1.0 - survival
