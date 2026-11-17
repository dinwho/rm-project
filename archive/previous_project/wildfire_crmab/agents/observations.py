"""Shared observation and mask validation at the learning boundary."""
from typing import Mapping
import numpy as np
OBSERVATION_KEYS = ("map", "arms", "context")

def observation_shapes(observation_space) -> dict[str, tuple[int, ...]]:
    """Accept a Gymnasium Dict space or a serialized mapping of shape tuples."""
    source = getattr(observation_space, "spaces", observation_space)
    if set(source) != set(OBSERVATION_KEYS):
        raise ValueError(f"Expected observation keys {OBSERVATION_KEYS}")
    shapes = {}
    for key in OBSERVATION_KEYS:
        value = source[key]
        shapes[key] = tuple(int(x) for x in getattr(value, "shape", value))
        if not shapes[key] or any(d <= 0 for d in shapes[key]):
            raise ValueError(f"Invalid {key} shape")
    if len(shapes["map"]) != 3 or shapes["map"][0] != 6:
        raise ValueError("Map must have six channels and two spatial dimensions")
    if len(shapes["arms"]) != 2 or shapes["context"] != (4,):
        raise ValueError("Expected an arm-feature matrix and four context scalars")
    return shapes



def checked_observation(observation: Mapping, shapes: dict) -> dict[str, np.ndarray]:
    if set(observation) != set(shapes):
        raise ValueError("Observation schema does not match the network")
    arrays = {}
    for key, shape in shapes.items():
        value = np.asarray(observation[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Wrong shape or nonfinite values in {key}")
        arrays[key] = value
    return arrays



def checked_mask(mask, action_count: int, *, allow_empty: bool = False) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != (action_count,) or mask.dtype != np.bool_:
        raise ValueError("Action mask must be a boolean vector of action_count entries")
    if not allow_empty and not mask.any():
        raise ValueError("A nonterminal decision requires a feasible action")
    return mask

