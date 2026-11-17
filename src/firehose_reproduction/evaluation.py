"""Matched original-Firehose policy comparisons without changing simulator rules."""
from .recording import record


def compare(seed=123, ignition=372):
    """One fixed-scenario illustration, not a paper-wide statistical result."""
    return [record(name, seed=seed, ignition=ignition)
            for name in ("ppo", "no_intervention", "random", "min_l2", "max_l2")]
