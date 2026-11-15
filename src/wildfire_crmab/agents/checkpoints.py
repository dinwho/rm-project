"""Bind a learned output head to its exact physical allocation catalogue."""
from dataclasses import asdict

def catalogue_metadata(env):
    return {"environment": asdict(env.config), "budget_scale": env.budget_scale,
            "observe_wind": env.observe_wind,
            "allocations": [list(s) for s in env.allocations.subsets],
            "observation_schema": "maps6-arms10-context4-v1"}

