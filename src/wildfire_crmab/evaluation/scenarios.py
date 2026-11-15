"""Held-out random fields shared across policies and budget regimes.

With a fixed layout, ignition and wind in EnvironmentConfig, these seeds vary
only the fire's spread randomness. They do not introduce unseen layouts.
"""
from wildfire_crmab.environment.scenario import Scenario

def scenario_suite(split, worlds, budgets):
    offset = {"validation": 1_000_000_000, "test": 2_000_000_000}[split]
    return [Scenario(seed=offset + i, budget=b) for i in range(worlds) for b in budgets]

