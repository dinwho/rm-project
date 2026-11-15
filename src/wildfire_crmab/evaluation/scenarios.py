"""Fixed held-out seed namespaces, with common worlds across budgets."""
from wildfire_crmab.environment.scenario import Scenario

def scenario_suite(split, worlds, budgets):
    offset = {"validation": 1_000_000_000, "test": 2_000_000_000}[split]
    return [Scenario(seed=offset + i, budget=b) for i in range(worlds) for b in budgets]

