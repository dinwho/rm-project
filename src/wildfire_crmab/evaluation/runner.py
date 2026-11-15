"""Shared observation-only policy rollout and physical outcome accounting."""
from time import perf_counter
import numpy as np

def run_episode(env, policy, scenario, gamma=.99):
    observation, info = env.reset(options={"scenario": scenario})
    discounted = 0.0
    seconds = 0.0
    steps = 0
    multi = 0
    # Wind strength is a reporting label only; policy still receives its own observation.
    wind_speed = float(np.linalg.norm(env.wind))
    while not env.finished:
        start = perf_counter()
        action = policy.act(observation, info["action_mask"])
        seconds += perf_counter() - start
        if not info["action_mask"][action]:
            raise AssertionError("Policy selected an infeasible allocation")
        multi += len(env.allocations.decode(action)) > 1
        observation, reward, terminated, truncated, info = env.step(action)
        discounted += gamma**steps * reward
        steps += 1
    return dict(scenario_id=scenario.scenario_id, world_seed=scenario.seed, budget=scenario.budget,
                condition="weaker" if wind_speed < .6 else "stronger",
                discounted_return=discounted, episode_return=info["episode_return"],
                burned_fraction=info["burned_fraction"], affected_fraction=info["affected_fraction"],
                settlement_survived=info["settlement_survived"], settlement_damage=info["settlement_damage"],
                budget_used=info["budget_used"], steps=steps, decision_ms=1000*seconds/steps,
                truncated=int(truncated), active_fire_cells=info["active_fire_cells"],
                multi_construction_intervals=int(multi), invalid_actions=0)

