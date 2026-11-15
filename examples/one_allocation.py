"""Run after installing the project: python examples/one_allocation.py."""

import numpy as np
from wildfire_crmab import WildfireEnv, Scenario

env = WildfireEnv()
observation, info = env.reset(options={"scenario": Scenario(seed=42, budget=6)})

# Find one feasible allocation containing two simultaneously active arms.
action = next(int(j) for j in np.flatnonzero(info["action_mask"])
              if len(env.allocations.decode(int(j))) == 2)
active_arms = env.allocations.decode(action)
before = env.arm_states()
observation, reward, terminated, truncated, info = env.step(action)

print(f"Candidate arms: {env.arms.count}; complete allocation actions: {env.action_space.n}")
print(f"Selected subset: {active_arms}")
print(f"Physical intervals elapsed: {env.step_number} (two segments, one interval)")
print(f"Material spent: {info['resources_spent']}; remaining: {env.budget}")
print(f"Persistent build flags: {env.built[ list(active_arms) ]}")
print(f"Interval reward: {reward:.4f}")
print(f"Observation shapes: { {k: v.shape for k, v in observation.items()} }")
print("Subsequent empty allocations keep every candidate passive while fire evolves.")
while not env.finished:
    env.step(0)
print(f"Final burned-or-cleared fraction: {env.info()['affected_fraction']:.3f}")
assert env.built[list(active_arms)].all()
