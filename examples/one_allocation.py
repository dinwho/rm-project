"""One legal cell construction, followed by passive evolution. Run from research/."""
import numpy as np
from wildfire_crmab import WildfireEnv, Scenario

env = WildfireEnv()
observation, info = env.reset(options={"scenario": Scenario(seed=42, budget=None)})
action = int(np.flatnonzero(info["action_mask"])[1])  # First eligible cell; 0 is wait.
active_arms = env.allocations.decode(action)
observation, reward, terminated, truncated, info = env.step(action)

print(f"Candidate arms: {env.arms.count}; allocation actions: {env.action_space.n}")
print(f"Constructed arm: {active_arms}; physical intervals elapsed: {env.step_number}")
print(f"Resources spent: {info['budget_used']}; episode cap: {env.budget} (None = unlimited)")
print(f"Persistent build flags: {env.built[list(active_arms)]}")
print(f"Interval reward: {reward:.4f}")
print(f"Observation shapes: {{ {', '.join(f'{k}: {v.shape}' for k, v in observation.items())} }}")
print("Wait actions now leave all arms passive while fire evolves.")
while not env.finished:
    env.step(0)
assert env.built[list(active_arms)].all()
print(f"Final burned-or-cleared fraction: {env.info()['affected_fraction']:.3f}")
