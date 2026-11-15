"""Generate inspectable replays using the research environment and real policies."""
from dataclasses import asdict
from pathlib import Path
import csv
import numpy as np
import torch

from wildfire_crmab import EnvironmentConfig, WildfireEnv, Scenario
from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.baselines import make_baseline, BASELINE_NAMES
from wildfire_crmab.agents.checkpoints import catalogue_metadata

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("dqn", "dqn_no_wind", *BASELINE_NAMES)


def rollout(policy_name="dqn", seed=42, budget=6, wind=None):
    if policy_name not in POLICIES or not 0 <= seed < 2**31 or budget not in (0, 2, 4, 6):
        raise ValueError("Choose a listed policy, nonnegative seed, and budget 0, 2, 4 or 6")
    torch.set_num_threads(1)
    model_info = None
    if policy_name.startswith("dqn"):
        path = next((ROOT/"artifacts"/run/"models"/policy_name/"seed_11"/"best.pt"
                     for run in ("pilot", "smoke")
                     if (ROOT/"artifacts"/run/"models"/policy_name/"seed_11"/"best.pt").exists()), None)
        if path is None:
            raise FileNotFoundError("Train the smoke configuration before opening a DQN demo")
        policy, metadata = DQNAgent.load(path)
        config = EnvironmentConfig(**metadata["environment"])
        env = WildfireEnv(config, metadata["observe_wind"], metadata["budget_scale"])
        if any(metadata.get(k) != v for k, v in catalogue_metadata(env).items()):
            raise ValueError("Model and simulator catalogue do not match")
        model_info = dict(checkpoint=str(path.relative_to(ROOT)), training_step=metadata["step"],
                          parameter_count=sum(p.numel() for p in policy.online.parameters()),
                          architecture=str(policy.online), gamma=policy.gamma)
    else:
        config = EnvironmentConfig()
        env = WildfireEnv(config, policy_name == "wind_heuristic", budget_scale=6)
        policy = make_baseline(policy_name, config, seed+123)
    obs, info = env.reset(options={"scenario": Scenario(seed, budget, wind)})
    frames = []
    previous_reward = 0.0
    discounted = 0.0
    while True:
        mask = info["action_mask"]
        action = None if env.finished else policy.act(obs, mask)
        scores = []
        if model_info and not env.finished:
            with torch.no_grad():
                q = policy.online({k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()})[0].numpy()
            valid = np.flatnonzero(mask)
            top = valid[np.argsort(-q[valid], kind="stable")[:5]]
            scores = [dict(action=int(j), arms=list(env.allocations.decode(int(j))), q=float(q[j])) for j in top]
        frames.append(dict(t=env.step_number, rgb=env.render().tolist(), built=env.built.tolist(),
                           eligible=env.eligibility().tolist(), arm_features=obs["arms"].tolist(),
                           wind_observation=obs["context"][:2].tolist(),
                           allocation=[] if action is None else list(env.allocations.decode(action)),
                           action=action, feasible=int(mask.sum()), q_values=scores,
                           budget=env.budget, burned=info["burned_fraction"], affected=info["affected_fraction"],
                           asset_damage=info["settlement_damage"], active_fire=info["active_fire_cells"],
                           previous_reward=previous_reward, discounted_return=discounted,
                           finished=env.finished))
        if env.finished:
            break
        obs, previous_reward, terminated, truncated, info = env.step(action)
        discounted += .99**(env.step_number-1)*previous_reward
    return dict(policy=policy_name, seed=seed, initial_budget=budget, config=asdict(config),
                arms=[asdict(s) for s in env.arms.segments], action_count=int(env.action_space.n),
                actual_wind=env.wind.tolist(), model=model_info, frames=frames,
                ending="Horizon reached with active fire" if truncated else "Fire extinguished")


def experiment_results():
    path = ROOT/"artifacts"/"pilot"/"evaluation"/"summary.csv"
    if not path.exists():
        return dict(status="Pilot still running", rows=[], learning=[])
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["group"] == "all"]
    learning = []
    for method in ("dqn", "dqn_no_wind"):
        for seed in (11, 22, 33):
            path = ROOT/"artifacts"/"pilot"/"models"/method/f"seed_{seed}"/"validation.csv"
            with path.open() as handle:
                learning.extend(dict(method=method, seed=seed, **r) for r in csv.DictReader(handle))
    return dict(status="Pilot: 3 training seeds · 10,000 intervals each · 24 held-out worlds × 3 budgets", rows=rows, learning=learning)
