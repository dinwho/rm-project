"""Inspect real cell-construction policies; never substitute legacy segment models."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import numpy as np
import torch

from wildfire_crmab import EnvironmentConfig, WildfireEnv, Scenario
from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.baselines import make_baseline, BASELINE_NAMES
from wildfire_crmab.agents.checkpoints import catalogue_metadata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "firehose"
POLICIES = ("dqn", *BASELINE_NAMES)


def demo_config():
    """All demo policies face the same fixed ignition, vegetation and wind."""
    with (ROOT / "configs" / "firehose.json").open() as handle:
        return EnvironmentConfig(**json.load(handle)["environment"])


def rollout(policy_name="wind_heuristic", seed=42, budget=None, wind=None):
    if policy_name not in POLICIES or not 0 <= seed < 2**31 or budget not in (None, 0, 2, 4, 6):
        raise ValueError("Choose a listed policy, nonnegative seed, and no episode cap or budget 0, 2, 4, 6")
    torch.set_num_threads(1)
    model_info = None
    config = demo_config()
    if policy_name.startswith("dqn"):
        if budget is not None:
            raise ValueError("This DQN was trained without an episode cap; choose No episode cap or a baseline policy")
        path = RUN / "models" / policy_name / "seed_11" / "best.pt"
        if not path.exists():
            raise FileNotFoundError("Firehose-inspired DQN weights are missing. Run configs/firehose.json into "
                                    "artifacts/firehose, or choose a placement heuristic.")
        policy, metadata = DQNAgent.load(path)
        if metadata["environment"] != asdict(config):
            raise ValueError("Saved DQN does not match configs/firehose.json; use the matching experiment configuration")
        env = WildfireEnv(config, metadata["observe_wind"], metadata["budget_scale"])
        if any(metadata.get(k) != v for k, v in catalogue_metadata(env).items()):
            raise ValueError("Model and simulator catalogue do not match")
        model_info = dict(checkpoint=str(path.relative_to(ROOT)), training_step=metadata["step"],
                          parameter_count=sum(p.numel() for p in policy.online.parameters()),
                          architecture=str(policy.online), gamma=policy.gamma)
    else:
        env = WildfireEnv(config, policy_name == "wind_heuristic", budget_scale=6)
        policy = make_baseline(policy_name, config, seed+123)
    obs, info = env.reset(options={"scenario": Scenario(seed, budget, wind)})
    frames = []
    previous_reward = 0.0
    discounted = 0.0
    truncated = False
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
                           budget=env.budget, resources_used=info["budget_used"],
                           burned=info["burned_fraction"], affected=info["affected_fraction"],
                           asset_damage=info["settlement_damage"], active_fire=info["active_fire_cells"],
                           previous_reward=previous_reward, discounted_return=discounted,
                           finished=env.finished))
        if env.finished:
            break
        obs, previous_reward, terminated, truncated, info = env.step(action)
        gamma = policy.gamma if model_info else .99
        discounted += gamma**(env.step_number-1)*previous_reward
    return dict(policy=policy_name, seed=seed, initial_budget=budget, config=asdict(config),
                arms=[asdict(s) for s in env.arms.segments], action_count=int(env.action_space.n),
                actual_wind=env.wind.tolist(), model=model_info, frames=frames,
                ending="Horizon reached with active fire" if truncated else "Fire extinguished")


def experiment_results():
    path = RUN / "evaluation" / "summary.csv"
    if not path.exists():
        return dict(status="24x24 matched-world results are not available yet.", rows=[], learning=[])
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["group"] == "all"]
    paired_path = RUN / "evaluation" / "paired_differences.csv"
    with paired_path.open() as handle:
        paired = [row for row in csv.DictReader(handle)
                  if row["group"] == "all" and row["metric"] == "discounted_return"]
    curve_path = RUN / "evaluation" / "learning_curve.csv"
    if curve_path.exists():
        with curve_path.open() as handle:
            learning = list(csv.DictReader(handle))
    else:
        learning = []
    return dict(status="Matched held-out worlds; paired fire-spread draws across all policies. "
                       "This is a synthetic mathematical model, not an operational fire simulator.",
              rows=rows, paired=paired, learning=learning)
