"""Train, validate and separately evaluate the lower-level Maskable PPO policy."""
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from wildfire_crmab import Scenario, WildfireEnv
from wildfire_crmab.agents.baselines import make_baseline
from wildfire_crmab.utils.io import write_csv, write_json
from .policy import FrozenLocalPolicy, LocalConstructionEnv, SmallFireCNN, local_observation, save_checkpoint


class ModelPolicy:
    def __init__(self, model):
        self.model = model

    def act(self, obs, mask):
        action, _ = self.model.predict(local_observation(obs, mask), action_masks=mask, deterministic=True)
        return int(action)


def evaluate_policy(policy, config, seeds, service_probability=1.0, gamma=.99):
    rows = []
    for seed in seeds:
        env = WildfireEnv(config)
        obs, info = env.reset(options={"scenario": Scenario(int(seed), config.initial_budget)})
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 835732]))
        total = discounted = 0.
        service_ticks = 0
        while not env.finished:
            available = rng.random() < service_probability
            mask = info["action_mask"].copy()
            if not available:
                mask[1:] = False
            action = policy.act(obs, mask)
            if not mask[action]:
                raise RuntimeError("Invalid evaluation action")
            service_ticks += int(available)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            discounted += gamma**(env.step_number-1)*reward
        burned, cleared = int(env.ever_burned.sum()), int(env.firebreak.sum())
        remaining = int(((env.fuel > 0) & ~env.ever_burned & ~env.firebreak).sum())
        rows.append(dict(seed=int(seed), service_probability=service_probability, reward=total,
                         discounted_return=discounted, burned_cells=burned, cleared_cells=cleared,
                         affected_fraction=info["affected_fraction"], contained=int(terminated and remaining > 0),
                         extinguished=int(terminated), truncated=int(truncated), steps=env.step_number,
                         service_ticks=service_ticks))
    return rows


class ValidationCallback(BaseCallback):
    def __init__(self, output, config, seeds, every, service_probability, metadata):
        super().__init__()
        self.output, self.config, self.seeds = output, config, seeds
        self.every, self.service_probability, self.metadata = every, service_probability, metadata
        self.rows = []
        self.best = -np.inf

    def measure(self):
        rows = evaluate_policy(ModelPolicy(self.model), self.config, self.seeds, self.service_probability, self.model.gamma)
        score = float(np.mean([row["discounted_return"] for row in rows]))
        record = dict(step=int(self.model.num_timesteps), discounted_return=score,
                      affected_fraction=float(np.mean([r["affected_fraction"] for r in rows])),
                      containment_rate=float(np.mean([r["contained"] for r in rows])))
        self.rows.append(record)
        if self.model.num_timesteps > 0 and score > self.best:
            self.best = score
            save_checkpoint(self.model, self.output/"best.zip", self.config,
                            {**self.metadata, "validation_return": score})
        write_csv(self.output/"validation.csv", self.rows)
        print(f"Local PPO step={self.model.num_timesteps}: validation return={score:.4f}, "
              f"affected={record['affected_fraction']:.3f}", flush=True)

    def _on_training_start(self):
        if not self.rows:
            self.measure()

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self.measure()
        return True

    def _on_training_end(self):
        if not self.rows or self.rows[-1]["step"] != self.num_timesteps:
            self.measure()


def train_local(output, config, steps=131072, seed=11, validation_seeds=None,
                validation_every=8192, service_probability=.5, reward_scale=10.,
                rollout_steps=512, n_epochs=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if validation_seeds is None:
        validation_seeds = list(range(1_100_000_000, 1_100_000_016))
    if not validation_seeds or any(s < 1_000_000_000 for s in validation_seeds):
        raise ValueError("Validation seeds must be outside the training seed range")
    torch.set_num_threads(1)
    metadata = dict(seed=seed, service_probability=service_probability, reward_scale=reward_scale,
                    validation_seeds=validation_seeds, requested_steps=steps,
                    selection="best trained checkpoint by mean discounted validation return",
                    gamma=.99, n_epochs=n_epochs, rollout_steps=rollout_steps,
                    reference_commit="e49a52a9a987a1585d25ddca2f711222ca500ffb")
    write_json(output/"config.json", {"environment": asdict(config), **metadata})
    files = [Path(__file__), Path(__file__).with_name("policy.py")]
    write_json(output/"manifest.json", {"packages": {p: version(p) for p in
               ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")},
               "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "reference_commit": metadata["reference_commit"]})
    for path in files:
        (output/path.name).write_bytes(path.read_bytes())
    env = LocalConstructionEnv(config, service_probability, reward_scale)
    model = MaskablePPO("MultiInputPolicy", env, seed=seed, device="cpu", learning_rate=3e-4,
                       n_steps=rollout_steps, batch_size=64, n_epochs=n_epochs, gamma=.99,
                       gae_lambda=.95, ent_coef=.01, verbose=0,
                       policy_kwargs={"features_extractor_class": SmallFireCNN,
                                      "normalize_images": False, "net_arch": {"pi": [64], "vf": [64]}})
    save_checkpoint(model, output/"initial.zip", config, metadata)
    callback = ValidationCallback(output, config, validation_seeds, validation_every, service_probability, metadata)
    model.learn(total_timesteps=steps, callback=callback)
    save_checkpoint(model, output/"last.zip", config, metadata)
    chosen = json.loads((output/"best.json").read_text())
    write_json(output/"training_summary.json", {"initial_validation": callback.rows[0],
               "selected_step": chosen["training_steps"], "selected_validation_return": callback.best,
               "actual_steps": model.num_timesteps, "checkpoint_sha256": chosen["sha256"]})
    env.close()
    return FrozenLocalPolicy.load(output/"best.zip", config)


def evaluate_local(output, policy, config, seeds=None, service_probabilities=(1., .5)):
    """Holdouts compare construction alone; not global allocation performance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if seeds is None:
        seeds = list(range(2_100_000_000, 2_100_000_064))
    if set(seeds) & set(policy.metadata.get("validation_seeds", [])):
        raise ValueError("Local holdout seeds overlap checkpoint validation seeds")
    rows = []
    for probability in service_probabilities:
        for method in ("local_ppo", "random", "greedy", "wind_heuristic", "no_intervention"):
            for seed in seeds:
                candidate = policy if method == "local_ppo" else make_baseline(method, config, int(seed)+123)
                results = evaluate_policy(candidate, config, [seed], probability)
                rows.extend(dict(method=method, **row) for row in results)
    summary = []
    for probability in service_probabilities:
        for method in ("local_ppo", "random", "greedy", "wind_heuristic", "no_intervention"):
            subset = [r for r in rows if r["method"] == method and r["service_probability"] == probability]
            summary.append(dict(method=method, service_probability=probability, episodes=len(subset),
                **{key: float(np.mean([r[key] for r in subset])) for key in
                   ("discounted_return", "burned_cells", "cleared_cells", "affected_fraction", "contained", "truncated")}))
    write_csv(output/"episodes.csv", rows)
    write_csv(output/"summary.csv", summary)
    write_json(output/"protocol.json", {"seeds": seeds, "service_probabilities": service_probabilities,
               "local_checkpoint_sha256": policy.sha256,
               "interpretation": "One trained local seed; separate placement evaluation, not allocation performance."})
    return rows
