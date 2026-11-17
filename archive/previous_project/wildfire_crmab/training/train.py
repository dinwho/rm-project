"""DQN training and validation-only checkpoint selection."""
import numpy as np
from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.environment.env import WildfireEnv
from wildfire_crmab.environment.scenario import Scenario
from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.replay import ReplayBuffer
from wildfire_crmab.agents.checkpoints import catalogue_metadata
from wildfire_crmab.config import budget_scale
from wildfire_crmab.evaluation.scenarios import scenario_suite
from wildfire_crmab.evaluation.runner import run_episode
from wildfire_crmab.utils.io import write_json, write_csv

def train_one(config, output, seed, observe_wind):
    t = config["training"]
    name = "dqn" if observe_wind else "dqn_no_wind"
    folder = output / "models" / name / f"seed_{seed}"
    folder.mkdir(parents=True)
    scale = budget_scale(config)
    env = WildfireEnv(EnvironmentConfig(**config["environment"]), observe_wind, scale)
    validation_env = WildfireEnv(env.config, observe_wind, scale)
    agent = DQNAgent(env.observation_space, env.action_space.n, seed,
                     learning_rate=t["learning_rate"], gamma=t["gamma"])
    replay = ReplayBuffer(t["replay_capacity"], env.observation_space, env.action_space.n, seed)
    rng = np.random.default_rng(seed)
    suite = scenario_suite("validation", t["validation_worlds"], t["budgets"])
    def reset_train():
        # Select the list entry itself: numpy.choice would coerce mixed null/int budgets.
        budget = t["budgets"][int(rng.integers(len(t["budgets"])))]
        return env.reset(options={"scenario": Scenario(int(rng.integers(1_000_000_000)), budget)})

    def validate(step, loss):
        results = [run_episode(validation_env, agent, s, t["gamma"]) for s in suite]
        return dict(step=step,
                    discounted_return=float(np.mean([r["discounted_return"] for r in results])),
                    affected_fraction=float(np.mean([r["affected_fraction"] for r in results])),
                    loss=loss)

    obs, info = reset_train()
    # Measuring initialization makes a small fixed-world run a learning diagnostic,
    # rather than presenting a working training loop as evidence of improvement.
    initial_validation = validate(0, None)
    validation = [initial_validation]
    agent.save(folder/"initial.pt", {**catalogue_metadata(env), "step": 0})
    episodes = []
    best = -np.inf
    best_step = 0
    loss = None
    updates = 0
    for step in range(1, t["steps"]+1):
        epsilon = 1 - .95 * min(1., (step-1)/max(1, .6*t["steps"]))
        action = agent.act(obs, info["action_mask"], epsilon)
        nxt, reward, terminated, truncated, nxt_info = env.step(action)
        replay.add(obs, action, reward, nxt, terminated or truncated, info["action_mask"], nxt_info["action_mask"])
        obs, info = nxt, nxt_info
        if step >= t["learning_starts"] and replay.size >= t["batch_size"] and step % t["train_every"] == 0:
            loss = agent.learn(replay.sample(t["batch_size"]))
            updates += 1
        if step % t["target_every"] == 0:
            agent.update_target()
        if env.finished:
            episodes.append(dict(step=step, episode_return=info["episode_return"], epsilon=epsilon,
                                 budget=env.scenario.budget, loss=loss))
            obs, info = reset_train()
        if step % t["validation_every"] == 0 or step == t["steps"]:
            measurement = validate(step, loss)
            score = measurement["discounted_return"]
            validation.append(measurement)
            if score > best:
                best, best_step = score, step
                agent.save(folder/"best.pt", {**catalogue_metadata(env), "step": step})
            write_csv(folder/"validation.csv", validation)
            write_csv(folder/"episodes.csv", episodes)
            print(f"{name} seed={seed} step={step}/{t['steps']} validation discounted return={score:.4f}", flush=True)
    agent.save(folder/"last.pt", {**catalogue_metadata(env), "step": t["steps"]})
    write_json(folder/"training_summary.json", dict(best_step=best_step, best_validation_return=best,
               initial_validation_return=initial_validation["discounted_return"],
               validation_improvement_from_initial=best-initial_validation["discounted_return"],
               optimizer_updates=updates, episodes=len(episodes), observe_wind=observe_wind,
               selection="maximum mean discounted validation return; earlier checkpoint wins ties"))

