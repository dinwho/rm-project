"""Train the global allocator only after a constructor checkpoint is frozen."""
from pathlib import Path
import numpy as np
from wildfire_crmab.utils.io import write_json, write_csv
from .environment import GlobalWildfireEnv
from .dqn import GlobalDQN, GlobalReplay, checkpoint_metadata


def world_seeds(split, count):
    if split not in ('validation', 'test') or count < 1:
        raise ValueError('Expected validation/test and positive count')
    offset = 1_000_000_000 if split == 'validation' else 2_000_000_000
    return [offset+31337+31*i for i in range(count)]


def run_global_episode(env, policy, world_seed, capacity, gamma=.99):
    obs, info = env.reset(seed=int(world_seed), options={'team_capacity': int(capacity)})
    discounted = 0.
    terminated = truncated = False
    invalid = 0
    while not (terminated or truncated):
        action = policy.act(obs, info['action_mask'])
        if not info['action_mask'][action]:
            raise AssertionError('Policy selected infeasible global action')
        obs, reward, terminated, truncated, info = env.step(action)
        discounted += gamma**(info['steps']-1)*reward
    keys = ('burned_cells', 'cleared_cells', 'burned_fraction', 'cleared_fraction', 'affected_fraction',
            'contained_count', 'contained_fraction', 'extinguished_count', 'team_utilization',
            'construction_utilization', 'assigned_team_intervals', 'steps', 'active_fire_cells', 'episode_return')
    return dict(world_seed=int(world_seed), team_capacity=int(capacity),
                scenario_id=f's{world_seed}-k{capacity}', incident_seeds=','.join(map(str, info['incident_seeds'])),
                **{k: info[k] for k in keys}, discounted_return=float(discounted),
                terminated=int(terminated), truncated=int(truncated), invalid_actions=invalid)


def train_global(output, config, local_policy, *, steps=12000, seed=11, incident_count=4,
                 team_counts=(0, 1, 2, 3), validation_worlds=8, validation_every=3000,
                 batch_size=64, replay_capacity=10000, learning_starts=500,
                 train_every=4, target_every=500, gamma=.99, learning_rate=3e-4):
    """Return a validation-selected checkpoint; no test outcome selects weights.

    Capacity is sampled at episode reset and explicitly observed, so evaluation
    at K=0,1,2,3 does not silently change an unobserved training constraint.
    """
    if (not team_counts or len(set(team_counts)) != len(team_counts)
            or any(type(k) is not int or not 0 <= k <= incident_count for k in team_counts)):
        raise ValueError('Invalid training team counts')
    if min(steps, validation_worlds, validation_every, batch_size, replay_capacity, train_every, target_every) < 1:
        raise ValueError('Positive training counts required')
    if batch_size > replay_capacity or learning_starts < 0:
        raise ValueError('Invalid global replay settings')
    folder = Path(output)/'global'/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    env = GlobalWildfireEnv(config, local_policy, incident_count, team_counts[0])
    validation_env = GlobalWildfireEnv(config, local_policy, incident_count, team_counts[0])
    metadata = checkpoint_metadata(env, local_policy)
    agent = GlobalDQN(env.observation_space, env.action_space.n, seed, gamma, learning_rate)
    replay = GlobalReplay(replay_capacity, env.observation_space, env.action_space.n, seed)
    rng = np.random.default_rng(seed)
    suite = world_seeds('validation', validation_worlds)
    settings = dict(steps=steps, seed=seed, incident_count=incident_count, team_counts=list(team_counts),
                    validation_seeds=suite, validation_every=validation_every, batch_size=batch_size,
                    replay_capacity=replay_capacity, learning_starts=learning_starts, train_every=train_every,
                    target_every=target_every, gamma=gamma, learning_rate=learning_rate,
                    epsilon_schedule='1 to .05 over first 60% of physical global steps', **metadata)
    write_json(folder/'settings.json', settings)

    def measure(step, loss):
        records = [run_global_episode(validation_env, agent, world, k, gamma) for world in suite for k in team_counts]
        return dict(step=step, discounted_return=float(np.mean([r['discounted_return'] for r in records])),
                    affected_fraction=float(np.mean([r['affected_fraction'] for r in records])), loss=loss)

    def reset():
        return env.reset(seed=int(rng.integers(1_000_000_000)),
                         options={'team_capacity': int(rng.choice(team_counts))})

    validation = [measure(0, None)]
    agent.save(folder/'initial.pt', {**metadata, 'step': 0, 'training_team_counts': list(team_counts)})
    episodes = []
    best, best_step, loss, updates = -np.inf, 0, None, 0
    obs, info = reset()
    for step in range(1, steps+1):
        epsilon = 1-.95*min(1., (step-1)/max(1, .6*steps))
        action = agent.act(obs, info['action_mask'], epsilon)
        nxt, reward, terminated, truncated, nxt_info = env.step(action)
        replay.add(obs, action, reward, nxt, terminated or truncated, info['action_mask'], nxt_info['action_mask'])
        obs, info = nxt, nxt_info
        if step >= learning_starts and replay.size >= batch_size and step % train_every == 0:
            loss = agent.learn(replay.sample(batch_size)); updates += 1
        if step % target_every == 0:
            agent.update_target()
        if terminated or truncated:
            episodes.append(dict(step=step, capacity=env.team_capacity, episode_return=info['episode_return'],
                                 affected_fraction=info['affected_fraction'], epsilon=epsilon, loss=loss))
            obs, info = reset()
        if step % validation_every == 0 or step == steps:
            record = measure(step, loss)
            validation.append(record)
            if record['discounted_return'] > best:
                best, best_step = record['discounted_return'], step
                agent.save(folder/'best.pt', {**metadata, 'step': step, 'training_team_counts': list(team_counts)})
            write_csv(folder/'validation.csv', validation)
            write_csv(folder/'episodes.csv', episodes)
            print(f'global DQN seed={seed} step={step}/{steps} validation return={record["discounted_return"]:.4f}', flush=True)
    agent.save(folder/'last.pt', {**metadata, 'step': steps, 'training_team_counts': list(team_counts)})
    write_json(folder/'summary.json', dict(best_step=best_step, best_validation_return=best,
               initial_validation_return=validation[0]['discounted_return'],
               validation_improvement=best-validation[0]['discounted_return'], optimizer_updates=updates,
               episodes=len(episodes), local_checkpoint_sha256=metadata['local_checkpoint_sha256'],
               selection='best trained checkpoint by mean validation discounted return; earlier wins ties'))
    return folder/'best.pt'
