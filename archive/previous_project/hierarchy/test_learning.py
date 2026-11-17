from dataclasses import replace
import csv
import json
import numpy as np
import pytest
import torch

from wildfire_crmab.environment.config import EnvironmentConfig
from wildfire_crmab.hierarchy.environment import GlobalWildfireEnv
from wildfire_crmab.hierarchy.dqn import GlobalDQN, GlobalReplay, checkpoint_metadata
from wildfire_crmab.hierarchy.training import train_global, world_seeds
from wildfire_crmab.hierarchy.evaluation import evaluate_global


class FrozenTestConstructor:
    """Only a test fixture; production experiments load a trained PPO checkpoint."""
    sha256 = 'a'*64

    def act(self, observation, action_mask):
        return int(np.flatnonzero(action_mask)[-1])


def environment(k=2):
    return GlobalWildfireEnv(EnvironmentConfig(grid_size=6, max_steps=6), FrozenTestConstructor(), team_capacity=k)


def test_global_network_uses_incident_observations_and_masks_invalid_large_q():
    env = environment(0)
    obs, info = env.reset(seed=12)
    agent = GlobalDQN(env.observation_space, env.action_space.n, seed=4)
    with torch.no_grad():
        agent.online.head[-1].weight.zero_()
        agent.online.head[-1].bias.copy_(torch.arange(env.action_space.n)*100.)
    assert agent.act(obs, info['action_mask']) == 0
    assert {agent.act(obs, info['action_mask'], 1.) for _ in range(20)} == {0}
    obs, info = env.reset(seed=12, options={'team_capacity': 2})
    expected = int(np.flatnonzero(info['action_mask'])[-1])
    assert agent.act(obs, info['action_mask']) == expected
    assert all(info['action_mask'][agent.act(obs, info['action_mask'], 1.)] for _ in range(50))
    with pytest.raises(ValueError, match='schema'):
        agent.act({'map': obs['maps'][0]}, info['action_mask'])


def test_global_replay_target_optimization_roundtrip_and_constructor_binding(tmp_path):
    env = environment()
    obs, info = env.reset(seed=4)
    agent = GlobalDQN(env.observation_space, env.action_space.n, seed=4)
    replay = GlobalReplay(10, env.observation_space, env.action_space.n)
    action = int(np.flatnonzero(info['action_mask'])[-1])
    next_obs, reward, term, trunc, nxt = env.step(action)
    replay.add(obs, action, reward, next_obs, term or trunc, info['action_mask'], nxt['action_mask'])
    originals = [p.detach().clone() for p in agent.online.parameters()]
    loss = agent.learn(replay.sample(4))
    assert np.isfinite(loss)
    assert any(not torch.equal(a,b) for a,b in zip(originals,agent.online.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(originals,agent.target.parameters()))
    agent.update_target()
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),agent.target.parameters()))
    meta = checkpoint_metadata(env, FrozenTestConstructor())
    agent.save(tmp_path/'global.pt', meta)
    restored, saved = GlobalDQN.load(tmp_path/'global.pt', meta)
    batch = {k: torch.as_tensor(v).unsqueeze(0) for k,v in obs.items()}
    with torch.no_grad():
        torch.testing.assert_close(restored.online(batch),agent.online(batch),rtol=0,atol=0)
    assert saved == meta
    with pytest.raises(ValueError,match='mismatch'):
        GlobalDQN.load(tmp_path/'global.pt', {**meta,'local_checkpoint_sha256':'b'*64})
    changed = GlobalWildfireEnv(replace(env.local_config, base_spread=.3), FrozenTestConstructor())
    with pytest.raises(ValueError,match='mismatch'):
        GlobalDQN.load(tmp_path/'global.pt',checkpoint_metadata(changed,FrozenTestConstructor()))


def test_replay_rejects_invalid_transition_without_mutating():
    env = environment(0)
    obs, info = env.reset(seed=3)
    replay = GlobalReplay(2,env.observation_space,env.action_space.n)
    with pytest.raises(ValueError,match='Infeasible'):
        replay.add(obs,1,0,obs,False,info['action_mask'],info['action_mask'])
    assert replay.size == replay.position == 0
    replay.add(obs,0,0,obs,True,info['action_mask'],np.zeros(env.action_space.n,bool))
    obs['maps'][:]=0
    assert replay.sample(1)['observations']['maps'].sum() > 0


def test_global_training_evaluation_pipeline_and_matched_k0(tmp_path):
    config = EnvironmentConfig(grid_size=6,max_steps=5)
    constructor = FrozenTestConstructor()
    best = train_global(tmp_path,config,constructor,steps=20,seed=7,
                        team_counts=(0,2),validation_worlds=1,validation_every=10,
                        batch_size=4,replay_capacity=30,learning_starts=4,train_every=2,target_every=4)
    summary = json.loads((best.parent/'summary.json').read_text())
    assert summary['optimizer_updates'] > 0 and summary['best_step'] in (10,20)
    rows = evaluate_global(tmp_path,config,constructor,[best],worlds=2,
                           team_counts=(0,2),bootstrap_samples=20)
    assert len(rows)==5*2*2
    assert not any(row['invalid_actions'] for row in rows)
    assert {row['local_checkpoint_sha256'] for row in rows} == {constructor.sha256}
    for world in world_seeds('test',2):
        k0 = [row for row in rows if row['world_seed']==world and row['team_capacity']==0]
        assert len({row['episode_return'] for row in k0})==1
        assert len({row['burned_cells'] for row in k0})==1
        assert all(row['cleared_cells']==0 and row['team_utilization']==0 for row in k0)
    assert (tmp_path/'GLOBAL_RESULTS.md').is_file()
    paired=list(csv.DictReader((tmp_path/'global_evaluation'/'paired_differences.csv').open()))
    assert paired and set(world_seeds('validation',2)).isdisjoint(world_seeds('test',2))
