"""Masked DQN and replay for global, simultaneous incident allocation."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import torch
from torch import nn
from wildfire_crmab.agents.network import bellman_targets
from wildfire_crmab.agents.observations import checked_mask, checked_observation
from .network import observation_shapes, GlobalQNetwork

FORMAT = 'wildfire-hierarchical-global-dqn-v1'


def checkpoint_metadata(env, local_policy):
    digest = getattr(local_policy, 'sha256', None)
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Global learning requires a frozen constructor checkpoint SHA256')
    # Canonical JSON values keep tuples/lists stable across disk roundtrips.
    return json.loads(json.dumps(dict(local_checkpoint_sha256=digest,
        environment=asdict(env.local_config), incident_count=env.incident_count,
        subsets=env.subsets, observation_schema='incidents-maps6-context4-eligibility-global2-v1',
        construction_rate=1, max_teams_per_incident=1)))


class GlobalDQN:
    def __init__(self, observation_space, action_count, seed=11, gamma=.99, learning_rate=3e-4):
        if not 0 < gamma <= 1 or learning_rate <= 0 or not np.isfinite(learning_rate) or action_count < 1:
            raise ValueError('Invalid global DQN parameters')
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.shapes = observation_shapes(observation_space)
        self.action_count, self.seed = int(action_count), int(seed)
        self.gamma, self.learning_rate = float(gamma), float(learning_rate)
        self.rng = np.random.default_rng(seed)
        self.online = GlobalQNetwork(self.shapes, action_count)
        self.target = GlobalQNetwork(self.shapes, action_count)
        self.update_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    def act(self, observation, action_mask, epsilon=0.):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must lie in [0,1]')
        obs = checked_observation(observation, self.shapes)
        mask = checked_mask(action_mask, self.action_count)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            values = self.online({k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()})[0]
            if not torch.isfinite(values[torch.as_tensor(mask)]).all():
                raise ValueError('Nonfinite feasible Q values')
            return int(values.masked_fill(~torch.as_tensor(mask), -torch.inf).argmax())

    def learn(self, batch):
        if not batch['masks'].gather(1, batch['actions'].unsqueeze(1)).all():
            raise ValueError('Infeasible replay action')
        values = self.online(batch['observations']).gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            targets = bellman_targets(batch['rewards'], batch['done'], self.target(batch['next_observations']),
                                      batch['next_masks'], self.gamma)
        loss = nn.functional.smooth_l1_loss(values, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite global DQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, metadata):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format=FORMAT, shapes=self.shapes, action_count=self.action_count,
                        seed=self.seed, gamma=self.gamma, learning_rate=self.learning_rate,
                        state_dict=self.online.state_dict(), metadata=metadata), path)

    @classmethod
    def load(cls, path, expected_metadata=None):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') != FORMAT:
            raise ValueError('Not a hierarchical global DQN checkpoint')
        metadata = saved['metadata']
        if expected_metadata is not None and any(metadata.get(k) != v for k, v in expected_metadata.items()):
            raise ValueError('Global checkpoint/frozen constructor/environment mismatch')
        agent = cls(saved['shapes'], saved['action_count'], saved['seed'], saved['gamma'], saved['learning_rate'])
        agent.online.load_state_dict(saved['state_dict'])
        agent.online.eval()
        agent.update_target()
        return agent, metadata


class GlobalReplay:
    def __init__(self, capacity, space, action_count, seed=11):
        if capacity < 1 or action_count < 1:
            raise ValueError('Positive replay dimensions required')
        self.capacity, self.action_count = int(capacity), int(action_count)
        self.shapes = observation_shapes(space)
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.observations = {k: np.empty((capacity, *v), np.float32) for k, v in self.shapes.items()}
        self.next_observations = {k: np.empty_like(v) for k, v in self.observations.items()}
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.done = np.empty(capacity, bool)
        self.masks = np.empty((capacity, action_count), bool)
        self.next_masks = np.empty_like(self.masks)

    def add(self, obs, action, reward, successor, done, mask, next_mask):
        current = checked_observation(obs, self.shapes)
        nxt = checked_observation(successor, self.shapes)
        mask = checked_mask(mask, self.action_count)
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError('done must be boolean')
        next_mask = checked_mask(next_mask, self.action_count, allow_empty=done)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < self.action_count or not mask[action] or not np.isfinite(reward)):
            raise ValueError('Infeasible or nonfinite replay experience')
        i = self.position
        for key in self.shapes:
            self.observations[key][i], self.next_observations[key][i] = current[key], nxt[key]
        self.actions[i], self.rewards[i], self.done[i] = action, reward, done
        self.masks[i], self.next_masks[i] = mask, next_mask
        self.position = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, count):
        if self.size < 1 or count < 1:
            raise ValueError('Cannot sample empty replay or nonpositive batch')
        ix = self.rng.integers(self.size, size=count)
        result = {group: {k: torch.as_tensor(v[ix]) for k, v in getattr(self, group).items()}
                  for group in ('observations', 'next_observations')}
        result.update({key: torch.as_tensor(getattr(self, key)[ix])
                       for key in ('actions', 'rewards', 'done', 'masks', 'next_masks')})
        return result
