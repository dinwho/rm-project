"""A shared incident encoder and a joint team-allocation Q head."""
import numpy as np
import torch
from torch import nn


KEYS = ('maps', 'local_context', 'eligibility', 'context')


def observation_shapes(space):
    source = getattr(space, 'spaces', space)
    if set(source) != set(KEYS):
        raise ValueError('Global observation schema mismatch')
    shapes = {key: tuple(int(v) for v in getattr(source[key], 'shape', source[key])) for key in KEYS}
    maps = shapes['maps']
    if len(maps) != 4 or maps[1] != 6 or min(maps) < 1:
        raise ValueError('Expected incident × 6 × height × width maps')
    n, _, h, w = maps
    if (shapes['local_context'] != (n, 4) or shapes['eligibility'] != (n, h*w)
            or shapes['context'] != (2,)):
        raise ValueError('Global context/eligibility shape mismatch')
    return shapes


class GlobalQNetwork(nn.Module):
    """Q(s, subset), not one independent value per wildfire arm.

    All incidents share an encoder; the allocation head sees all encoded states
    together and can value the opportunity cost of assigning scarce teams.
    """
    def __init__(self, shapes, action_count):
        super().__init__()
        self.incident_count = shapes['maps'][0]
        self.cnn = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.eligible_encoder = nn.Sequential(nn.Linear(shapes['eligibility'][1], 16), nn.ReLU())
        self.incident_encoder = nn.Sequential(nn.Linear(64+16+4, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64*self.incident_count+2, 128), nn.ReLU(),
                                  nn.Linear(128, action_count))

    def forward(self, obs):
        batch, count, channels, height, width = obs['maps'].shape
        maps = self.cnn(obs['maps'].reshape(batch*count, channels, height, width)).reshape(batch, count, -1)
        eligible = self.eligible_encoder(obs['eligibility'])
        incidents = self.incident_encoder(torch.cat((maps, eligible, obs['local_context']), dim=-1))
        return self.head(torch.cat((incidents.flatten(1), obs['context']), dim=-1))
