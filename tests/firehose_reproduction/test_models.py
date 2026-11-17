import io
import zipfile
import numpy as np
import pytest
import torch
from gymnasium import spaces
import gymnasium as gym
from firehose_reproduction.models import load_pretrained, load_trained
from firehose_reproduction.paths import REFERENCE, BINARY


class SpaceOnlyEnv(gym.Env):
    observation_space = spaces.Box(0,255,(3,20,20),np.uint8)
    action_space = spaces.Discrete(400)
    map_name = 'Sub20x20'


@pytest.mark.skipif(not REFERENCE.exists(),reason='Run firehose setup')
def test_published_tensors_preserved_and_masking_prevents_invalid_selection():
    path=REFERENCE/'pretrained_models/20x20-fixed-ppo-maskable.zip'
    model=load_pretrained(path,SpaceOnlyEnv())
    with zipfile.ZipFile(path) as archive:
        state=torch.load(io.BytesIO(archive.read('policy.pth')),weights_only=True,map_location='cpu')
    target=model.policy.state_dict()
    for name,value in state.items():
        torch.testing.assert_close(value,target[name],rtol=0,atol=0)
    mask=np.zeros(400,bool);mask[7]=True
    assert int(model.predict(np.zeros((3,20,20),np.uint8),action_masks=mask,deterministic=True)[0])==7
    with pytest.raises(ValueError,match='trained on'):
        load_pretrained(path,SpaceOnlyEnv(),expected_map='mit_m')


@pytest.mark.skipif(not BINARY.exists(),reason='Run firehose setup')
def test_fresh_training_optimizer_and_checkpoint_roundtrip(tmp_path):
    from firehose_reproduction.training import train
    from firehose_reproduction.environment import make_env
    path=train(tmp_path/'smoke',steps=32,n_steps=32,batch_size=16,n_epochs=1)
    env=make_env()
    task={'map':'Sub20x20','ignition':372,'reward':'FireSizeReward','warmup':20,'steps_per_action':8}
    try:
        model=load_trained(path,env,expected_task=task)
        obs,_=env.reset(seed=1)
        action=int(model.predict(obs,action_masks=env.action_masks(),deterministic=True)[0])
        assert env.action_masks()[action]
        assert model.num_timesteps==32
        with pytest.raises(ValueError,match='task or checksum'):
            load_trained(path,env,expected_task={})
    finally:
        env.close()
