"""Fresh Maskable PPO training using the original Firehose CNN and environment."""
from dataclasses import asdict
from pathlib import Path
from .environment import make_env
from .models import make_model, save_model


def train(output, *, steps=5_000_000, map_name="Sub20x20", ignition=372, seed=1,
          n_steps=2048, batch_size=64, n_epochs=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(map_name=map_name, ignition=ignition)
    task = {"map": map_name, "ignition": ignition, "reward": "FireSizeReward",
            "warmup": env.steps_before_sim, "steps_per_action": env.steps_per_action}
    try:
        model = make_model(env, seed=seed, n_steps=n_steps, batch_size=batch_size,
                           n_epochs=n_epochs, verbose=1)
        model.learn(total_timesteps=steps)
        return save_model(model, output/"policy.zip", task,
                          metadata={"seed":seed,"requested_steps":steps,
                                    "note":"Fresh local training; not the published checkpoint or published evaluation."})
    finally:
        env.close()
