# The simple demo

Open **[simple_demo.gif](simple_demo.gif)**. It shows the actual simulator using
the **wind placement heuristic**, a simple rule that clears a currently threatened
cell. The GIF is not evidence that DQN has learned containment.

Run from `research`:

```bash
.venv/bin/python demo/simple_demo.py
```

Each grid cell is one candidate construction arm. At each interval the policy
can clear **one unburned vegetation cell**, or wait. The fire then advances once.
Cleared cells stay cleared and can join into continuous lines or perimeters.
There is no episode-wide construction cap; time, eligibility and the one-cell
capacity still limit construction. Burning and already-cleared cells are invalid.

The title shows the interval, number of constructed cells and last construction.
Green is vegetation, orange is active fire, dark is burned vegetation, and tan is
a permanent firebreak. A cell ID is `row * grid_size + column`, starting at zero.

## An explanation you can give

> “The simulator updates the fire whether we build or wait. Each eligible cell is
> an arm that we can activate once by clearing it. Construction costs vegetation,
> so the reward penalizes both clearing and new burning. Adjacent cleared cells
> can block spread. DQN learns which valid cell to clear, or whether to wait.
> This particular animation uses the wind heuristic; I can load a trained DQN
> separately.”

The short [script](simple_demo.py) calls [engine.py](engine.py) to run the real
simulator, then uses Matplotlib to display the returned grid images. The central
interaction in the engine is:

```python
action = policy.act(obs, mask)
obs, previous_reward, terminated, truncated, info = env.step(action)
```

The policy chooses a feasible action. `step` performs that construction and one
shared fire update. The animation does not create another simulator or train DQN.

## Compare policies on the same incident

```bash
.venv/bin/python demo/simple_demo.py --policy random --output demo/random_demo.gif
.venv/bin/python demo/simple_demo.py --policy no_intervention --output demo/no_construction.gif
.venv/bin/python demo/simple_demo.py --policy dqn --output demo/dqn_demo.gif
.venv/bin/python demo/simple_demo.py --policy dqn_no_wind --output demo/unaware_demo.gif
```

DQN requires matching saved weights at
`artifacts/continuous_fixed/models/<policy>/seed_11/best.pt`. If they are absent,
the script says so; it never substitutes the old segment policy or an untrained
network. All policies use `configs/fixed.json` and the same seed (42 by default).
Different seeds provide fresh fire-spread draws while this diagnostic keeps layout,
ignition and wind fixed. A single animation is illustrative, not an evaluation.

The [browser inspector](README.md) is optional. It adds a timeline and the actual
DQN action values when weights are available.
