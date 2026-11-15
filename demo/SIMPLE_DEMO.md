# The simple demo

Open **[simple_demo.gif](simple_demo.gif)**. This is an animation from the actual
Python simulator and saved DQN. There is no dashboard or web framework to explain.

The short script is **[simple_demo.py](simple_demo.py)**. It does four things:

1. Loads a trained policy and its matching environment.
2. Starts one incident with one ignition and six material units.
3. Repeatedly asks the policy for a feasible allocation, then advances the simulator.
4. Saves the resulting grid images as a GIF using Matplotlib.

Run it from `research`:

```bash
.venv/bin/python demo/simple_demo.py
```

The title shows the physical interval, remaining materials, and arms constructed
during the last interval. Green cells contain vegetation; orange cells are burning;
dark cells have burned; tan cells are persistent firebreaks; blue cells are the
settlement. One material unit builds one three-cell segment. At most two segments
can be constructed in one interval.

## A short explanation you can give while showing it

> “Each arm is a candidate firebreak segment. The DQN receives the current grid,
> arm information, wind, time and remaining budget. It chooses one feasible
> allocation, which can contain up to two segments. The simulator constructs those
> segments together and then advances all fire by one interval. The firebreaks
> persist, and fire keeps evolving around unselected candidates. During training,
> rewards penalize newly burned vegetation, clearing and settlement damage.
> This animation uses an already trained model.”

If asked about the animation code: **“Matplotlib displays the grid images returned
by the existing environment. It does not implement another simulator or learner.”**

The key interaction in the script is:

```python
action = agent.act(observation, info["action_mask"])
selected_arms = env.allocations.decode(action)
observation, reward, terminated, truncated, info = env.step(action)
```

The first line chooses a valid complete allocation. The second tells us which
candidate arms it contains. The third performs construction and one fire update.

## Optional comparisons

Keep seed 42 to use the same world and randomness:

```bash
.venv/bin/python demo/simple_demo.py --policy dqn_no_wind --output demo/unaware_demo.gif
.venv/bin/python demo/simple_demo.py --policy random --output demo/random_demo.gif
.venv/bin/python demo/simple_demo.py --policy wind_heuristic --output demo/heuristic_demo.gif
```

This is one illustrative episode, not an evaluation result. The pilot found that
the placement heuristics outperformed DQN. The [full beginner guide](../docs/BEGINNERS_GUIDE.md)
explains learning and evaluation in detail. The earlier browser dashboard remains
available as an optional detailed inspector.
