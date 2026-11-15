# Interactive research demo

For a simpler demonstration, use **[the Python-only demo](SIMPLE_DEMO.md)** and
its [animation](simple_demo.gif). The dashboard below is an optional detailed view.

Open **[demo.html](demo.html)** directly in a browser for the portable version.
It contains actual simulator replays for world 42, budget 6, all six policies,
saved DQN Q-values, architecture details and pilot results. It requires no server,
network or Python once built. The file is a frozen replay, not JavaScript training
or a second implementation of the simulator.

To explore different seeds and budgets, run from `research`:

```bash
.venv/bin/python demo/server.py --port 8765
```

Then open **http://127.0.0.1:8765**. This server runs the actual Python environment
and PyTorch policy to generate each requested episode; Play, Next and the timeline
inspect that generated trajectory. The server listens only on localhost. Stop it
with Ctrl+C. It loads validation-selected seed-11 pilot weights, or clearly labelled
smoke weights if pilot weights do not exist; it never substitutes an untrained model.

## What to show

1. **Simulation:** choose a policy, step once, and inspect the simultaneous subset,
   persistent firebreaks, material accounting and restless fire evolution. Select
   an arm to inspect construction status, eligibility and local conditions.
2. **Allocation values:** see actual Q outputs for the best five feasible subsets,
   the number of masked actions and the checkpoint used.
3. **DQN model and training:** expand this section to follow the map/arm/context inputs through the actual network and
   read the replay/target update procedure. The architecture can be expanded.
4. **Pilot results:** expand this section to compare all six policies and see recorded validation
   trajectories from three training seeds for each DQN variant.
5. Switch between the two DQNs on the same seed. The observed wind disappears for
   the unaware model, while actual wind stays the same.

The protected asset and vegetation are synthetic. Model inference in the demo does
not continue training. The demo uses existing learned checkpoints and does not
claim DQN outperforms the heuristics.

## Files

| File | Purpose |
| --- | --- |
| `engine.py` | Shared research simulator/policy adapter and replay generation |
| `server.py` | Standard-library local HTTP server and two read-only API routes |
| `static/index.html` | Simulation, model and results views |
| `static/styles.css` | Responsive visual layout |
| `static/app.js` | Timeline, drawing, policy selection and data display |
| `build_snapshot.py` | Embed real replays and UI into portable `demo.html` |
| `check_browser.py` | Live/offline Chromium interaction and mobile checks |
| `preview.png` | Browser screenshot of the tested demo |

Regenerate the portable demo after training:

```bash
.venv/bin/python demo/build_snapshot.py
```

Optional browser QA (server must be running):

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
.venv/bin/python demo/check_browser.py
```
