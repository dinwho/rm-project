# Interactive research demo

Start with the **[simple Python demo](SIMPLE_DEMO.md)** and its
[animation](simple_demo.gif). This browser view is an optional inspector.

Open **[demo.html](demo.html)** directly. It contains real simulator replays for
seed 42 in the fixed-scenario cell-construction experiment. Available DQN weights
are embedded as replay data when the file is built; missing models are labelled
unavailable. The HTML performs playback, not training or another fire simulation.

For live episodes, run from `research`:

```bash
.venv/bin/python demo/server.py --port 8765
```

Open **http://127.0.0.1:8765**; stop the server with Ctrl+C. The default is the wind
heuristic, which needs no saved weights. The server uses the actual Python
simulator and policy. All policies use `configs/fixed.json` so comparisons face
the same layout, ignition and wind; the seed changes the fire-spread draws.

Each arm is a cell. One cell can be cleared before each fire update. Clearing
persists, cannot extinguish burning cells, and can form connected barriers.
No episode cap is the default. Optional finite caps are available for baselines;
the diagnostic DQN is restricted to its training setting without an episode cap.

To inspect DQN, train the fixed experiment into `artifacts/continuous_fixed`.
The demo loads only the matching validation-selected `seed_11/best.pt` checkpoint,
never the historical fixed-segment weights. Expand **DQN model and training** for
its actual architecture and five best feasible Q-values. Expand **Fixed-scenario
diagnostic results** for the new experiment's results when available. Historical
pilot results are not mixed into this display.

The two DQNs face the same wind-driven simulator; the unaware model gets zero in
its explicit wind inputs. Fixed wind is only an initial learning diagnostic and
cannot demonstrate generalization across wind conditions. This demo makes no
claim that DQN learned a successful perimeter.

| File | Purpose |
| --- | --- |
| `simple_demo.py` | Small Python script that saves one episode as a GIF |
| `engine.py` | Real simulator, model loading, replay data and diagnostic results |
| `server.py` | Local standard-library server and read-only API routes |
| `static/` | Plain HTML, CSS and JavaScript playback interface |
| `build_snapshot.py` | Freeze real replays into portable `demo.html` |
| `check_browser.py` | Optional browser checks |

Regenerate the offline file after training:

```bash
.venv/bin/python demo/build_snapshot.py
```
