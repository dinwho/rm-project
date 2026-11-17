# Interactive research demo

Start with the **[simple Python demo](SIMPLE_DEMO.md)** and its
[animation](simple_demo.gif). This browser view is an optional inspector.

Open **[demo.html](demo.html)** directly. It contains real model replays for
seed 42 on the 24×24 cell-construction experiment. The page labels CRMAB as a
spatially coupled mathematical model, not as the Firehose/Cell2Fire simulator.
Available DQN weights are embedded as replay data when the file is built; missing
weights are labelled unavailable. The HTML performs playback, not training.

For live episodes, run from `research`:

```bash
.venv/bin/python demo/server.py --port 8765
```

Open **http://127.0.0.1:8765**; stop the server with Ctrl+C. The default is the wind
heuristic, which needs no saved weights. The server uses the actual Python
environment and policy. All policies use `configs/firehose.json` and matched
scenario seeds for held-out comparisons.

Each arm is a cell. One cell can be cleared before each fire update. Clearing
persists, cannot extinguish burning cells, and can form connected barriers.
No episode cap is the default. Optional finite caps are available for baselines;
the diagnostic DQN is restricted to its training setting without an episode cap.

To inspect DQN, train the larger experiment:
`wildfire-crmab run --config configs/firehose.json --output artifacts/firehose`.
The demo loads the matching validation-selected `seed_11/best.pt` checkpoint.
Expand **CRMAB mathematical model** for the transition, reward and coupling
qualification; the results section shows matched held-out comparisons and the
seed-aggregated validation curve.

The comparison uses varied held-out worlds and common fire-spread draws across
policies. Results remain synthetic, and the learning curve is a validation
diagnostic rather than proof of convergence or policy superiority.

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
