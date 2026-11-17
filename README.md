# Firehose — original simulator and trained policies

This workspace now runs William Shen and Aidan Curtis's **Firehose**: the original
Cell2Fire simulator, RGB CNN and Maskable PPO policies. The previous CRMAB and
two-level experiments are preserved under `archive/previous_project/`.

**Open [the demo](firehose_demo/demo.html) directly, or watch [the comparison GIF](firehose_demo/comparison.gif).**

The public trained models are used for inference. They are the authors' weights,
not models trained by this project. Harvesting permanently treats a selected cell
and can extinguish a burning cell. Repeated treatments can form a barrier.

```bash
cd /home/d1nu/LEBRON/BLOND/research
.venv/bin/python -m firehose_reproduction setup
.venv/bin/python -m firehose_reproduction record
.venv/bin/python -m firehose_reproduction demo
```

Open **http://127.0.0.1:8765**. The demo compares recorded neural-policy and baseline
episodes produced by the real simulator. Playback does not train the model.

```text
src/firehose_reproduction/
  setup.py, paths.py     pinned source/dependencies and simulator build
  environment.py        original Firehose environment with Gymnasium compatibility
  models.py             original CNN and verified published checkpoint tensors
  training.py           original Maskable PPO training recipe
  evaluation.py         policy and baseline evaluation
  recording.py          real episode recordings and offline demo export
  cli.py                setup, record and demo commands
firehose_demo/          plain browser UI and real recorded trajectories
tests/firehose_reproduction/  simulator and model compatibility checks
artifacts/             upstream source, native dependencies and full outputs
archive/previous_project/    superseded work, retained separately
```

Reference: [original source](https://github.com/aidan-curtis/firehose), pinned to
`e49a52a9a987a1585d25ddca2f711222ca500ffb`.
See [technical notes](docs/FIREHOSE_REPRODUCTION.md) for compatibility changes,
checkpoint provenance and reproduction limits. The MIT-letter experiment is not
required for this demo. Matching the paper's numerical results is not assumed.

Upstream source and derived CNN code retain the GPL-3.0 license in `LICENSE`.

Fresh checkout: create a virtual environment, install the project with `pip install -e '.[dev]'`, then run `firehose setup`. A C++ compiler, make and ar are required. `requirements.lock.txt` records the tested Python packages.

To train a fresh policy with the original architecture and PPO recipe:

```bash
firehose train --output artifacts/my-training --steps 5000000
```

Current demo measurements are in [results/README.md](results/README.md).

The demo now includes both **Sub20x20** and **Sub40x40**, each with its matching published checkpoint and four baselines. Select the map in **Map / neural checkpoint**.
