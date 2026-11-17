"""Record actual original-Firehose trajectories; no scripted fire or barriers."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from .models import UPSTREAM_COMMIT, PUBLIC_MODELS, load_pretrained

ROOT = Path(__file__).resolve().parents[2]


def record(policy_name, *, seed=123, checkpoint_name="20x20-fixed-ppo-maskable.zip", ignition=None):
    from .environment import make_env
    map_name = PUBLIC_MODELS[checkpoint_name][0]
    ignition = (372 if map_name == "Sub20x20" else 909) if ignition is None else ignition
    env = make_env(map_name=map_name, ignition=ignition, seed=seed, max_steps=2000)
    rng = np.random.default_rng(seed)
    checkpoint = ROOT / "artifacts/references/firehose/pretrained_models" / checkpoint_name
    model = load_pretrained(checkpoint, env) if policy_name == "ppo" else None
    frames = []
    cumulative = 0.0
    started = time.perf_counter()
    try:
        obs, info = env.reset(seed=seed)
        base = env.legacy
        def snapshot(step, action, reward):
            frames.append(dict(rgb=base.get_painted_image().tolist(), step=step, action=action,
                reward=float(reward), burning=int((base.state > 0).sum()),
                harvested=int((base.state < 0).sum()), **{"return": cumulative}))
        snapshot(0, None, 0)
        done = False
        while not done:
            if policy_name == "ppo":
                action = int(model.predict(obs, action_masks=env.action_masks(), deterministic=True)[0])
            elif policy_name == "no_intervention":
                action = -1
            elif policy_name == "random":
                # Original random baseline samples all cells, including repeats.
                action = int(rng.integers(env.action_space.n))
            else:
                points = np.argwhere(base.state == 1)
                if not len(points):
                    action = -1
                else:
                    anchor = np.array(base.flatten_idx_to_yx[ignition-1])
                    distances = np.linalg.norm(points-anchor, axis=1)
                    point = points[np.argmin(distances) if policy_name == "min_l2" else np.argmax(distances)]
                    action = int(point[0]*base.width+point[1])
            obs, reward, terminated, truncated, info = env.step(action)
            cumulative += reward
            snapshot(len(frames), action, reward)
            done = terminated or truncated
        metadata = {} if model is None else model.reproduction_metadata
        return dict(id=f"{checkpoint_name.removesuffix('.zip')}-{policy_name}-s{seed}",
            label="Published Maskable PPO" if model else policy_name.replace('_',' ').title(),
            policy=policy_name, map=map_name, seed=seed, ignition=ignition,
            warmup_steps=env.steps_before_sim, physics_steps_per_action=env.steps_per_action,
            checkpoint=checkpoint_name if model else None,
            checkpoint_sha256=metadata.get("checkpoint_sha256"), training_steps=metadata.get("training_steps"),
            frames=frames, metrics={"reward":cumulative,"fire_affected_cells":frames[-1]["burning"],
                "harvested_cells":frames[-1]["harvested"],"decisions":len(frames)-1},
            status="Time limit" if truncated else "Simulator finished",
            runtime_seconds=time.perf_counter()-started)
    finally:
        env.close()


def export_offline(data):
    folder = ROOT / "firehose_demo"
    page = (folder/"index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="styles.css">', '<style>'+(folder/"styles.css").read_text()+'</style>')
    script = '<script>window.FIREHOSE_DATA='+json.dumps(data,allow_nan=False).replace('<','\\u003c')+';</script>'
    page = page.replace('<script src="app.js"></script>',script+'<script>'+(folder/"app.js").read_text()+'</script>')
    (folder/"demo.html").write_text(page)


def generate():
    folder=ROOT/"firehose_demo";folder.mkdir(exist_ok=True)
    data={"provenance":{"source":"https://github.com/aidan-curtis/firehose", "commit":UPSTREAM_COMMIT,
          "simulator":"Original Cell2Fire C++",
          "weights":"Published upstream weights; no local training claimed",
          "state_note":"Original state=1 combines burning and burned cells. Red counts are fire-affected cells, not a verified active-fire count.",
          "scope":"Fixed ignitions: Sub20x20=372, Sub40x40=909. These episodes are illustrations, not a reproduction of paper-wide statistics."},"episodes":[]}
    for checkpoint_name in ("20x20-fixed-ppo-maskable.zip", "40x40-fixed-ppo-maskable.zip"):
        for name in ("ppo","no_intervention","random","min_l2","max_l2"):
            episode=record(name, checkpoint_name=checkpoint_name)
            data["episodes"].append(episode)
            (folder/"data.json").write_text(json.dumps(data,allow_nan=False))
            print(episode["map"],name,episode["metrics"],f"{episode['runtime_seconds']:.1f}s",flush=True)
    data["provenance"]["data_sha256_note"]="data.json contains full recorded RGB frames and actions"
    (folder/"data.json").write_text(json.dumps(data,allow_nan=False))
    export_offline(data)
    from .animation import export
    export(folder)
    out=ROOT/"results";out.mkdir(exist_ok=True)
    (out/"recorded_metrics.json").write_text(json.dumps({"provenance":data["provenance"],"episodes":[{k:v for k,v in e.items() if k!="frames"} for e in data["episodes"]]},indent=2))
    print(f"Demo: {folder/'demo.html'}",flush=True)
