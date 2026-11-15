"""Create a standalone figure of candidate arms and persistent construction."""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wildfire_crmab import WildfireEnv, Scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/allocation.png"))
    args = parser.parse_args()
    env = WildfireEnv()
    _, info = env.reset(options={"scenario": Scenario(42, 6)})
    action = next(int(a) for a in np.flatnonzero(info["action_mask"])
                  if len(env.allocations.decode(int(a))) == 2)
    selected = env.allocations.decode(action)
    frames = [env.render().copy()]
    env.step(action)
    frames.append(env.render().copy())
    while env.step_number < 10 and not env.finished:
        env.step(0)
    frames.append(env.render().copy())
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    titles = ["Fixed construction candidates, t=0", f"Build arms {selected}, t=1",
              f"All candidates passive afterward, t={env.step_number}"]
    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(0, 12, 2))
        ax.set_yticks(range(0, 12, 2))
    for segment in env.arms.segments:
        rows, cols = env.arms.cells(segment.arm_id)
        axes[0].plot(cols, rows, color="white", linewidth=1, alpha=.7)
    for i in selected:
        r, c = env.arms.cells(i)
        for ax in axes[:2]:
            ax.plot(c, r, color="gold", linewidth=2)
            ax.text(c.mean(), r.mean()-.3, str(i), color="black", fontsize=9)
    fig.suptitle("Green: fuel · orange: fire · dark: burned · tan: firebreak · blue: asset\n"
                 "Illustrative feasible allocation, not an optimized policy", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
