"""Plot legal adjacent placements using the connected_barrier example scenario."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wildfire_crmab import WildfireEnv, Scenario, EnvironmentConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/allocation.png"))
    args = parser.parse_args()
    env = WildfireEnv(EnvironmentConfig(fixed_ignition=(0, 0), fixed_wind=(0., .8)))
    env.reset(options={"scenario": Scenario(0, None)})
    frames = [env.render().copy()]
    labels = ["Initial fire at (0,0)"]
    for row, col in ((1, 0), (1, 1), (0, 1)):
        env.step(env.allocations.encode((row*env.config.grid_size+col,)))
        frames.append(env.render().copy())
        labels.append(f"Build ({row},{col}), then fire update {env.step_number}")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), layout="constrained")
    for ax, frame, label in zip(axes, frames, labels):
        ax.imshow(frame[:5, :5], interpolation="nearest")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_title(label, fontsize=9)
    fig.suptitle("One construction per tick: adjacent firebreaks join the boundary\n"
                 "Green: vegetation · orange: burning · dark: burned · tan: cleared\n"
                 "Fixed hand-built example, not a learned policy; upper-left 5×5 shown", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
