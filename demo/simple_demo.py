"""A small Python-only demo: simulate one policy, then save the grid frames as a GIF."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from engine import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["dqn", "dqn_no_wind", "random", "wind_heuristic", "no_intervention"],
                        default="wind_heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("demo/simple_demo.gif"))
    args = parser.parse_args()

    # 1. Run the real simulator. DQN loads actual weights; heuristics need none.
    try:
        episode = rollout(args.policy, args.seed)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    frames = episode["frames"]
    for previous, current in zip(frames, frames[1:]):
        print(f"Interval {current['t']:2}: cells={previous['allocation']}, "
              f"built={sum(current['built'])}, reward={current['previous_reward']:.3f}")

    # 2. Display the images returned by the environment, with ordinary labels.
    figure, axis = plt.subplots(figsize=(6, 6.7))
    figure.subplots_adjust(top=.81, bottom=.17)
    picture = axis.imshow(frames[0]["rgb"], interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    colors = [("Vegetation", "#41874b"), ("Fire", "#f45320"),
              ("Burned", "#413c37"), ("Firebreak", "#cfad75")]
    figure.legend(handles=[Patch(color=color, label=name) for name, color in colors],
                  loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .04))
    figure.text(.5, .02, "Build one cell or wait → fire advances once. Clearing persists.",
                ha="center", fontsize=9)
    wind = episode["actual_wind"]
    figure.suptitle(f"Cell firebreak construction\nWind (row, col): ({wind[0]:.2f}, {wind[1]:.2f})", fontsize=13)

    def show_frame(index):
        frame = frames[index]
        picture.set_data(frame["rgb"])
        selected = frames[index-1]["allocation"] if index else []
        decision = f"Last construction: cell {selected[0]}" if selected else "Last construction: none"
        axis.set_title(f"{args.policy} | interval {frame['t']} | cells built {sum(frame['built'])}\n"
                       f"{decision} | no episode cap", fontsize=11)
        return [picture]

    # 3. Save the animation. Playback performs no training.
    animation = FuncAnimation(figure, show_frame, frames=len(frames), interval=400)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=2.5), dpi=110)
    show_frame(min(5, len(frames)-1))
    figure.savefig(args.output.with_name(args.output.stem + "_preview.png"), dpi=110)
    plt.close(figure)
    print(f"Saved {args.output}. Policy: {args.policy}. No training occurs in this demo.")


if __name__ == "__main__":
    main()
