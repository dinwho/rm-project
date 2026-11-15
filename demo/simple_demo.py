"""A small Python-only demo: load a policy, simulate one incident, save a GIF."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save an animation without needing a desktop window.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

from wildfire_crmab import EnvironmentConfig, WildfireEnv, Scenario
from wildfire_crmab.agents.dqn import DQNAgent
from wildfire_crmab.agents.baselines import make_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=["dqn", "dqn_no_wind", "random", "wind_heuristic"], default="dqn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("demo/simple_demo.gif"))
    args = parser.parse_args()

    # 1. Load the trained DQN (or a comparison rule) and its environment.
    if args.policy.startswith("dqn"):
        root = Path(__file__).resolve().parents[1]
        checkpoint = root / "artifacts" / "pilot" / "models" / args.policy / "seed_11" / "best.pt"
        agent, saved = DQNAgent.load(checkpoint)
        env = WildfireEnv(EnvironmentConfig(**saved["environment"]),
                          observe_wind=saved["observe_wind"], budget_scale=saved["budget_scale"])
    else:
        env = WildfireEnv(budget_scale=6)
        agent = make_baseline(args.policy, env.config, seed=args.seed + 123)

    # 2. Start one incident. A shared seed lets different policies face the same world.
    observation, info = env.reset(options={"scenario": Scenario(seed=args.seed, budget=6)})
    frames = [(env.render().copy(), 0, env.budget, "Initial state")]

    # 3. Let the policy choose complete allocations; advance the real simulator.
    while not env.finished:
        action = agent.act(observation, info["action_mask"])
        selected_arms = env.allocations.decode(action)
        observation, reward, terminated, truncated, info = env.step(action)
        decision = "Arms built this interval: " + (", ".join(map(str, selected_arms)) or "none")
        frames.append((env.render().copy(), env.step_number, env.budget, decision))
        print(f"Interval {env.step_number:2}: arms={selected_arms}, materials={env.budget}, reward={reward:.3f}")

    # 4. Turn the recorded simulator images into a simple animation.
    figure, axis = plt.subplots(figsize=(6, 6.7))
    figure.subplots_adjust(top=.83, bottom=.18)
    picture = axis.imshow(frames[0][0], interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    colors = [("Vegetation", "#41874b"), ("Fire", "#f45320"), ("Burned", "#413c37"),
              ("Firebreak", "#cfad75"), ("Settlement", "#487dd2")]
    figure.legend(handles=[Patch(color=color, label=name) for name, color in colors],
                  loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .04))
    figure.text(.5, .02, "One allocation → construct selected segments → one fire interval",
                ha="center", fontsize=9)
    wind_text = f"Fixed wind (row, col): ({env.wind[0]:.2f}, {env.wind[1]:.2f})"

    def show_frame(index):
        image, time, budget, decision = frames[index]
        picture.set_data(image)
        axis.set_title(f"{args.policy} | interval {time} | materials {budget}/6\n{decision}", fontsize=12)
        figure.suptitle("CRMAB wildfire allocation\n" + wind_text, fontsize=13)
        return [picture]

    animation = FuncAnimation(figure, show_frame, frames=len(frames), interval=400)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(args.output, writer=PillowWriter(fps=2.5), dpi=110)
    plt.close(figure)
    print(f"Saved {args.output}. This demonstrates a saved policy; it does not train one.")


if __name__ == "__main__":
    main()
