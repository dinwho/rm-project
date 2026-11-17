"""Thin Gymnasium bridge to the pinned, original Firehose Python/C++ environment.

Preserves original observations, rewards, action masks, warm-up and step timing.
Only API conversion, pipe timeouts, deterministic ignition sampling and resource
cleanup are added. In particular the upstream reset observation remains unpainted
by warm-up fire, and upstream grid value1 includes burning AND burned cells.
"""
import contextlib
import io
import queue
import sys
import threading
from types import MethodType, SimpleNamespace
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .paths import RUNTIME, REFERENCE, UPSTREAM_COMMIT
from .setup import ensure_runtime

PRESETS = {"Sub20x20": (20, 8, 372), "Sub40x40": (25, 5, 909),
           "mit_m": (25, 5, None), "mit_i": (25, 5, None), "mit_t": (25, 5, None)}


def upstream_imports():
    ensure_runtime(build=False)
    for path in (RUNTIME, RUNTIME / "cell2fire"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from cell2fire.gym_env import FireEnv
    from firehose.helpers import IgnitionPoint, IgnitionPoints
    from firehose.rewards import REWARD_FUNCTIONS
    import firehose.config as configuration
    configuration.TRAINING_ENABLED = True
    return FireEnv, IgnitionPoint, IgnitionPoints, REWARD_FUNCTIONS


class OriginalFirehoseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, map_name="Sub20x20", ignition=372, seed=123, max_steps=2000,
                 reward="FireSizeReward", steps_before_sim=None, steps_per_action=None,
                 timeout=30, render_mode="rgb_array"):
        super().__init__()
        if map_name not in PRESETS:
            raise ValueError(f"Unsupported map; choose one of {tuple(PRESETS)}")
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Positive horizon and pipe timeout required")
        FireEnv, Point, Points, rewards = upstream_imports()
        if reward not in rewards:
            raise ValueError(f"Unknown upstream reward {reward!r}")
        shape = np.loadtxt(REFERENCE / "data" / map_name / "Forest.asc", skiprows=6).shape
        if ignition is not None and (type(ignition) is not int or not 1 <= ignition <= np.prod(shape)):
            raise ValueError("Ignition is a one-based in-bounds cell index, or None for random")
        self.map_name, self.ignition, self.simulator_seed = map_name, ignition, int(seed)
        self.render_mode, self.timeout = render_mode, float(timeout)
        before, per_action, _ = PRESETS[map_name]
        self.steps_before_sim = before if steps_before_sim is None else steps_before_sim
        self.steps_per_action = per_action if steps_per_action is None else steps_per_action
        self.config = SimpleNamespace(map_name=map_name, ignition=ignition, simulator_seed=self.simulator_seed, max_steps=max_steps, reward=reward, steps_before_sim=self.steps_before_sim, steps_per_action=self.steps_per_action)
        points = None if ignition is None else Points([Point(ignition, 1, (ignition-1)%shape[1], (ignition-1)//shape[1])])
        with contextlib.redirect_stdout(io.StringIO()):
            self.legacy = FireEnv(fire_map=map_name, action_type="flat", observation_type="forest_rgb",
                max_steps=max_steps, output_dir=str(RUNTIME / "runs"), ignition_points=points,
                reward_func_cls=rewards[reward], steps_before_sim=self.steps_before_sim,
                steps_per_action=self.steps_per_action, action_diameter=1)
        self.base = self.raw = self.legacy
        self.action_space = spaces.Discrete(self.legacy.action_space.n)
        self.observation_space = spaces.Box(0, 255, (3, *shape), np.uint8)
        self.finished, self._closed = True, False
        self.episode_return = 0.0
        self._install_process_guards()

    def _install_process_guards(self):
        process = self.legacy.fire_process
        spawn = process.spawn
        command = process.get_command_str
        owner = self
        def command_with_seed(this):
            return command().replace("--seed 123 ", f"--seed {owner.simulator_seed} ")
        def guarded_spawn(this):
            spawn()
            output = queue.Queue()
            this._output_queue = output
            this._stderr_lines = []
            def stdout_reader():
                for line in iter(this.process.stdout.readline, b""):
                    output.put(line)
                output.put(None)
            def stderr_reader():
                for line in iter(this.process.stderr.readline, b""):
                    this._stderr_lines.append(line.decode("utf8", errors="replace"))
            threading.Thread(target=stdout_reader, daemon=True).start()
            threading.Thread(target=stderr_reader, daemon=True).start()
        def timed_read(this):
            try:
                line = this._output_queue.get(timeout=owner.timeout)
            except queue.Empty as error:
                this.kill()
                raise TimeoutError("Cell2Fire did not reach its next action boundary") from error
            if line is None:
                code = this.process.wait(timeout=2)
                if code:
                    raise RuntimeError(f"Cell2Fire exited with status {code}: {''.join(this._stderr_lines)[-2000:]}")
                return ""
            text = line.strip().decode("cp1252")
            this.lines.append(text)
            return text
        process.get_command_str = MethodType(command_with_seed, process)
        process.spawn = MethodType(guarded_spawn, process)
        process.read_line = MethodType(timed_read, process)

    def _observation(self, observation):
        return np.ascontiguousarray(observation.transpose(2, 0, 1), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Environment was closed")
        # Seed ignition selection while keeping the upstream default simulator seed123.
        # The two sources of randomness are intentionally distinguished.
        state = np.random.get_state()
        try:
            np.random.seed(seed if seed is not None else int(self.np_random.integers(2**32)))
            with contextlib.redirect_stdout(io.StringIO()):
                observation = self.legacy.reset(**(options or {}))
        except BaseException:
            self.legacy.fire_process.kill()
            raise
        finally:
            np.random.set_state(state)
        self.finished = False
        self.episode_return = 0.0
        return self._observation(observation), self.info()

    def action_masks(self):
        return np.asarray(self.legacy.action_masks(), dtype=bool)

    def step(self, action):
        if self.finished:
            raise RuntimeError("Reset before stepping a finished episode")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("Action must be an integer")
        action = int(action)
        # -1 is the existing upstream no-op; only external baseline callers use it.
        if action != -1 and not self.action_space.contains(action):
            raise ValueError("Action is outside the grid")
        # Upstream accepts repeated actions; Maskable PPO avoids them via the mask.
        # Random baseline callers retain the original ability to repeat a cell.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                observation, reward, done, _ = self.legacy.step(action)
        except BaseException:
            self.legacy.fire_process.kill()
            raise
        terminated = bool(self.legacy.fire_process.finished)
        truncated = bool(done and not terminated)
        self.finished = bool(done)
        self.episode_return += float(reward)
        if self.finished:
            self.legacy.fire_process.kill()
        return self._observation(observation), float(reward), terminated, truncated, self.info()

    def info(self):
        raw = self.legacy
        protected = None if raw.reward_mask is None else len(raw.reward_mask[0])
        return {"action_mask": self.action_masks(), "step": raw.iter,
                "fire_affected_cells": int(np.count_nonzero(raw.state > 0)),
                "ever_observed_fire_cells": len(raw.cells_burned),
                "harvested_cells": len(raw.cells_harvested),
                "protected_cells": protected,
                "protected_affected_cells": None if protected is None else int(np.count_nonzero(raw.state[raw.reward_mask] > 0)),
                "episode_return": self.episode_return, "simulator_finished": bool(raw.fire_process.finished),
                "ignition_index": int(raw.ignition_points.points[0].idx),
                "simulator_seed": self.simulator_seed, "upstream_commit": UPSTREAM_COMMIT}

    def render(self):
        return self.legacy.get_painted_image().copy()

    def close(self):
        if not self._closed:
            self.legacy.fire_process.kill()
            with contextlib.redirect_stdout(io.StringIO()):
                self.legacy.close()
            self._closed = True


def make_env(map_name="Sub20x20", ignition=372, seed=123, max_steps=2000, **kwargs):
    return OriginalFirehoseEnv(map_name, ignition, seed, max_steps, **kwargs)
