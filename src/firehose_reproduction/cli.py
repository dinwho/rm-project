"""Commands for the original simulator, checkpoint evaluation and demo."""
import argparse
from pathlib import Path
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Fetch pinned upstream source and compile Cell2Fire")
    demo = commands.add_parser("demo", help="Serve the recorded real-simulator demo")
    demo.add_argument("--port", type=int, default=8765)
    commands.add_parser("record", help="Run the original policies and generate demo recordings")
    train = commands.add_parser("train", help="Train a fresh original-CNN Maskable PPO policy")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=5_000_000)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--random-ignition", action="store_true")
    args = parser.parse_args()
    if args.command == "setup":
        from .setup import ensure_runtime
        print(ensure_runtime())
    elif args.command == "record":
        from .recording import generate
        generate()
    elif args.command == "train":
        from .training import train
        print(train(args.output, steps=args.steps, seed=args.seed,
                    ignition=None if args.random_ignition else 372))
    else:
        directory = ROOT / "firehose_demo"
        if not (directory / "data.json").exists():
            parser.error("Run firehose record first to generate simulator episodes")
        server = HTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(directory)))
        print(f"Firehose demo: http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
