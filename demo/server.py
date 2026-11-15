"""Local demo server: python demo/server.py --port 8765."""
import argparse
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from engine import rollout, experiment_results


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            try:
                if url.path == "/api/results":
                    data = experiment_results()
                elif url.path == "/api/episode":
                    params = parse_qs(url.query)
                    data = rollout(params.get("policy", ["dqn"])[0],
                                   int(params.get("seed", [42])[0]), int(params.get("budget", [6])[0]))
                else:
                    raise ValueError("Unknown API route")
                status = 200
            except (ValueError, FileNotFoundError) as error:
                data, status = {"error": str(error)}, 400
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = partial(Handler, directory=str(Path(__file__).parent/"static"))
    print(f"Research demo: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), handler).serve_forever()
