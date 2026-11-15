"""Build a portable offline HTML demo with real model replays for one world."""
import json
from pathlib import Path
from engine import POLICIES, rollout, experiment_results

folder = Path(__file__).parent
data = {"episodes": {p: rollout(p) for p in POLICIES}, "results": experiment_results()}
html = (folder/"static"/"index.html").read_text()
html = html.replace('<link rel="stylesheet" href="styles.css">', '<style>'+(folder/"static"/"styles.css").read_text()+'</style>')
script = '<script>window.DEMO_DATA='+json.dumps(data, allow_nan=False).replace('<', '\\u003c')+';</script>'
html = html.replace('<script src="app.js"></script>', script+'<script>'+(folder/"static"/"app.js").read_text()+'</script>')
(folder/"demo.html").write_text(html)
print(folder/"demo.html")
