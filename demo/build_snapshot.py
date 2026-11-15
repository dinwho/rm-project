"""Build an offline replay; include DQN only when matching saved weights exist."""
import json
from pathlib import Path
from engine import POLICIES, rollout, experiment_results

folder = Path(__file__).parent
episodes = {}
unavailable = {}
for policy in POLICIES:
    try:
        episodes[policy] = rollout(policy)
    except FileNotFoundError as error:
        unavailable[policy] = str(error)
data = {"episodes": episodes, "unavailable": unavailable, "results": experiment_results()}
html = (folder/"static"/"index.html").read_text()
html = html.replace('<link rel="stylesheet" href="styles.css">', '<style>'+(folder/"static"/"styles.css").read_text()+'</style>')
script = '<script>window.DEMO_DATA='+json.dumps(data, allow_nan=False).replace('<', '\\u003c')+';</script>'
html = html.replace('<script src="app.js"></script>', script+'<script>'+(folder/"static"/"app.js").read_text()+'</script>')
(folder/"demo.html").write_text(html)
print(f"{folder/'demo.html'}: {len(episodes)} real policy replays; {len(unavailable)} unavailable saved models")
