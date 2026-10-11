"""(Snellius) Build an Arbor workspace for one NatureBench task, matched to the BRS protocol.

The workspace starts from the same root program BRS starts from (the task's starter run.py),
sees the same public data (mounted read-only at data/ when Arbor runs), and is scored by the
same official evaluator: eval.sh sends a snapshot of the code to a grading process that runs
run.py in the offline bwrap sandbox and evaluates it. NatureBench has one official evaluation,
so `eval.sh dev` and `eval.sh test` return the same number (Arbor's single-split mode).
"""
import argparse, json, shutil
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--task-root", required=True, type=Path)
p.add_argument("--starter", required=True, type=Path, help="the cold-start run.py BRS starts from")
p.add_argument("--out", required=True, type=Path); p.add_argument("--port", required=True, type=int)
a = p.parse_args()
ws = a.out / "ws"; ws.mkdir(parents=True, exist_ok=True)
shutil.copy2(a.starter, ws / "run.py")
readme = (a.task_root / a.task / "problem/README.md").read_text()
(ws / "description.md").write_text(readme + """

---
## Protocol for this run (read carefully)

- `run.py` is the solution. It reads the public inputs from the directory in the environment variable
  `DATA_DIR` (here `data/`, read-only) and writes its outputs to `OUTPUT_DIR` (here `output/`), in exactly
  the layout and file names the task README above requires. Keep that contract.
- `bash eval.sh` runs `run.py` in an isolated, network-less sandbox and scores its outputs with the official
  NatureBench evaluator. It prints `score: <value>` (aggregate improvement over the published SOTA; higher is
  better) or an error with the evaluator's reason. `bash eval.sh test` returns the same official score.
- Budget: at most 30 official evaluations for the whole run. Every `bash eval.sh` call counts (dev or
  test, valid or not) and each reply says how many remain; after that nothing more is scored. The result
  of this run is the best official score among those evaluations, so check changes yourself on the public
  data first and spend evaluations only on versions you expect to improve.
- No pretrained-model files are available offline, here or inside the evaluator.
- Only `run.py` and the small code/text files next to it are sent to the evaluator; files over 5 MB,
  `data/`, `output/` and `.git` are not. The evaluator has no network: download nothing and use only the
  Python packages already installed.
""")
(ws / "README.md").write_text(f"# NatureBench {a.task}\n\nMaximize the score printed by `bash eval.sh` (higher is better; at most 30 evaluations). "
                              "Do not modify `data/` or `eval.sh`. See `description.md`.\n")
(ws / "eval.sh").write_text(f"""#!/usr/bin/env bash
# Sends this workspace's code to the grading process (official NatureBench evaluator, offline sandbox).
set -euo pipefail
SPLIT="${{1:-dev}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
# Code only: no data/, output/, .git, .arbor, nothing over 5 MB (weights cannot ride along).
find . -type f -size -5M ! -path './data/*' ! -path './output/*' ! -path './.git/*' ! -path './.arbor/*' -print0 \
  | tar --null -T - -czf - \
  | /opt/science/bin/python -c "import sys,urllib.request as u; print(u.build_opener(u.ProxyHandler({{}})).open(u.Request('http://127.0.0.1:{a.port}/score?split=$SPLIT', data=sys.stdin.buffer.read(), method='POST'), timeout=7500).read().decode())"
""")
(ws / "eval.sh").chmod(0o755)
(ws / ".gitignore").write_text("data/\noutput/\n")
json.dump({"task": a.task, "port": a.port, "starter": str(a.starter)}, open(a.out / "spec.json", "w"), indent=1)
print(json.dumps({"ws": str(ws), "starter": str(a.starter)}))
