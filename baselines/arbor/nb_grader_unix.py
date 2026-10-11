"""(Snellius: served on a unix socket bound into the Arbor sandbox.) Score an Arbor NatureBench workspace with the official evaluator, one snapshot per request.

Copies run.py and the small files beside it (no data/, output/, .git, .arbor, nothing over 5 MB)
into a fresh directory and runs naturebench_evaluator.py in full mode: execute in the offline bwrap
sandbox, then evaluate. Every request and its result is kept under results/.
"""
import io, json, os, shutil, subprocess, sys, tarfile, threading, time
from http.server import BaseHTTPRequestHandler
import socketserver
from pathlib import Path
from urllib.parse import urlparse, parse_qs
spec = json.load(open(sys.argv[1])); cfg = json.load(open(sys.argv[2]))
results = Path(cfg["results"]); results.mkdir(parents=True, exist_ok=True)
gate = threading.Semaphore(2); lock = threading.Lock()
MAX = int(cfg["max_scores"])  # official evaluations per run; counted across grader restarts
counter = [len([p for p in results.iterdir() if p.name.isdigit()])]
SKIP = {"data", "output", ".git", ".arbor", "__pycache__"}

def snapshot(body: bytes, dest: Path):
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = [x for x in Path(m.name).parts if x not in (".", "")]
            if (not m.isfile() or m.size > 5 * 1024**2 or not parts or parts[0] in SKIP
                    or ".." in parts or Path(m.name).is_absolute()):
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())

def score(body: bytes, split: str) -> str:
    with lock:
        if counter[0] >= MAX:
            return (f"error: official evaluation budget exhausted ({MAX} of {MAX} used). Nothing more will be "
                    "scored; the result of this run is the best official score already obtained.")
        counter[0] += 1; n = counter[0]
    root = results / f"{n:04d}"; code = root / "workspace"; code.mkdir(parents=True)
    snapshot(body, code)
    if not (code / "run.py").is_file():
        return "error: run.py is missing from the workspace root"
    (code / "output").mkdir(exist_ok=True)
    t0 = time.time()
    with gate:
        run = subprocess.run([cfg["python"], cfg["evaluator"], "--workspace", str(code), "--result", str(root / "result.json"),
                              "--task-root", cfg["task_root"], "--runtime-python", cfg["python"],
                              "--timeout-seconds", str(cfg["timeout"]), "--require-candidate-sandbox"],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": cfg["pythonpath"]})
    (root / "capture.json").write_text(json.dumps({"split": split, "returncode": run.returncode, "seconds": time.time() - t0,
                                                   "stdout": run.stdout[-4000:], "stderr": run.stderr[-4000:]}))
    try:
        res = json.load(open(root / "result.json"))
    except Exception:
        return f"error: evaluator produced no result (exit {run.returncode}): {run.stderr[-600:]}"
    if res.get("valid") is True:
        return f"score: {res['metrics']['aggregate_improvement']:.6f}  (official NatureBench, evaluation {n} of {MAX}; {MAX - n} left)"
    reason = json.dumps({k: res.get(k) for k in ("category", "reason", "detail", "error") if res.get(k)})[:1500]
    tail = ""
    ev = code / ".evolvcoder/execution_validation.json"
    if ev.is_file():
        tail = json.dumps(json.load(open(ev)))[-1500:]
    return f"error: invalid candidate (evaluation {n} of {MAX}; {MAX - n} left) {reason} {tail}"

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        q = parse_qs(urlparse(self.path).query); split = q.get("split", ["dev"])[0]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            out = score(body, split)
        except Exception as e:
            out = f"error: could not score workspace: {type(e).__name__}: {e}"
        self.send_response(200); self.end_headers(); self.wfile.write(out.encode())
    def log_message(self, *a):
        pass

class UnixHTTP(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

class HU(H):
    def address_string(self):
        return "sandbox"

UnixHTTP(cfg["socket"], HU).serve_forever()
