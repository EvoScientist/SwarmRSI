#!/bin/bash
# One Arbor NatureBench cell under the matched protocol: Arbor in a network-less bwrap sandbox whose
# only exits are a unix-socket CONNECT proxy (api.deepseek.com:443) and the grader (official evaluator,
# <=30 evaluations). env: SHORT TASK [SMOKE=1: sandbox/proxy/grader checks only, no model calls]
set -u
K=$(cd "$(dirname "$0")" && pwd)
: "${CELL_DIR:?Set CELL_DIR to the prepared cell directory}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the SwarmRSI checkout}"
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to the scientific environment}"
: "${TASKS_ROOT:?Set TASKS_ROOT to prepared NatureBench tasks}"
: "${ARBOR_ENV:?Set ARBOR_ENV to the installed Arbor environment}"
: "${KEY_FILE:?Set KEY_FILE to a controller credential file}"
: "${TIKTOKEN_CACHE:?Set TIKTOKEN_CACHE to a prepopulated cache}"
: "${TASK:?Set TASK to the full NatureBench task ID}"
C=$CELL_DIR; SRC=$SOURCE_ROOT; RT=$RUNTIME_ROOT; TR=$TASKS_ROOT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
[ "${SMOKE:-0}" = 1 ] || { [ -e $C/started.json ] && { echo "cell already started; never a second run"; exit 1; }; }
mkdir -p $C/tmp/home $C/sock $C/grader_results
rm -f $C/sock/*.sock
python3 $K/allow_proxy_unix.py $C/sock/proxy.sock > $C/proxy.log 2>&1 & PROXY=$!
cat > $C/grader_cfg.json <<CFG
{"results": "$C/grader_results", "python": "$RT/bin/python", "evaluator": "$SRC/src/evolvcoder/naturebench_evaluator.py",
 "task_root": "$TR/$TASK", "timeout": ${GRADER_TIMEOUT:-2400}, "pythonpath": "$SRC/src", "max_scores": 30, "socket": "$C/sock/grader.sock"}
CFG
$RT/bin/python $K/nb_grader_unix.py $C/spec.json $C/grader_cfg.json > $C/grader.log 2>&1 & GRADER=$!
for i in $(seq 30); do [ -S $C/sock/proxy.sock ] && [ -S $C/sock/grader.sock ] && break; sleep 1; done
PORT=$(python3 -c "import json;print(json.load(open('$C/spec.json'))['port'])")
INSTR="Maximize the official NatureBench score printed by 'bash eval.sh' (higher is better). You have at most 30 official evaluations in total; each call counts. run.py is the solution: it must read DATA_DIR and write OUTPUT_DIR exactly as description.md requires. Do not modify data/ or eval.sh."
SB=(bwrap --die-with-parent --unshare-net --unshare-pid --new-session
  --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /etc /etc --ro-bind /sys /sys
  --proc /proc --dev /dev --ro-bind $RT /opt/science --ro-bind $ARBOR_ENV $ARBOR_ENV --ro-bind $(realpath $ARBOR_ENV) $(realpath $ARBOR_ENV) --ro-bind $(realpath $ARBOR_ENV) $(realpath $ARBOR_ENV) --ro-bind $K /k
  --bind $C/ws /ws --ro-bind $TR/$TASK/problem/data /ws/data --bind $C/tmp /tmp --bind $C/sock /sock
  --ro-bind $KEY_FILE /secrets/ds.key --ro-bind $TIKTOKEN_CACHE /tiktoken_cache)
[ -d /proc/driver/nvidia ] && SB+=(--ro-bind /proc/driver/nvidia /proc/driver/nvidia)
for d in /dev/nvidia*; do [ -e "$d" ] && SB+=(--dev-bind "$d" "$d"); done
SB+=(--chdir /ws --setenv HOME /tmp/home --setenv PATH /opt/science/bin:/usr/local/bin:/usr/bin:/bin
  --setenv HTTPS_PROXY http://127.0.0.1:63101 --setenv HTTP_PROXY http://127.0.0.1:63101 --setenv NO_PROXY localhost,127.0.0.1
  --setenv TIKTOKEN_CACHE_DIR /tiktoken_cache --setenv DATA_DIR /ws/data --setenv OUTPUT_DIR /ws/output --setenv PYTHONDONTWRITEBYTECODE 1
  --setenv ARBOR_VENV $ARBOR_ENV --setenv GRADER_PORT $PORT --setenv ARBOR_INSTRUCTION "$INSTR")
if [ "${SMOKE:-0}" = 1 ]; then
  "${SB[@]}" bash -c '/opt/science/bin/python /k/sandbox_fwd.py 63101=/sock/proxy.sock '"$PORT"'=/sock/grader.sock & sleep 2
    python3 - <<PY
import urllib.request as u, json, os
def get(url, proxy=True):
    h = u.ProxyHandler({"https": "http://127.0.0.1:63101"}) if proxy else u.ProxyHandler({})
    try: return u.build_opener(h).open(url, timeout=20).status
    except Exception as e: return type(e).__name__ + ":" + str(e)[:60]
print("SMOKE deepseek_models", get("https://api.deepseek.com/models"))
print("SMOKE huggingface", get("https://huggingface.co"))
print("SMOKE pypi", get("https://pypi.org/simple/"))
print("SMOKE direct_no_proxy", get("https://api.deepseek.com/models", proxy=False))
print("SMOKE gpu_dev", [d for d in os.listdir("/dev") if d.startswith("nvidia")][:3])
print("SMOKE data_readonly", os.access("/ws/data", os.W_OK) is False)
PY
    bash /ws/eval.sh | head -c 600; echo'
  kill $PROXY $GRADER; exit 0
fi
# Matched protocol: the research config (4 h budget, 20 cycles, no search) goes into the workspace.
cp $K/research_config_nb.yaml $C/ws/research_config.yaml
python3 -c "import json,time,os;json.dump({'job':os.environ.get('SLURM_JOB_ID','manual'),'task':'$TASK','started':time.time(),'host':os.uname().nodename},open('$C/started.json','w'))"
"${SB[@]}" bash /k/arbor_entry.sh > $C/arbor.log 2>&1 & ARBOR=$!
# Matched protocol: stop the cell once its 30 official evaluations are scored.
while kill -0 $ARBOR 2>/dev/null; do
  n=$(ls $C/grader_results 2>/dev/null | grep -c '^[0-9]')
  done_n=$(ls $C/grader_results/*/capture.json 2>/dev/null | wc -l)
  if [ "$n" -ge 30 ] && [ "$done_n" -ge 30 ]; then echo "30 evaluations scored; stopping" >> $C/arbor.log; kill -TERM $ARBOR; sleep 60; kill -KILL $ARBOR 2>/dev/null; break; fi
  sleep 60
done
wait $ARBOR 2>/dev/null
kill $PROXY $GRADER 2>/dev/null
python3 - <<PY
import json, glob, os
best = None; n = 0
for r in sorted(glob.glob("$C/grader_results/[0-9]*/result.json")):
    n += 1
    d = json.load(open(r))
    if d.get("valid") is True:
        g = d["metrics"]["aggregate_improvement"]
        if best is None or g > best[0]: best = (g, os.path.basename(os.path.dirname(r)))
json.dump({"task": "$TASK", "evaluations": n, "best_g": best and best[0], "best_eval": best and best[1]},
          open("$C/final.json", "w"), indent=1)
print(json.dumps({"task": "$TASK", "evaluations": n, "best": best}))
PY
