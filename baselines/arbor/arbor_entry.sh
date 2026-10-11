#!/usr/bin/env bash
# Runs inside the Arbor bwrap sandbox. The key goes only into the sandbox-private user config
# (HOME is on the private /tmp): never on a command line, never into the workspace.
set -uo pipefail
/opt/science/bin/python /k/sandbox_fwd.py 63101=/sock/proxy.sock "$GRADER_PORT=/sock/grader.sock" &
mkdir -p "$HOME/.arbor"
"$ARBOR_VENV/bin/python" - <<'PY'
import os, yaml
k = open("/secrets/ds.key").read().strip()
path = os.path.expanduser("~/.arbor/config.yaml")
with open(path, "w") as f:
    f.write("llm:\n  provider: litellm\n  model: openai/deepseek-flash\n"
            f"  api_key: {k}\n  base_url: https://api.deepseek.com\n  max_tokens: 32768\n")
os.chmod(path, 0o600)
assert yaml.safe_load(open(path))["llm"]["api_key"]
print("user config ok")
PY
git config --global user.email arbor@local
git config --global user.name arbor
[ -d .git ] || git init -q
git add -A && (git diff --cached --quiet || git commit -qm "harness: task files and config")
timeout --signal=TERM --kill-after=60s 15000 "$ARBOR_VENV/bin/arbor" run "$ARBOR_INSTRUCTION" --yes --yes-cwd /ws --no-webui
echo "arbor exit $?"
