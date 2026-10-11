"""Point upstream MLEvolve's config at our backbone and match our budget.

Only the model route and the resource budget change; every search parameter
(branching, fusion, evolution, memory) is left exactly as upstream ships it.
"""
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
parser.add_argument("--api-key-placeholder", default="@OPENROUTER_KEY@")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--time-limit", type=int, default=21600)
parser.add_argument("--exec-timeout", type=int, default=3600)
parser.add_argument("--embedding-device", default="cpu")
args = parser.parse_args()

p = Path(args.config)
s = p.read_text()
s = s.replace("model: gemini-3-pro-preview", f"model: {args.model}")
s = re.sub(r'base_url: ""\s*#[^\n]*', f'base_url: "{args.base_url}"', s)
s = re.sub(r'api_key: ""\s*#[^\n]*', f'api_key: "{args.api_key_placeholder}"', s)
s = re.sub(r"^  steps: \d+", f"  steps: {args.steps}", s, flags=re.M)
s = re.sub(r"^  time_limit: \d+.*$", f"  time_limit: {args.time_limit}", s, flags=re.M)
s = re.sub(r"^  timeout: \d+", f"  timeout: {args.exec_timeout}", s, flags=re.M)
s = s.replace("memory_embedding_device: cuda", f"memory_embedding_device: {args.embedding_device}")
p.write_text(s)
for line in s.splitlines():
    if re.match(r"\s*(model|base_url|api_key|steps|time_limit|timeout|memory_embedding_device):", line):
        print(line)
