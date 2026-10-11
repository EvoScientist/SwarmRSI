#!/usr/bin/env python3
"""Fetch a pinned upstream baseline; never executes it or installs dependencies."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    pins = json.loads((ROOT / 'baselines/upstreams.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=sorted(pins))
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    pin = pins[args.name]
    dest = args.destination or ROOT / 'external' / args.name
    if dest.exists():
        raise SystemExit(f'Destination already exists; refusing to overwrite: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), 'remote', 'add', 'origin', pin['url']], check=True)
    subprocess.run(['git', '-C', str(dest), 'fetch', '--depth=1', 'origin', pin['commit']], check=True)
    subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    head = subprocess.check_output(['git', '-C', str(dest), 'rev-parse', 'HEAD'], text=True).strip()
    if head != pin['commit']:
        raise SystemExit('Fetched revision does not match the pinned source')
    print(json.dumps({'name': args.name, 'path': str(dest.resolve()), 'commit': head}))

if __name__ == '__main__':
    main()
