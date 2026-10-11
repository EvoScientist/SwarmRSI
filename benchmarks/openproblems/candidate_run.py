"""Candidate-visible driver; consumes only sanitized data and function solution.py."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    data = Path(os.environ['DATA_DIR'])
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / 'inputs.json').read_text())
    if manifest['seed'] != 0:
        raise ValueError('campaign permits only seed 0')
    for row in manifest['datasets']:
        arguments = []
        for arg in row['arguments']:
            arguments.append(arg.replace('{DATA}', str(data)).replace('{OUTPUT}', str(output)))
        subprocess.run([sys.executable, str(data / 'runner.py'), *arguments,
                        '--solution', str(Path(__file__).parent / 'solution.py')], check=True)

if __name__ == '__main__':
    main()
