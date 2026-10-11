"""Cold-start root: a runnable program that solves nothing, for tasks with no baseline.

NatureBench gives an agent a data directory and an output directory and nothing
else; the built-in agents write their solution from scratch. This search needs a
root program to evolve from, and hand-writing one per task costs a day and hands a
reviewer a degree of freedom we cannot defend: we would be choosing the baseline
our own improvement is measured against.

So the root does the one thing every task's contract shares and nothing that any
task's contract makes specific. It mirrors DATA_DIR's instance directories into
OUTPUT_DIR, drops one placeholder file in each, and exits zero. The placeholder is
there because the execution smoke rejects an output tree holding no files at all.
It carries an empty JSON object under a name no task asks for, so no evaluator can
read it as a submission and the root's own metric stays absent rather than
invented. The first coder action reads the README and the data with the tools it
already has, writes the real program, and replaces the placeholder with whatever
that task's contract actually asks for.
"""

from __future__ import annotations

import os
from pathlib import Path

# Deliberately not any task's expected artifact name: this must not parse as a submission.
PLACEHOLDER = "cold_start_placeholder.json"


def instance_names(data_dir: Path) -> list[str]:
    """Every immediate sub-directory of DATA_DIR is one scored instance.

    Tasks that ship their data as loose files at the top level have no instance
    layer; the empty list is correct for them and the caller writes at the root.
    """
    if not data_dir.is_dir():
        return []
    return sorted(entry.name for entry in data_dir.iterdir() if entry.is_dir())


def main() -> None:
    data_dir = Path(os.environ["DATA_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in instance_names(data_dir) or [""]:
        target = output_dir / name if name else output_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / PLACEHOLDER).write_text("{}\n")


main()
