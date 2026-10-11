# HPCoEvolve

Hypothesis–program coevolution for scientific coding. This repository contains the stable **BFC f** search implementation, benchmark adapters, and baseline integrations used in the research project.

## Contents

- `src/evolvcoder/`: hypothesis/program frontiers, coupled selection, adaptation, planner/coder, isolated execution, and evaluation. The original Python package name is retained for compatibility.
- `benchmarks/openproblems/`: all six tasks—denoising, label prediction, batch integration, spatially variable genes, spatial decomposition, and dimensionality reduction.
- `benchmarks/naturebench/`, `config/naturebench/`: NatureBench's 90-task inventory and the stable f run profile.
- `benchmarks/mlebench/`, `deploy/mle/`: MLE-Bench Lite task preparation, public validation, and separate official terminal grading.
- `baselines/` and `src/evolvcoder/baselines/`: AIDE, Arbor, MLEvolve, and TusoAI integrations with pinned upstream sources.
- `tests/`: implementation regression tests. Tests inside benchmark/baseline directories cover those integrations.

No datasets, generated solutions, trajectories, credentials, experiment logs, paper files, or historical working directories are included.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,openproblems,mlebench-insults]' pillow pyarrow
python -m evolvcoder --help
python -m pytest tests -q
```

Candidate execution requires Linux with working `bubblewrap` and a separate scientific Python environment containing each task's dependencies. Controller installation alone does not prepare datasets or qualify a GPU runtime. See the README in each benchmark directory for entrypoints and preflight commands.

## Source version

The frozen f core is BFC commit `fd98216ae41c62ef853020428e8180bec0316dde`. This export uses its tested integration descendant `8e3d80f4ba0d9d402b08152df95865f5651a7d6e`, which includes PID namespace isolation, configurable model request settings, optional additional actions after a target, and grader compatibility fixes. It is **not a byte-identical export of every historical experiment**.

The active f search configuration is `config/search/stable_f.json`: `bidirectional_research` with `local_burst_counts_recombination` and `realization_diagnosis` enabled. Other optional research policies remain internal compatibility code; their presence does not mean they were enabled in the paper experiments. Benchmark profiles specify the actual enabled settings.

The six-task OpenProblems adapter and final AIDE/Arbor deployment integrations were maintained outside the frozen core and are included separately. `SOURCE_VERSION.json` records their source identities. Public profiles replace machine-specific paths; verify your local assets with preflight before a run.

## Baselines

```bash
python scripts/fetch_baseline.py aide
python scripts/fetch_baseline.py arbor
python scripts/fetch_baseline.py mlevolve
python scripts/fetch_baseline.py tusoai
```

These commands fetch pinned upstream source into ignored `external/` directories; they do not install packages or call a model. See [baseline setup](baselines/README.md). Upstream search algorithms are not reimplemented here. Published reference-only systems are not presented as runnable implementations.

## License

Project code is Apache-2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for upstream and adapted-component attribution.
