# OpenProblems: six-task adapter

Tasks: `denoise`, `label`, `batch`, `svg`, `decomp`, `visual`.

`build_package.py` makes a sealed task package for the unchanged f core. `candidate_run.py` and the bundled `runners/` consume sanitized public inputs; `output_evaluator.py` reads predictions independently and holds evaluation truth outside the candidate workspace. Optimization and deployment packages are separate.

Install `requirements.txt` into the scientific runtime. Supply a JSON manifest with `task`, `role` (`optimization` or `deployment`), `metric_protocol`, `metric_audit_approved`, and `datasets`. Each dataset has a unique `name`, named `inputs`, and a controller-only `gold` H5AD path. Only set `metric_audit_approved` after checking the intended metric and data split. The builder refuses unapproved manifests and existing destinations.

```bash
python benchmarks/openproblems/build_package.py \
  --manifest /absolute/denoise-optimization.json \
  --destination data/openproblems/optimization/denoise
python -m swarmrsi run-naturebench \
  --profile config/openproblems/profile.json \
  --tasks-root data/openproblems/optimization --task-id denoise \
  --runtime-python /absolute/science/bin/python \
  --credential-env DEEPSEEK_API_KEY --base-url https://api.deepseek.com/v1 \
  --seed 0 --run-dir /absolute/new-run --preflight
```

The command name `run-naturebench` denotes the existing generic file/metric runtime here; package metadata identifies OpenProblems. Never supply deployment packages to an active search. Freeze the final source, execute it against deployment inputs, then score its outputs separately.

The adapter retains the actual campaign metric semantics: SVG reports released mean AUC/F1 plus literal Accuracy; batch uses strong directed graph connectivity and the declared PCA/neighbors preprocessing; Visual uses historical trustworthiness-15, SVD-500 distance correlation and density-30 components. SVG's mapping to the paper's Accuracy and Visual's normalized aggregate are reconstructed protocols, not verified exact reproductions of published TusoAI numbers.

Visual optimization can use the exact controller reference cache (`visual_reference_cache.py` and `publish_cached_visual.py`). The cache binds gold hashes, observation identities, implementation, and library versions; it does not change candidate-dependent scoring. The original campaign used cached optimization and uncached deployment.

Local interface check (no model calls, synthetic data only):

```bash
PYTHONPATH=src python benchmarks/openproblems/synthetic_smoke.py
PYTHONPATH=src:benchmarks/openproblems:benchmarks/openproblems/runners \
  python -m pytest benchmarks/openproblems -q
```
