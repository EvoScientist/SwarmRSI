# Baseline integrations

Fetch exact upstream revisions with `scripts/fetch_baseline.py`; install each upstream's dependencies in its own environment. The upstream algorithms and licenses remain upstream. These integrations control task visibility, provider routing, budget accounting and shared evaluation.

| Baseline | Adapter | Benchmark |
|---|---|---|
| AIDE | `scripts/run_aide_naturebench_cell.py`, `src/evolvcoder/baselines/` | NatureBench |
| Arbor | `baselines/arbor/` | NatureBench |
| MLEvolve | `baselines/mlevolve/` | MLE-Bench |
| TusoAI | `baselines/tusoai/` | OpenProblems six tasks |

Biomni and general coding-agent scores cited as external references in the manuscript are not new implementations in this code release.

## AIDE

```bash
python scripts/run_aide_naturebench_cell.py \
  --root /absolute/new-aide-run --task TASK_ID \
  --profile config/naturebench/profile.json --tasks-root /absolute/tasks \
  --runtime-python /absolute/science/bin/python --upstream-root external/aide \
  --secret /absolute/controller-only/ds.key --model deepseek-flash \
  --base-url https://api.deepseek.com --reasoning-param reasoning_effort \
  --tool-choice auto --request-timeout 600 --dry-run
```

The upstream source hash check remains enabled. Remove `--dry-run` to execute. The final integration handles non-UTF8 previews and nullable wall-only budgets; it can bind another scorer checkout using `--evaluator-src`.

## Arbor

`prepare_cell.py` copies a contract-compatible starter and creates the scoring client. `nb_grader_unix.py` executes candidates using the shared official evaluator. `run_cell.sh` is the final deployment launcher with paths supplied through environment variables instead of a cluster account.

```bash
python baselines/arbor/prepare_cell.py --task TASK_ID --task-root /absolute/tasks \
  --starter /absolute/starter/run.py --out /absolute/cell --port 63102
```

Before invoking `bash baselines/arbor/run_cell.sh`, set `CELL_DIR`, `SOURCE_ROOT`, `RUNTIME_ROOT`, `TASKS_ROOT`, `ARBOR_ENV`, `KEY_FILE`, `TIKTOKEN_CACHE`, and `TASK`. Use `SMOKE=1` to check sandbox/grader wiring without model calls (it still probes network policy and grades the starter). The final research config uses 14,400 seconds, 20 cycles, no literature/web search, and 30 official evaluations. Linux bubblewrap, offline scientific dependencies and prepopulated tokenizer cache are required.

## MLEvolve

`mlevolve_config.py` configures a copy of the pinned upstream config (model, endpoint and resource budget); `mlevolve_layout.py` exposes prepared public data in the layout upstream expects. `runtime.py` wraps upstream execution in an offline candidate namespace and records actual provider usage. `grader.py` performs separate official grading.

The deployed runtime expects `/upstream`, `/public`, `/models`, `/output`, `/slots`, and a controller-only `/run/secrets/ds.key` mount, plus `CAMPAIGN_GPUS` (or `CPU_ONLY=1`). Paths inside the wrapper are container mount points, not host defaults. Install upstream dependencies and provide the scientific environment before using it. `transport_hotfix.py` is the recorded optional DeepSeek output-cap integration; apply only when reproducing that transport variant, not silently to another backbone.

## TusoAI

`run_research_baseline.py` is the final v3 research runner; `isolated_bridge.py` scores outputs through the six-task adapter. `ds_guard_v3.py` is the matching controller-side transport/accounting guard. Search and literature initialization use upstream TusoAI at the pinned revision. Start with `--help`; execution is opt-in via `--execute`.

Start from `tusoai/config.example.json` and replace its explicit placeholders. A config must provide the upstream directory, template, work directory, task package, model (`deepseek-flash` for this preserved integration), metric direction, limits, task description, public contract, literature mode, and `bridge_command`. Use `upstream_live` literature or a hash-bound locally extracted paper cache; article contents are not redistributed. The bridge command invokes `isolated_bridge.py` with `{candidate}`, `{task_root}`, `{output_dir}`, the bundled candidate driver and your scientific Python. The guard publishes a local `/api.deepseek.com` endpoint and private client credential; the upstream API key stays in the guard.

Set `EVOLVCODER_TRIAL_STARTED_AT_UNIX` and `OP_TRIAL_DEADLINE_UNIX` to the shared trial interval before execution. The 30-second finalization reserve and total wall include initialization. Deployment scoring remains separate from the search session.

Offline integration checks:

```bash
python -m pytest baselines/tusoai -q
python baselines/mlevolve/test_transport_hotfix.py
```
