# NatureBench

The installed adapter is `src/swarmrsi/naturebench_session.py`, with isolated execution and official scoring in `naturebench_evaluator.py`. `config/naturebench/profile.json` provides all 90 task IDs, declared task budgets, and the final f search/coder settings. It is a portable run template, not a manifest of every historical task-specific runtime override.

Obtain official NatureBench task assets separately (FrontisAI/NatureBench on Hugging Face). Preserve the task README, public data, hidden evaluation assets, metadata, and custody records expected by the bridge. The included `config/starters/naturebench_cold_start_v1/run.py` is the actual generic cold-start root. It creates placeholders, not a scientific solution; the three initial drafts implement task-specific programs. The template retains the strict Surpass target of 0.1 (stored under the legacy `claude_code_score` key). Point `starter` to a different prepared starter only when that is your declared protocol. Resolve profile paths relative to the profile file.

```bash
python -m swarmrsi run-naturebench \
  --profile config/naturebench/profile.json \
  --tasks-root /absolute/prepared/naturebench/tasks \
  --task-id s41467-025-63412-3 \
  --runtime-python /absolute/science/bin/python \
  --credential-env DEEPSEEK_API_KEY \
  --base-url https://api.deepseek.com/v1 \
  --seed 0 --run-dir /absolute/new-run --preflight
```

Preflight checks local assets/runtime and does not start search. Remove `--preflight` only for a funded execution with a fresh run directory. Set the profile's model and matching endpoint explicitly for other backbones. Task runtime dependencies and evaluator timeouts differ by task; configure them before freezing a run.

AIDE uses `scripts/run_aide_naturebench_cell.py`; Arbor uses `baselines/arbor/`. Both use the shared NatureBench evaluator. Aggregate improvement determines Match (`>= 0`) and Surpass (`> 0.1`); an invalid run is not a successful result.
