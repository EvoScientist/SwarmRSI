# MLE-Bench Lite

Use the official low-complexity split (22 competitions) from openai/mle-bench commit `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`. Obtain competition data and accept their terms separately. This repository does not distribute data or official answers.

- `deploy/mle/prepare_one.py`: prepare a competition with the official benchmark.
- `deploy/mle/make_task_root.py`: separate public search validation from official private-test grading; includes image, page, record and subject-aware projections.
- `deploy/mle/run_matched.py`: assemble a run package/profile/config and launcher.
- `src/swarmrsi/mlebench_run.py`: preflight, search, retained candidate delivery and official terminal grading.
- `deploy/envelopes/mlebench.json`: declared resource envelope; override paths and choose budgets explicitly.

```bash
python deploy/mle/run_matched.py --help
python deploy/mle/run_matched.py detecting-insults-in-social-commentary \
  --root /absolute/new-campaign \
  --prepared-root /absolute/prepared/data \
  --mlebench-data-dir /absolute/mlebench/registry \
  --official-source /absolute/mle-bench \
  --secret /absolute/controller-only/credential.json \
  --model deepseek-flash --base-url https://api.deepseek.com/v1 \
  --seed 0 --internal-validation-policy controller_only
```

Inspect the generated configuration and use the installed CLI's `preflight` entrypoint before starting a run. Pass the scientific runtime, container image and resource options for your machine; historical image names are not downloadable images supplied by this repository. The generator's default budget is not a claim that all paper runs used that budget.

Public validation selects candidates. Only an artifact-bound official terminal report establishes a medal. Private terminal results must not be used as search feedback. `baselines/mlevolve/` integrates the pinned upstream baseline.
