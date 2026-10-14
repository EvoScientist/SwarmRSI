# SwarmRSI

**Recursive Self-Improvement through Evolving Scientific Agent Swarms**

SwarmRSI is a framework in which a swarm of scientific agents improves its research by evolving the research states its agents inherit: hypotheses, programs, experimental evidence and unresolved needs. A shared graph records every attempt, including failed attempts and attempts without gain, so each experiment updates the states that later agents start from.

The framework combines three mechanisms:

- **Feedback-Guided Swarm Mutation:** uses other agents' experimental feedback to decide whether to revise a hypothesis (H-mutation) or repair its program (P-mutation).
- **Complementary Swarm Recombination:** tests a retained hypothesis on another agent's program (rebase) and combines components from programs with complementary strengths across instances (crossover).
- **Recursive Swarm Evolution:** evaluates every candidate, returns its outcome to the shared graph, and uses a discounted-UCB index (H/P selection) to decide whether the next step revises a hypothesis or a program.

SwarmRSI supports **OpenProblems**, **NatureBench**, and **MLE-Bench Lite**, with integrations for AIDE, Arbor, MLEvolve, and TusoAI.

## 1. Install

Use Python 3.11 or 3.12. Candidate execution requires Linux and `bubblewrap`.

```bash
git clone <repository-url> SwarmRSI
cd SwarmRSI

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,openproblems,mlebench-insults]' pillow pyarrow

# Ubuntu / Debian
sudo apt-get install bubblewrap

python -m evolvcoder --help
```

The Python package and command-line interface use the name `evolvcoder`.

## 2. Prepare the runtime and credentials

Create a separate scientific Python environment with the packages required by your benchmark tasks. For OpenProblems, install the provided requirements into that environment:

```bash
/absolute/science/bin/python -m pip install -r benchmarks/openproblems/requirements.txt
```

Set the path to that interpreter and your model credential:

```bash
export SCIENCE_PYTHON=/absolute/science/bin/python
export DEEPSEEK_API_KEY="your-api-key"
```

The examples below use the DeepSeek endpoint and the model configured in the supplied profiles. To use another backbone, update the profile's `model` and request settings, and pass its corresponding `--base-url`.

## 3. Run a benchmark

### NatureBench

Prepare the official NatureBench task packages, including public inputs, task descriptions, metadata, and evaluators. The profile contains all 90 task IDs and uses the included cold-start program. See [NatureBench setup](benchmarks/naturebench/README.md) for the expected layout.

```bash
export NB_TASKS_ROOT=/absolute/naturebench/tasks

# Check the task package and execution environment.
python -m evolvcoder run-naturebench \
  --profile config/naturebench/profile.json \
  --tasks-root "$NB_TASKS_ROOT" \
  --task-id s41467-025-63412-3 \
  --runtime-python "$SCIENCE_PYTHON" \
  --credential-env DEEPSEEK_API_KEY \
  --base-url https://api.deepseek.com/v1 \
  --seed 0 --run-dir "$PWD/runs/naturebench/example" --preflight

# Start the search.
python -m evolvcoder run-naturebench \
  --profile config/naturebench/profile.json \
  --tasks-root "$NB_TASKS_ROOT" \
  --task-id s41467-025-63412-3 \
  --runtime-python "$SCIENCE_PYTHON" \
  --credential-env DEEPSEEK_API_KEY \
  --base-url https://api.deepseek.com/v1 \
  --seed 0 --run-dir "$PWD/runs/naturebench/example"
```

Change `--task-id` and `--run-dir` for each experiment. Task budgets and execution settings are configured in `config/naturebench/profile.json`.

### OpenProblems

The adapter supports six tasks: `denoise`, `label`, `batch`, `svg`, `decomp`, and `visual`.

Prepare a dataset manifest following [OpenProblems setup](benchmarks/openproblems/README.md), then build the task package:

```bash
python benchmarks/openproblems/build_package.py \
  --manifest /absolute/denoise-optimization.json \
  --destination data/openproblems/optimization/denoise

python -m evolvcoder run-naturebench \
  --profile config/openproblems/profile.json \
  --tasks-root data/openproblems/optimization \
  --task-id denoise \
  --runtime-python "$SCIENCE_PYTHON" \
  --credential-env DEEPSEEK_API_KEY \
  --base-url https://api.deepseek.com/v1 \
  --seed 0 --run-dir "$PWD/runs/openproblems/denoise" --preflight
```

After preflight succeeds, repeat the run command without `--preflight`. OpenProblems uses the shared `run-naturebench` execution interface. Build optimization and deployment packages separately; use deployment data only to evaluate the selected final program.

### MLE-Bench Lite

Prepare the official MLE-Bench data and grader, then assemble a competition run. The 22 competition IDs are listed in [`lite22.txt`](benchmarks/mlebench/lite22.txt).

```bash
python deploy/mle/run_matched.py detecting-insults-in-social-commentary \
  --root "$PWD/runs/mlebench" \
  --prepared-root /absolute/prepared/data \
  --mlebench-data-dir /absolute/mlebench/registry \
  --official-source /absolute/mle-bench \
  --python "$SCIENCE_PYTHON" \
  --model deepseek-flash \
  --base-url https://api.deepseek.com/v1 \
  --seed 0 --internal-validation-policy controller_only
```

This generates a task package, profile, and run configuration. Set the generated configuration's credential source to your environment variable:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/mlebench/configs/detecting-insults-in-social-commentary.json")
config = json.loads(path.read_text())
config["model"]["api_key_file"] = None
config["model"]["api_key_env"] = "DEEPSEEK_API_KEY"
path.write_text(json.dumps(config, indent=2) + "\n")
PY

python -m evolvcoder preflight \
  --config runs/mlebench/configs/detecting-insults-in-social-commentary.json

python -m evolvcoder run \
  --config runs/mlebench/configs/detecting-insults-in-social-commentary.json
```

Configure resource limits and any offline model-cache paths for your machine before running. See [MLE-Bench setup](benchmarks/mlebench/README.md) for data preparation and official terminal grading.

## 4. Run baselines

Fetch the upstream implementations:

```bash
python scripts/fetch_baseline.py aide
python scripts/fetch_baseline.py arbor
python scripts/fetch_baseline.py mlevolve
python scripts/fetch_baseline.py tusoai
```

Install each baseline's dependencies in a separate environment, then follow the corresponding commands in [Baseline setup](baselines/README.md).

| Baseline | Benchmark | Integration |
|---|---|---|
| AIDE | NatureBench | `scripts/run_aide_naturebench_cell.py` |
| Arbor | NatureBench | `baselines/arbor/` |
| MLEvolve | MLE-Bench | `baselines/mlevolve/` |
| TusoAI | OpenProblems | `baselines/tusoai/` |

## 5. Inspect results

Search runs save candidate programs, evaluation records, and final reports under the chosen run directory.

```bash
python -m evolvcoder report --run /absolute/completed-run
```

NatureBench reports aggregate improvement over the reference method. OpenProblems reports task-specific scores. MLE-Bench uses public validation during search and official private-test grading for the final submission.

## Development

```bash
python -m pytest tests -q

# Six-task OpenProblems interface check
PYTHONPATH=src "$SCIENCE_PYTHON" benchmarks/openproblems/synthetic_smoke.py
```

| Directory | Contents |
|---|---|
| `src/evolvcoder/` | Search, planning, coding, execution, and evaluation |
| `config/` | Search settings, benchmark profiles, and starters |
| `benchmarks/` | Benchmark setup and OpenProblems task adapters |
| `deploy/mle/` | MLE-Bench task and run preparation |
| `baselines/` | Baseline integrations and setup |
| `tests/` | Regression tests |

## License

SwarmRSI is released under the [Apache-2.0 License](LICENSE). See [Third-party notices](THIRD_PARTY_NOTICES.md) for dependency and upstream attribution.
