"""Terminal-Bench benchmark adapter.

The candidate sees the official instruction and nothing else: no tests, no oracle
solution, no verifier image contents. The hidden suite lives inside the pinned
verifier image and is reached only by the official harbor harness, once, on the
solution the search finally selects.

There is no per-task reference threshold here on purpose. Terminal-Bench reports
one binary outcome per trial and the public leaderboard publishes only aggregate
accuracy per agent and model, so a per-task `claude_code_score` would have to be
invented. Comparisons are made at the campaign level against the aggregate, and
the profile records which reference run it is being compared to.
"""
from __future__ import annotations

from pathlib import Path

from .api import TaskSpec
from .naturebench_cli import executable_path, _overlap
from .retained_results import read_json, require
from .terminalbench_tasks import BENCHMARK, METRIC, SOLUTION_NAME, verify_assets

ASSETS_SCHEMA = "evolvcoder-terminalbench-assets-v1"
REQUIRED_ASSET_FIELDS = {"schema_version", "profile", "tasks_root", "upstream_root",
                         "harbor_python", "trial_timeout_seconds", "terminal_timeout_seconds"}


def manifest(config) -> dict:
    require(config.benchmark_config is not None, "Terminal-Bench requires benchmark_config")
    raw = read_json(config.benchmark_config)
    require(set(raw) == REQUIRED_ASSET_FIELDS, "invalid Terminal-Bench assets manifest fields")
    require(raw["schema_version"] == ASSETS_SCHEMA, "unknown Terminal-Bench assets schema")
    for name in ("trial_timeout_seconds", "terminal_timeout_seconds"):
        require(type(raw[name]) is int and raw[name] > 0, f"invalid {name}")
    resolved = {**raw, "model_cache": None}
    for field in ("profile", "tasks_root", "upstream_root", "harbor_python"):
        require(isinstance(raw[field], str) and raw[field], "invalid asset path")
        resolved[field] = (config.benchmark_config.parent / raw[field]).resolve()
    require(resolved["harbor_python"].is_file(),
            "harbor_python must point at the interpreter of the installed official harness")
    return resolved


class TerminalBenchAdapter:
    def task(self, task_id: str, config) -> TaskSpec:
        require(task_id == config.task, "no qualified task adapter for this Terminal-Bench task")
        assets = manifest(config)
        root = (assets["tasks_root"] / task_id).resolve()
        require(root.is_dir() and root.parent == assets["tasks_root"],
                "task package must be one directory directly under tasks_root")
        custody = verify_assets(root / "custody.json")
        require(custody["task_id"] == task_id and custody["metric"] == METRIC,
                "custody does not bind this task and metric")
        require(custody["search_signal"] == "candidate_self_validation",
                "this package was sealed for diagnostics; its results are not leaderboard-comparable "
                "and it must not be run as a benchmark cell")

        metadata = read_json(root / "metadata.json")
        require(metadata["search_metric"] == METRIC
                and metadata["hidden_suite_visible_to_candidate"] is False
                and metadata["leaderboard_comparable"] is True,
                "task metadata must declare the public metric and keep the hidden suite private")

        profile = read_json(assets["profile"])
        entry = profile["tasks"][task_id]
        require(entry.get("search_metric") == METRIC and "claude_code_score" not in entry,
                "Terminal-Bench profile must use the explicit self-validation metric, "
                "not a per-task Claude Code threshold")
        require((assets["profile"].parent / entry["custody"]).resolve() == (root / "custody.json"),
                "profile entry does not point at this task's sealed custody")

        # The learning stack keys off a `run.py` artifact and a scalar official
        # metric; neither holds here, so reuse stays off until it is qualified.
        require(not config.learning.rsi_enabled and not config.learning.wisdom_enabled
                and config.winner_registry is None,
                "Terminal-Bench learning/reuse is not yet qualified; use explicit search "
                "without RSI/Wisdom")

        starter = (assets["profile"].parent / entry["starter"]).resolve()
        require((starter / SOLUTION_NAME).is_file(),
                f"starter directory must contain {SOLUTION_NAME}")
        public = (root / "problem").resolve()
        protected = [root / "evaluation", root / "custody.json", assets["upstream_root"]]
        mounts = [public, starter, executable_path(config.runtime.python).parent.parent,
                  config.output_root, Path("/usr"), Path("/etc"), Path("/bin"), Path("/lib"), Path("/lib64")]
        require(not any(_overlap(mount, item) for mount in mounts for item in protected),
                "public/runtime/output mount overlaps the verifier reference or the upstream task tree")

        return TaskSpec(BENCHMARK, task_id, (root / "problem/README.md").read_text(),
                        starter, public, "terminalbench:" + custody["custody_sha256"],
                        METRIC["name"], METRIC["direction"], root / "evaluation/task_ref.json")
