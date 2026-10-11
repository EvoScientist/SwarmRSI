"""OpenProblems benchmark adapter: public search over a controller-held split.

The candidate sees only the public counts and the public README. The controller
holds the validation split used for the search metric, and the official held-out
test counts stay in the task package's private directory, unreachable from any
candidate mount. A public validation score is a search signal, not an official
OpenProblems result.
"""
from __future__ import annotations

from pathlib import Path

from .api import TaskSpec
from .naturebench_cli import executable_path, _overlap
from .openproblems_tasks import METRIC, TASK, verify_assets
from .retained_results import read_json, require

ASSETS_SCHEMA = "evolvcoder-openproblems-assets-v1"


def manifest(config) -> dict:
    require(config.benchmark_config is not None, "OpenProblems requires benchmark_config")
    raw = read_json(config.benchmark_config)
    require(set(raw) == {"schema_version", "profile", "tasks_root", "custody", "trial_timeout_seconds"},
            "invalid OpenProblems assets manifest fields")
    require(raw["schema_version"] == ASSETS_SCHEMA, "unknown OpenProblems assets schema")
    require(type(raw["trial_timeout_seconds"]) is int and raw["trial_timeout_seconds"] > 0,
            "invalid trial_timeout_seconds")
    resolved = {**raw, "model_cache": None}
    for field in ("profile", "tasks_root", "custody"):
        require(isinstance(raw[field], str) and raw[field], "invalid asset path")
        resolved[field] = (config.benchmark_config.parent / raw[field]).resolve()
    return resolved


class OpenProblemsAdapter:
    def task(self, task_id: str, config) -> TaskSpec:
        require(task_id == config.task == TASK, "no qualified task adapter for this OpenProblems task")
        assets = manifest(config)
        custody = verify_assets(assets["custody"])
        require(custody["task_id"] == task_id and custody["metric"] == METRIC,
                "custody does not bind this task and metric")
        root = assets["custody"].parent
        require(assets["tasks_root"] == root or root.is_relative_to(assets["tasks_root"]),
                "custody must live inside the configured tasks_root")
        metadata = read_json(root / "metadata.json")
        require(metadata["search_metric"] == METRIC and metadata["official_test_visible_to_candidate"] is False,
                "task metadata must declare the public metric and keep the official test private")
        profile = read_json(assets["profile"])
        entry = profile["tasks"][task_id]
        require(entry.get("search_metric") == METRIC and "claude_code_score" not in entry,
                "OpenProblems profile must use the explicit public metric, not a Claude/SOTA threshold")
        require(not config.learning.rsi_enabled and not config.learning.wisdom_enabled
                and config.winner_registry is None,
                "OpenProblems learning/reuse is not yet qualified; use explicit search without RSI/Wisdom")
        public = root / "problem/data"
        starter = (assets["profile"].parent / entry["starter"]).resolve()
        protected = [root / "evaluation", root / "private", assets["custody"]]
        mounts = [public, starter, executable_path(config.runtime.python).parent.parent, config.output_root,
                  Path("/usr"), Path("/etc"), Path("/bin"), Path("/lib"), Path("/lib64")]
        require(not any(_overlap(mount, item) for mount in mounts for item in protected),
                "public/runtime/output mount overlaps the held-out split or the official test counts")
        return TaskSpec("openproblems", task_id, (root / "problem/README.md").read_text(),
                        starter, public, "openproblems:" + custody["custody_sha256"],
                        METRIC["name"], METRIC["direction"], root / "evaluation/evaluator.py")
