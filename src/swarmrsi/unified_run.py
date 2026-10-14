"""Config-driven full NatureBench execution, shared by the public CLI and SDK.

No alternate Coder, evaluator or search loop lives here. Every trial executes
InstalledValidationRunner. Local hardware fields are admission requirements,
not isolation quotas; per-action/full-run budgets remain in the existing loop.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import threading

from .api import RunContext, RunResult, TaskSpec
from .configuration import RunConfig, load_config
from .contracts import canonical_sha256
from . import naturebench_session as session
from .installed_learning import InstalledValidationRunner, portable_profile
from .learning_cli import _LazyPolicyProposer, proposer
from .learning_evidence import (FeedbackPool, retain_learning_winners, source_ref,
                               verify_ref, verify_run_evidence, learning_winner_skips)
from .model_settings import profile_with_model_controls
from .naturebench_cli import executable_path, isolated_environment, run_lock, _overlap, validate_credential_env
from .retained_results import read_json, require, select_pending_tasks, verify_seal, write_immutable
from .rsi import RSIChain, ValidationPending
from .session_limits import profile_with_limits
from .tree_policy import TreePolicy, PolicyTreeStrategy
from .search_policy import SearchPolicy, matched_profile, policy_from_record, search_spec_of, validate_execution_identity
from .validation_failure import ValidationUnavailable, verify_unavailable
from .wisdom import WisdomLibrary, verify_snapshot


_SESSION_LOCK = threading.Lock()
_GLOBALS = ("TASKS_ROOT", "ACTIVE_TASK_ID", "RUNTIME", "SECRET", "SECRET_ENV",
            "_CONTROLLER_ENVIRONMENT", "MODEL_CACHE", "BASE_URL", "CANDIDATE_RUNTIME")


@contextmanager
def session_scope(cache):
    # The legacy full loop uses globals and os.environ. Reject overlapping SDK
    # calls, including reentrant callbacks, rather than cross-contaminating them.
    if not _SESSION_LOCK.acquire(blocking=False):
        raise ValidationPending("another SDK session owns process-global runtime state")
    previous = {name: getattr(session, name) for name in _GLOBALS}
    try:
        with isolated_environment(cache):
            yield
    finally:
        for name, value in previous.items():
            setattr(session, name, value)
        _SESSION_LOCK.release()


def result_dict(result: RunResult) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(result).items()}


def context_for(config: RunConfig | Path, run_id: str | None = None) -> RunContext:
    if isinstance(config, Path):
        config = load_config(config)
    require(isinstance(config, RunConfig), "expected RunConfig or configuration Path")
    run_id = run_id or canonical_sha256(config.to_dict())[:24]
    require(run_id not in {".", ".."} and Path(run_id).name == run_id and bool(run_id), "run-id must be one directory name")
    return RunContext(run_id, config.output_root / run_id, config)


def manifest_for(config):
    if config.benchmark == "mlebench":
        from .mlebench_run import manifest
        return manifest(config)
    if config.benchmark == "openproblems":
        from .openproblems_run import manifest as openproblems_manifest
        return openproblems_manifest(config)
    require(config.benchmark == "naturebench", "NatureBench adapter cannot handle another benchmark")
    path = config.benchmark_config
    require(path is not None, "NatureBench requires benchmark_config")
    data = read_json(path)
    allowed = {"schema_version", "profile", "tasks_root", "model_cache", "trial_timeout_seconds"}
    require(set(data).issubset(allowed) and {"profile", "tasks_root", "trial_timeout_seconds"}.issubset(data),
            "NatureBench manifest requires profile, tasks_root and trial_timeout_seconds; unknown fields are errors")
    require(data.get("schema_version") == "swarmrsi-naturebench-assets-v1", "unknown NatureBench manifest schema")
    require(type(data["trial_timeout_seconds"]) is int and data["trial_timeout_seconds"] > 0,
            "trial_timeout_seconds must be a positive process watchdog")
    resolved = dict(data)
    for name in ("profile", "tasks_root", "model_cache"):
        value = data.get(name)
        if value is not None:
            require(isinstance(value, str) and bool(value), "manifest asset must be a path")
            asset = Path(value).expanduser()
            resolved[name] = (path.parent / asset).resolve() if not asset.is_absolute() else asset.resolve()
        else:
            resolved[name] = None
    return resolved


class NatureBenchAdapter:
    def task(self, task_id, config):
        require(task_id == config.task and task_id not in {".", ".."} and Path(task_id).name == task_id,
                "NatureBench task must be one task directory")
        assets = manifest_for(config)
        profile = portable_profile(read_json(assets["profile"]), assets["profile"])
        task = profile["tasks"][task_id]
        root = assets["tasks_root"] / task_id
        evaluator = root / "evaluation/evaluator.py"
        # The benchmark-native aggregate improvement is higher-is-better even
        # when an underlying task metric is lower-is-better.
        return TaskSpec("naturebench", task_id, (root / "problem/README.md").read_text(),
                        Path(task["starter"]), root / "problem/data",
                        "naturebench:" + canonical_sha256([source_ref(evaluator), source_ref(root / "metadata.json"),
                                                           source_ref(session.EVALUATOR)]),
                        "aggregate_improvement", "maximize", evaluator)


def _imported_policy(config):
    record = read_json(config.learning.policy)
    if record.get("schema_version") == "swarmrsi-frozen-rsi-policy-v1":
        verify_seal(record, "export_sha256")
        record = record["policy"]
    return record


def initial_policy(config):
    """The policy this run starts from, bound to what the algorithm allows.

    Outer learning over a search plugin is only available through parameters
    that plugin declares itself; an algorithm that declares none is refused by
    name and never replaced with another algorithm's policy.
    """
    if config.search is not None:
        if not config.learning.rsi_enabled and config.learning.policy is None:
            return SearchPolicy(config.search)
        from .learning_policies import learnable_policy
        from .search_plugins import default_search_registry
        algorithm = default_search_registry().create(config.search, seed=config.seed)
        if config.learning.policy is None:
            return learnable_policy(config.search, algorithm)
        policy = learnable_policy(config.search, algorithm, None).__class__.from_dict(_imported_policy(config))
        require(policy.search.to_dict() == config.search.to_dict(),
                "imported search policy belongs to a different configured search")
        policy.verify_declaration(algorithm)
        return policy
    if config.learning.policy is None:
        return TreePolicy()
    return TreePolicy.from_dict(_imported_policy(config))


def compiled_profile(config, assets):
    profile = portable_profile(read_json(assets["profile"]), assets["profile"])
    require(config.task in profile["tasks"], "task is absent from NatureBench profile")
    profile = profile_with_model_controls(profile, config.model)
    profile = profile_with_limits(profile, config.task, config.limits)
    if config.benchmark == 'mlebench':
        from dataclasses import replace
        reserve = assets['terminal_timeout_seconds']
        require(config.limits.evaluations is None or config.limits.evaluations >= 2,
                'MLE requires an inner evaluation and one reserved terminal evaluation')
        require(config.limits.wall_seconds is None or config.limits.wall_seconds > reserve,
                'total wall limit must exceed reserved terminal time')
        inner = replace(config.limits,
                        evaluations=None if config.limits.evaluations is None else config.limits.evaluations - 1,
                        wall_seconds=None if config.limits.wall_seconds is None else config.limits.wall_seconds - reserve)
        profile = profile_with_limits(profile, config.task, inner)
        require(config.limits.wall_seconds is None or assets['trial_timeout_seconds'] <= inner.wall_seconds,
                'inner process watchdog exceeds inner wall allocation')
    from .session_runtime import runtime_clock_controls
    # Keep the declared resource envelope unchanged; only stop active work early
    # enough to serialize/export the verified result before the process watchdog.
    profile["session_runtime_controls"] = runtime_clock_controls(
        profile["tasks"][config.task]["limits"]["wall_time_seconds"], assets["trial_timeout_seconds"])
    profile.pop("rsi_policy", None)
    if config.search is not None:
        profile["search"] = config.search.to_dict()
    else:
        require(profile.get("search") is None, "set search in RunConfig, not only the benchmark profile")
    profile["wisdom"] = {"mode": "off"}
    if config.learning.wisdom_enabled:
        snapshot = read_json(config.learning.wisdom_snapshot)
        verify_snapshot(snapshot)
        profile["wisdom"] = {"mode": "readonly", "snapshot_path": str(config.learning.wisdom_snapshot)}
    elif config.learning.wisdom_snapshot is not None:
        raise ValueError("wisdom_snapshot supplied while Wisdom is disabled")
    return profile


def check_paths(context, task, assets):
    config, output = context.config, context.output_dir.resolve()
    require(context.output_dir == output, "output_dir must be resolved")
    if config.runtime.work_root is not None:
        require(output.is_relative_to(config.runtime.work_root), "output_dir must be within runtime.work_root")
    protected = [task.starter, assets["tasks_root"], assets["profile"], config.benchmark_config,
                 session.ROOT, executable_path(config.runtime.python).parent.parent]
    protected += [p for p in (assets["model_cache"], config.model.api_key_file,
                              config.learning.policy, config.learning.wisdom_snapshot, config.winner_registry) if p]
    require(not any(_overlap(output, path) for path in protected), "output overlaps input/private/runtime assets")
    if config.learning.feedback_store is not None:
        pool = config.learning.feedback_store
        require(not _overlap(output, pool) and not any(_overlap(pool, path) for path in protected),
                "persistent feedback_store must be separate from run outputs and inputs")


def local_admission(config):
    """Inspect effective host availability before provider use; do not reserve it.

    CUDA_VISIBLE_DEVICES and library thread limits are passed to every inner
    controller. Shared-host RAM/CPU availability is not a cgroup hard cap.
    """
    runtime = config.runtime
    require(runtime.kind in {"local", "container"}, "this adapter runs locally/in an existing allocation; remote submission is not implemented")
    if runtime.kind == "container":
        require(runtime.host is None and runtime.gpus == 0 and runtime.container is not None,
                "container requires explicit candidate options and local CPU execution")
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    require(available_cpus is not None and runtime.cpus <= available_cpus, "requested CPUs exceed effective host affinity")
    memory = None
    if runtime.memory_mb is not None:
        if Path("/proc/meminfo").exists():
            rows = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            memory = int(rows["MemAvailable"].split()[0]) * 1024
            # cgroup v2 memory can be lower than host MemAvailable.
            group = next((line.split("::", 1)[1] for line in Path("/proc/self/cgroup").read_text().splitlines()
                          if line.startswith("0::")), None)
            if group is not None:
                base = Path("/sys/fs/cgroup") / group.lstrip("/")
                for limit, current in ((base / "memory.max", base / "memory.current"),):
                    if limit.is_file() and current.is_file() and limit.read_text().strip() != "max":
                        memory = min(memory, max(0, int(limit.read_text()) - int(current.read_text())))
        require(memory is not None, "memory admission is currently available on Linux only")
        require(runtime.memory_mb * 1024**2 <= memory, "requested memory exceeds observed available memory")
    environment = {name: str(runtime.cpus) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    environment["CUDA_VISIBLE_DEVICES"] = ""
    devices = 0
    if runtime.gpus:
        probe = subprocess.run([str(executable_path(runtime.python)), "-c",
                                "import torch,json;print(json.dumps({'count':torch.cuda.device_count()}))"],
                               env=dict(os.environ), capture_output=True, text=True, timeout=60, check=False)
        require(probe.returncode == 0, "GPU admission probe failed in configured scientific Python")
        devices = json.loads(probe.stdout)["count"]
        require(type(devices) is int and devices >= runtime.gpus, "insufficient visible CUDA GPUs")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        choices = visible.split(",") if visible else [str(i) for i in range(devices)]
        require(len(choices) >= runtime.gpus, "CUDA binding has fewer entries than requested GPUs")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(choices[:runtime.gpus])
    return environment, {"cpu_affinity_available": available_cpus, "memory_bytes_available": memory,
                         "visible_gpu_count": devices, "semantics": "availability_check_and_library_thread_GPU_visibility_controls_not_hard_CPU_RAM_quotas"}


class SwarmRSIAdapter:
    def _inputs(self, task, context):
        config = context.config
        if config.benchmark == "mlebench":
            from .mlebench_run import MLEBenchAdapter
            expected_task = MLEBenchAdapter().task(config.task, config)
        elif config.benchmark == "openproblems":
            from .openproblems_run import OpenProblemsAdapter
            expected_task = OpenProblemsAdapter().task(config.task, config)
        else:
            expected_task = NatureBenchAdapter().task(config.task, config)
        require(task == expected_task, "TaskSpec differs from official manifest")
        require(config.runtime.kind in {"local", "container"}, "remote/Slurm submission is not implemented by this adapter")
        if config.model.api_key_env is not None:
            validate_credential_env(config.model.api_key_env)
        assets = manifest_for(config)
        check_paths(context, task, assets)
        profile, policy = compiled_profile(config, assets), initial_policy(config)
        from .candidate_execution import for_context
        candidate_runtime = for_context(context)
        require(profile.get("candidate_runtime") is None, "configure candidate runtime through RunConfig, not benchmark profile")
        if candidate_runtime is not None:
            require(assets["model_cache"] is None, "container model-cache mount is not implemented")
            profile["candidate_runtime"] = candidate_runtime
        search_identity = None
        executed_search = search_spec_of(policy)
        if executed_search is not None:
            original_inputs = context.output_dir / "inputs.json"
            if original_inputs.is_file():
                search_identity = read_json(original_inputs).get("search_identity")
                sources = validate_execution_identity(search_identity, policy, config.seed)
                require(all(source_ref(Path(p))["sha256"] == digest for p, digest in sources.items()),
                        "original search implementation source changed")
            else:
                from .search_plugins import default_search_registry
                search = default_search_registry().create(executed_search, seed=config.seed)
                search_identity = search.execution_identity
        else:
            PolicyTreeStrategy(policy=policy, base_config=profile.get("tree_search"))
        refs = [source_ref(config.benchmark_config), source_ref(assets["profile"])]
        if config.benchmark == "mlebench":
            refs += [source_ref(assets['custody'])]
            refs += read_json(assets['custody'])['refs']
        refs += [source_ref(p) for p in (config.learning.policy, config.learning.wisdom_snapshot, config.winner_registry) if p]
        refs += [source_ref(p) for p in sorted(session.PACKAGE_ROOT.rglob("*.py"))]
        if search_identity is not None:
            existing = {ref["path"] for ref in refs}
            refs += [source_ref(Path(p)) for p in search_identity["source_hashes"] if p not in existing]
        identity = {"schema_version": "swarmrsi-unified-run-v1", "run_id": context.run_id,
                    "output_dir": str(context.output_dir), "config": config.to_dict(), "input_refs": refs,
                    "evaluator_id": task.evaluator_id, "theta0": policy.to_dict(), "profile_sha256": canonical_sha256(profile)}
        if search_identity is not None:
            identity["search_identity"] = search_identity
        return assets, profile, policy, identity

    def _verify_existing(self, context, profile, identity):
        path = context.output_dir / "inputs.json"
        if not path.exists():
            require(not context.output_dir.exists() or not any(p.name != ".session.lock" for p in context.output_dir.iterdir()),
                    "run has orphaned or foreign artifacts; refusing takeover")
            return False
        require(read_json(path) == identity, "run identity/configuration/source changed")
        require(read_json(context.output_dir / "profile.json") == profile, "compiled profile changed")
        chain_root = context.output_dir / "rsi"
        require(not chain_root.exists() or (chain_root / "chain.json").is_file()
                or not any(chain_root.iterdir()), "orphaned RSI journals; original chain configuration is missing")
        for ref in identity["input_refs"]:
            verify_ref(ref)
        return True

    def _retained(self, task, context, profile):
        config = context.config
        if not config.skip_retained_winners or config.winner_registry is None:
            return None
        registry = read_json(config.winner_registry)
        if registry.get("schema_version") == "swarmrsi-retained-winners-v1":
            queue = select_pending_tasks(profile, registry)
            selected = next(row for row in queue["tasks"] if row["task_id"] == task.task_id)
            if selected["historical_score"] is None:
                return None
            row = next(r for r in registry["winners"] if r["final_sha256"] == selected["historical_final_sha256"])
            node = next(n for n in row["nodes"] if n["id"] == row["winner_node_id"])
            return RunResult(context.run_id, task.task_id, config.method, "retained", Path(node["artifact"]),
                             row["winner_score"], task.evaluator_id, config.winner_registry,
                             "verified historical winner; no current-version measurement", details={"new_version_score": None})
        skipped = learning_winner_skips([config.winner_registry], profile)
        if task.task_id not in skipped:
            return None
        files = [verify_ref(ref) for ref in registry["winner_refs"]] if "winner_refs" in registry else [config.winner_registry]
        row = next(read_json(p) for p in files if read_json(p)["winner_sha256"] == skipped[task.task_id]["winner_sha256"])
        node = next(n for n in row["run"]["programs"] if n["id"] == row["winner_node_id"])
        return RunResult(context.run_id, task.task_id, config.method, "retained", Path(node["artifact"]["path"]).parent,
                         row["winner_metric"], task.evaluator_id, config.winner_registry,
                         "verified historical learning winner; no current-version measurement", details={"new_version_score": None})

    def _runner(self, context, assets, profile, *, execute):
        config, root = context.config, context.output_dir / "validation"
        if (root / "runner.json").exists():
            runner = InstalledValidationRunner.resume(root, allow_execution=execute)
            require(runner.profile == profile and runner.config["cells"] == [{"task_id": config.task, "seed": config.seed}],
                    "stored validation runner differs from requested run")
            expected = {"runtime_python": str(executable_path(config.runtime.python)),
                        "tasks_root": str(assets["tasks_root"]),
                        "credential_file": str(config.model.api_key_file) if config.model.api_key_file else None,
                        "model_cache": str(assets["model_cache"]) if assets["model_cache"] else None,
                        "base_url": config.model.base_url}
            if config.model.api_key_env is not None:
                expected["credential_env"] = config.model.api_key_env
            require(runner.config["execution"] == expected and runner.config["timeout_seconds"] == assets["trial_timeout_seconds"],
                    "stored execution differs from requested runtime/model")
            return runner
        require(not root.exists() or not any(root.iterdir()), "orphaned validation artifacts; original runner configuration is missing")
        environment, observed = local_admission(config)
        write_immutable(context.output_dir / "resources.json", observed)
        return InstalledValidationRunner(root, profile_path=context.output_dir / "profile.json",
            runtime_python=executable_path(config.runtime.python), tasks_root=assets["tasks_root"],
            cells=[{"task_id": config.task, "seed": config.seed}], credential_file=config.model.api_key_file,
            credential_env=config.model.api_key_env, model_cache=assets["model_cache"], base_url=config.model.base_url,
            environment=environment, timeout_seconds=assets["trial_timeout_seconds"], allow_execution=execute)

    def _chain(self, context, profile, policy, *, read_only=False):
        config = context.config
        return RSIChain(context.output_dir / "rsi", theta0=policy, cells=[{"task_id": config.task, "seed": config.seed}],
                        rounds=config.learning.rsi_rounds, proposal_seed=config.seed,
                        targets={config.task: profile["tasks"][config.task]["claude_code_score"]}, stop_solved_tasks=True,
                        base_tree_config=profile.get("tree_search") if isinstance(policy, TreePolicy) else None,
                        read_only=read_only)

    def _result(self, task, context, record, *, details=None):
        verify_run_evidence(record)
        require((record["task_id"], record["seed"], record["model"]) ==
                (task.task_id, context.config.seed, context.config.model.model), "run evidence identity differs")
        final = read_json(Path(record["run_dir"]) / "final.json")
        node = next((n for n in record["programs"] if n["id"] == final["best_node_id"]), None)
        valid = record["artifact_valid"] and node is not None
        if task.benchmark != 'naturebench':
            details = {**(details or {}), 'metric_name': task.metric_name, 'metric_direction': task.metric_direction,
                       'metric_role': 'public_validation', 'search_utility': record['utility'],
                       'official_terminal_score': None}
        return RunResult(context.run_id, task.task_id, context.config.method,
                         "completed" if valid else "evaluation_invalid",
                         Path(node["artifact"]["path"]).parent if valid else None,
                         record["utility"] if valid else None, task.evaluator_id, Path(record["run_dir"]) / "final.json",
                         record["terminal_status"], record["accounting"],
                         {**(details or {}), "best_evolved_native_metric": record["best_evolved_native_metric"],
                          "best_evolved_node_id": record["best_evolved_node_id"], "best_node_id": final["best_node_id"]})

    def _trial_origin(self, context, runner, policy, root):
        identity = {"runner_sha256": canonical_sha256(runner.config), "policy": policy.to_dict(),
                    "task_id": context.config.task, "seed": context.config.seed}
        expected = runner.root / "trials" / canonical_sha256(identity)
        require(root == expected and root.resolve() == expected,
                "delivery trial does not belong to this run/runner/policy")
        require(read_json(root / "identity.json") == identity, "delivery original trial identity changed")

    def _delivery_refs(self, context, *, has_chain):
        names = ["inputs.json", "profile.json", "resources.json", "wisdom_next.json", "winners.json",
                 "validation/runner.json", "validation/runner_identity.json"]
        if has_chain:
            names.append("rsi/chain.json")
        if (context.output_dir / "learned_policy.json").exists():
            names.append("learned_policy.json")
        return [source_ref(context.output_dir / name) for name in names]

    def _terminal(self, task, context):
        path = context.output_dir / "result.json"
        if not path.exists():
            return None
        delivery = read_json(path)
        verify_seal(delivery, "delivery_sha256")
        for ref in delivery["refs"]:
            verify_ref(ref)
        profile, policy = read_json(context.output_dir / "profile.json"), initial_policy(context.config)
        require((context.output_dir / "validation/runner.json").is_file(), "terminal runner is missing")
        runner = self._runner(context, manifest_for(context.config), profile, execute=False)
        state = None
        has_chain = delivery.get("rsi_state") is not None
        require(delivery["refs"] == self._delivery_refs(context, has_chain=has_chain), "terminal required reference set changed")
        if has_chain:
            require(context.config.learning.rsi_enabled, "unexpected RSI state in non-RSI run")
            state = self._chain(context, profile, policy, read_only=True).state()
            require(state == delivery["rsi_state"], "terminal RSI state differs from original paired evidence")
            policy = TreePolicy.from_dict(state["current_policy"])
        elif context.config.learning.rsi_enabled and delivery["record"] is not None:
            record = delivery["record"]
            verify_run_evidence(record)
            require(record["best_evolved_native_metric"] is not None
                    and record["best_evolved_native_metric"] > profile["tasks"][context.config.task]["claude_code_score"]
                    and not (context.output_dir / "rsi").exists(),
                    "RSI completion requires its original chain unless the initial new program solved the task")
        if delivery["record"] is not None:
            record = delivery["record"]
            self._trial_origin(context, runner, policy, Path(record["run_dir"]).parent)
            require(record["policy"] == policy.to_dict() and record["source_role"] == "selection"
                    and record["comparison_context"]["fixed_profile"] == matched_profile(profile, policy)
                    and read_json(Path(record["run_dir"]).parent / "evidence.json") == record,
                    "terminal result differs from selected policy and original trial evidence")
            details = {"validation_accounting": runner.accounting()}
            if context.config.learning.rsi_enabled:
                if has_chain:
                    require(state["finished"] and not state.get("halted"), "RSI is not finished")
                    details["rsi"] = state
                    exported = read_json(context.output_dir / "learned_policy.json")
                    verify_seal(exported, "export_sha256")
                    require(exported["policy"] == policy.to_dict() and exported["state_sha256"] == state["state_sha256"],
                            "learned policy differs from measured current incumbent")
                else:
                    details["rsi"] = {"status": "initial_policy_already_solved", "proposals": 0, "effect_not_measured": True}
            result = self._result(task, context, record, details=details)
            require(result_dict(result) == delivery["result"], "stored result differs from bound evaluator evidence")
            return result
        failure = delivery["failure"]
        identity = verify_unavailable(failure)
        failed_policy = policy_from_record(identity["policy"])
        self._trial_origin(context, runner, failed_policy, Path(failure["trial_root"]))
        require(Path(failure["runner_ref"]["path"]) == runner.root / "runner.json", "failure belongs to another runner")
        if has_chain:
            require(state.get("halted") and state["validation_failure"] == failure, "failure is not the required halted RSI member")
        else:
            require(failed_policy == policy and not (context.output_dir / "rsi").exists(), "failure is not the initial policy trial")
        result = RunResult(context.run_id, task.task_id, context.config.method, "failed", evidence=path,
                           reason="required validation unavailable; original attempt retained", details={"validation_failure": failure})
        require(result_dict(result) == delivery["result"], "stored failure result identity differs")
        return result

    def status(self, task, context):
        if task.benchmark == 'mlebench':
            from .mlebench_run import workflow_status
            return workflow_status(self, task, context)
        return self._search_status(task, context)

    def _search_status(self, task, context):
        assets, profile, policy, identity = self._inputs(task, context)
        if not self._verify_existing(context, profile, identity):
            retained = self._retained(task, context, profile)
            return {"status": "retained" if retained else "not_started", "result": result_dict(retained) if retained else None}
        terminal = self._terminal(task, context)
        if terminal:
            return {"status": terminal.status, "result": result_dict(terminal)}
        state = self._chain(context, profile, policy, read_only=True).state() if (context.output_dir / "rsi/chain.json").exists() else None
        root = context.output_dir / "validation"
        accounting = InstalledValidationRunner.resume(root).accounting() if (root / "runner.json").exists() else None
        return {"status": "halted" if state and state.get("halted") else "unfinished", "rsi": state, "accounting": accounting}

    def preflight(self, task, context):
        if task.benchmark == 'mlebench':
            from .mlebench_run import workflow_preflight
            return workflow_preflight(self, task, context)
        return self._operate(task, context, execute=False)

    def run(self, task, context):
        if task.benchmark == 'mlebench':
            from .mlebench_run import workflow_run
            return workflow_run(self, task, context)
        return self._operate(task, context, execute=True)

    def _operate(self, task, context, *, execute):
        assets, profile, policy, identity = self._inputs(task, context)
        with session_scope(assets["model_cache"]), run_lock(context.output_dir):
            exists = self._verify_existing(context, profile, identity)
            if exists:
                terminal = self._terminal(task, context)
                if terminal:
                    return terminal if execute else {"ready": True, "reused_terminal": True, "result": result_dict(terminal)}
            else:
                retained = self._retained(task, context, profile)
                if retained:
                    return retained if execute else {"ready": True, "retained": result_dict(retained)}
                write_immutable(context.output_dir / "profile.json", profile)
                write_immutable(context.output_dir / "inputs.json", identity)
            runner = self._runner(context, assets, profile, execute=execute)
            cell = {"task_id": task.task_id, "seed": context.config.seed}
            if not execute:
                return runner.preflight(policy=policy, **cell)
            config = context.config
            pool = FeedbackPool(config.learning.feedback_store or context.output_dir / "feedback")
            details, record, failure = {}, None, None
            chain_exists = (context.output_dir / "rsi/chain.json").exists()
            try:
                if not config.learning.rsi_enabled or not chain_exists:
                    record = runner.evaluate(policy=policy, **cell)
                    pool.add(record)
                from .task_metrics import target
                threshold = target(profile['tasks'][task.task_id])
                solved = threshold is not None and record is not None and record["best_evolved_native_metric"] is not None and record["best_evolved_native_metric"] > threshold
                if config.learning.rsi_enabled and not solved:
                    chain = self._chain(context, profile, policy, read_only=chain_exists)
                    state = chain.state()
                    if not state["finished"] and not state.get("halted"):
                        # Required infrastructure checks precede the first new
                        # outer request. Already recorded proposals need no key.
                        if not (chain.root / f"round_{state['completed_rounds']:03d}/proposal_response.json").exists():
                            require(runner.preflight(policy=chain.codec.from_dict(state["current_policy"]), **cell)["ready"], "RSI preflight failed")
                        outer_record = {"requested_config": {}, "installed_execution": runner.config["execution"]}
                        state = chain.run(_LazyPolicyProposer(lambda: proposer(outer_record, runner)), runner, pool)
                    details["rsi"] = state
                    if state.get("halted"):
                        raise ValidationUnavailable(state["validation_failure"])
                    if not state["finished"]:
                        return RunResult(context.run_id, task.task_id, config.method, "pending", reason="RSI validation pending", details=details)
                    chain.export_policy(context.output_dir / "learned_policy.json")
                    record = runner.evaluate(policy=chain.codec.from_dict(state["current_policy"]), **cell)
                elif config.learning.rsi_enabled:
                    details["rsi"] = {"status": "initial_policy_already_solved", "proposals": 0, "effect_not_measured": True}
            except ValidationPending as exc:
                return RunResult(context.run_id, task.task_id, config.method, "pending", reason=str(exc), details=details)
            except ValidationUnavailable as exc:
                failure = exc.evidence
                verify_unavailable(failure)
            # Retain winners from this run only. The next Wisdom snapshot also
            # preserves the previously supplied source library and verified
            # feedback accumulated before this snapshot was taken.
            runs = [read_json(p) for p in sorted((runner.root / "trials").glob("*/evidence.json"))]
            library = WisdomLibrary(context.output_dir / "wisdom_next")
            for run in runs:
                pool.add(run)
            sources = list(pool.snapshot()["records"])
            if config.learning.wisdom_enabled:
                sources += list(read_json(config.learning.wisdom_snapshot)["sources"].values())
            for run in sources:
                library.ingest(run)
            library.freeze(context.output_dir / "wisdom_next.json")
            winners = retain_learning_winners(runs, profile, context.output_dir / "winners")
            body = {"schema_version": "swarmrsi-retained-learning-collection-v1", "winner_refs": [
                source_ref(context.output_dir / "winners" / (w["run"]["run_id"] + ".json")) for w in winners]}
            write_immutable(context.output_dir / "winners.json", {**body, "collection_sha256": canonical_sha256(body)})
            if failure is None:
                require(record is not None, "completed workflow has no verified run evidence")
                details["validation_accounting"] = runner.accounting()
                result = self._result(task, context, record, details=details)
            else:
                record = None
                result = RunResult(context.run_id, task.task_id, config.method, "failed", evidence=context.output_dir / "result.json",
                                   reason="required validation unavailable; original attempt retained", details={"validation_failure": failure})
            has_chain = (context.output_dir / "rsi/chain.json").exists()
            rsi_state = self._chain(context, profile, policy, read_only=True).state() if has_chain else None
            refs = self._delivery_refs(context, has_chain=has_chain)
            delivery = {"result": result_dict(result), "record": record, "failure": failure, "refs": refs, "rsi_state": rsi_state}
            write_immutable(context.output_dir / "result.json", {**delivery, "delivery_sha256": canonical_sha256(delivery)})
            # First delivery and replay use one verifier, not two definitions
            # of completion. Persisted results are not authority by themselves.
            return self._terminal(task, context)
