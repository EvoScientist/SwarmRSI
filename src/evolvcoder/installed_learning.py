"""Full RSI validation against the installed NatureBench session.

The orchestrator is this package's Python, while scientific code uses the
explicit candidate Python. Neither a developer checkout nor ambient EVOLVCODER
paths are needed. Algorithm, comparison and evidence semantics stay shared.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

from . import naturebench_session as session
from .learning_evidence import source_ref, verify_ref
from .learning_trials import SubprocessValidationRunner
from .naturebench_cli import OPERATIONAL_ENVIRONMENT_KEYS, configure, executable_path, _overlap, validate_credential_env
from .retained_results import read_json, require, write_immutable
from .tree_policy import PolicyTreeStrategy, TreePolicy


def portable_profile(profile: dict, profile_path: Path) -> dict:
    profile = deepcopy(profile)
    for task in profile["tasks"].values():
        starter = Path(task["starter"]).expanduser()
        task["starter"] = str((starter if starter.is_absolute() else profile_path.parent / starter).resolve())
    return profile


class InstalledValidationRunner(SubprocessValidationRunner):
    SCHEMA = "evolvcoder-installed-rsi-validation-runner-v1"

    def __init__(self, root: Path, *, profile_path: Path, runtime_python: Path,
                 tasks_root: Path, cells: list[dict], credential_file: Path | None = None,
                 credential_env: str | None = None,
                 model_cache: Path | None = None, base_url: str = "https://openrouter.ai/api/v1",
                 environment: dict[str, str] | None = None, allow_execution: bool = False,
                 timeout_seconds: int = 14400):
        require(type(timeout_seconds) is int and timeout_seconds > 0, "validation timeout must be a positive integer")
        require((credential_file is None) != (credential_env is None), "configure exactly one credential file or environment name")
        if credential_env is not None:
            validate_credential_env(credential_env)
        require(bool(cells) and all(set(c) == {"task_id", "seed"} and type(c["seed"]) is int
                                   and c["seed"] >= 0 for c in cells), "validation cells must be explicit")
        require(len({(c["task_id"], c["seed"]) for c in cells}) == len(cells), "duplicate validation cells")
        self.root, self.source_root = root.resolve(), session.ROOT
        require(model_cache is None or not _overlap(self.root.parent, model_cache),
                "model-cache must not overlap campaign feedback or validation outputs")
        self.runtime = executable_path(sys.executable)
        self.profile = portable_profile(read_json(profile_path), profile_path.resolve())
        self.profile.pop("rsi_policy", None)
        # Profile generation must happen upstream, not be silently rewritten in
        # just one arm or after the profile source reference has been recorded.
        original = read_json(profile_path); original.pop("rsi_policy", None)
        require(self.profile == original, "installed validation profile must resolve starters before generation")
        require(all(c["task_id"] in self.profile["tasks"] for c in cells), "validation task missing from base profile")
        if self.profile.get("search") is None:
            PolicyTreeStrategy(policy=TreePolicy(), base_config=self.profile.get("tree_search"))
        self.entrypoint = session.PACKAGE_ROOT / "naturebench_cli.py"
        execution = {"runtime_python": str(executable_path(runtime_python)),
                     "tasks_root": str(tasks_root.resolve()), "credential_file": str(credential_file.resolve()) if credential_file is not None else None,
                     "model_cache": str(model_cache.resolve()) if model_cache is not None else None,
                     "base_url": base_url}
        if credential_env is not None:
            execution["credential_env"] = credential_env
        overrides = dict(environment or {})
        require(all(isinstance(k, str) and (k in OPERATIONAL_ENVIRONMENT_KEYS or k.startswith("SLURM_"))
                    and isinstance(v, str) for k, v in overrides.items()), "unsupported operational environment override")
        self.environment = {k: v for k, v in os.environ.items()
                            if k in OPERATIONAL_ENVIRONMENT_KEYS or k.startswith("SLURM_")}
        self.environment.update(overrides)
        sources = {}
        contexts = {}
        # Only read/validate here. configure performs the same private mount
        # and resolved-path checks as the actual installed inner entry point.
        for cell in cells:
            args = argparse.Namespace(profile=profile_path, task_id=cell["task_id"], seed=cell["seed"],
                tasks_root=tasks_root, credential_file=credential_file, credential_env=credential_env, runtime_python=runtime_python,
                run_dir=self.root / "trials" / "input-check" / "run", model_cache=model_cache,
                base_url=base_url, version="v1", preflight=True)
            _, context = configure(args)
            contexts[cell["task_id"]] = {k: context[k] for k in ("runtime_python", "runtime_executable_sha256",
                                                               "tasks_root", "base_url", "model_cache")}
            for path in context["source_hashes"]:
                p = Path(path)
                p = p if p.is_absolute() else self.source_root / p
                if p.resolve() != profile_path.resolve():
                    sources[str(p)] = source_ref(p)
        self.config = {"schema_version": self.SCHEMA, "base_profile": self.profile,
                       "base_profile_source": source_ref(profile_path), "source_root": str(self.source_root),
                       "source_refs": [sources[p] for p in sorted(sources)], "cells": cells,
                       "runtime_python": str(self.runtime), "runtime_executable_resolved": str(self.runtime.resolve()),
                       "orchestrator_executable_ref": source_ref(self.runtime),
                       "candidate_executable_ref": source_ref(Path(execution["runtime_python"])),
                       "execution": execution, "task_contexts": contexts, "environment": self.environment,
                       "execution_control_root": str(self.root.parent),
                       "timeout_seconds": timeout_seconds}
        self.write_config(self.root, self.config)
        self.allow_execution = allow_execution

    @staticmethod
    def write_config(root: Path, config: dict) -> None:
        """Record the original captured context separately from mutable observers."""
        path = root / "runner.json"
        if path.exists():
            reference = read_json(root / "runner_identity.json")["source"]
            require(verify_ref(reference).resolve() == path.resolve(), "runner identity path changed")
        write_immutable(path, config)
        write_immutable(root / "runner_identity.json", {"source": source_ref(path)})

    def _verify_config(self) -> None:
        reference = read_json(self.root / "runner_identity.json")["source"]
        require(verify_ref(reference).resolve() == (self.root / "runner.json").resolve(), "runner identity path changed")
        require(read_json(self.root / "runner.json") == self.config, "in-memory runner context changed")

    @classmethod
    def resume(cls, root: Path, *, allow_execution: bool = False):
        instance = cls.__new__(cls)
        instance.root = root.resolve()
        instance.config = read_json(instance.root / "runner.json")
        instance._verify_config()
        require(instance.config["schema_version"] == cls.SCHEMA, "unknown installed validation runner")
        instance.source_root = Path(instance.config["source_root"])
        require(instance.source_root == session.ROOT, "installed package location changed; original sources required")
        instance.entrypoint = session.PACKAGE_ROOT / "naturebench_cli.py"
        instance.profile = instance.config["base_profile"]
        instance.runtime = Path(instance.config["runtime_python"])
        instance.environment = instance.config["environment"]
        require(str(instance.runtime.resolve()) == instance.config["runtime_executable_resolved"], "runtime executable changed")
        for ref in [instance.config["base_profile_source"], *instance.config["source_refs"],
                    instance.config["orchestrator_executable_ref"], instance.config["candidate_executable_ref"]]:
            verify_ref(ref)
        instance.allow_execution = allow_execution
        return instance

    def _prepare(self, policy: TreePolicy, task_id: str, seed: int):
        self._verify_config()
        for key in ("orchestrator_executable_ref", "candidate_executable_ref"):
            verify_ref(self.config[key])
        root, _ = super()._prepare(policy, task_id, seed)
        execution = self.config["execution"]
        command = [str(self.runtime), "-m", "evolvcoder.naturebench_cli", "--version", "v1",
                   "--task-id", task_id, "--seed", str(seed), "--profile", str(root / "profile.json"),
                   "--run-dir", str(root / "run")]
        for key in ("runtime_python", "tasks_root", "credential_file", "credential_env", "base_url", "model_cache"):
            if execution.get(key) is not None:
                command.extend(["--" + key.replace("_", "-"), execution[key]])
        return root, command

    def _environment(self, *, include_credentials=True):
        # A complete captured allowlist: values absent when prepared remain
        # absent. In particular an observer cannot alter GPU binding or libs
        # in one arm, nor inject PYTHONPATH or an arbitrary credential variable.
        environment = {**self.environment, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        name = self.config["execution"].get("credential_env")
        if name is not None:
            validate_credential_env(name)
            provider_environment = session._CONTROLLER_ENVIRONMENT if session._CONTROLLER_ENVIRONMENT is not None else os.environ
            # Only the installed controller child receives a key at actual
            # execution. Preflight needs existence, never credential bytes.
            if name in provider_environment:
                environment[name] = provider_environment[name] if include_credentials else "preflight-placeholder-not-a-key"
        return environment

    def _preflight_environment(self):
        return self._environment(include_credentials=False)
