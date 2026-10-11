"""Durable phases around the pinned, otherwise unchanged AIDE Agent.

The original Agent chooses parents, builds prompts, generates actual Nodes,
reviews the actual ExecutionResult, and appends to its Journal. Wrappers only
persist phase boundaries and bind the external benchmark metric. A resumed
pending phase is resolved before another search_policy invocation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterator, Mapping

from .aide_execution import AideExecutionUnresolved
from .aide_openrouter import apply_official_evaluation
from .aide_state import AideJournalStore, atomic_json, canonical, seal, unseal
from .aide_transport import AideCallLedger, AideTransportUnresolved


class AidePhaseCalls:
    """Replay already accepted responses in a resumed logical phase, in order."""

    def __init__(self, ledger: AideCallLedger,
                 response_factory: Callable[[dict[str, Any]], Any]) -> None:
        self.ledger, self.response_factory = ledger, response_factory
        self._cursor: int | None = None
        self._end: int | None = None

    @property
    def count(self) -> int:
        return len(self.ledger.read()["attempts"])

    @contextmanager
    def phase(self, start: int) -> Iterator[None]:
        if self._cursor is not None:
            raise RuntimeError("nested AIDE provider phases are not supported")
        end = self.count
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start <= end:
            raise ValueError("invalid AIDE phase call cursor")
        self._cursor, self._end = start, end
        try:
            yield
            if self._cursor != end:
                raise AideTransportUnresolved("phase did not consume all previously accepted calls")
        finally:
            self._cursor = self._end = None

    def create(self, **kwargs: Any) -> Any:
        if self._cursor is None or self._end is None:
            raise RuntimeError("AIDE provider call outside a durable phase")
        if self._cursor < self._end:
            response = self.ledger.replay(
                self._cursor, logical_request=kwargs, response_factory=self.response_factory,
            )
            self._cursor += 1
            return response
        response = self.ledger.create(**kwargs)
        # A process-backed transport returns raw JSON. Account and save it
        # before SDK schema validation; replay must use the same factory.
        return self.response_factory(response) if isinstance(response, dict) else response


class AideSession:
    """Single-owner session; after any exception reconstruct from its checkpoint.

This class deliberately does not reimplement draft/debug/improve or parent
selection. ``executor.reconcile`` may reuse complete, verified receipts, but
must not repeat an uncertain candidate or scorer operation.
    """

    def __init__(self, *, agent: Any, store: AideJournalStore, calls: AidePhaseCalls,
                 executor: Any, node_factory: Callable[..., Any],
                 journal_factory: Callable[..., Any], metric_factory: Callable[..., Any],
                 worst_metric_factory: Callable[..., Any],
                 execution_result_factory: Callable[..., Any]) -> None:
        self.agent, self.store, self.calls, self.executor = agent, store, calls, executor
        self.metric_factory, self.worst_metric_factory = metric_factory, worst_metric_factory
        self.execution_result_factory = execution_result_factory
        self.pending: Any | None = None
        self.phase = "ready"
        self.metadata: dict[str, Any] = {}
        self.completed_evidence: dict[str, dict[str, str]] = {}
        self._needs_reload = False
        if store.path.exists():
            state = store.restore(
                node_factory=node_factory, journal_factory=journal_factory,
                metric_factory=metric_factory, worst_metric_factory=worst_metric_factory,
            )
            agent.journal, agent.data_preview = state.journal, state.preview
            self.pending, self.phase, self.metadata = state.pending, state.phase, state.metadata
            self.completed_evidence = self.metadata.get("completed_evidence", {})
            random.setstate(state.rng_state)
        else:
            if agent.journal.nodes:
                raise ValueError("a fresh AIDE session must start with an empty original Journal")
            self._save("ready")
        if self.phase in {"ready", "generating"}:
            if self.pending is not None:
                raise ValueError("unexpected pending node in AIDE phase")
        elif self.phase in {"generated", "executing", "execution_complete", "evaluating",
                            "executed", "reviewing", "reviewed"}:
            if self.pending is None:
                raise ValueError("AIDE phase lost its pending Node")
        else:
            raise ValueError("unsupported AIDE session phase")
        self._validate_completed_history()
        self._original_generate = {name: getattr(agent, name)
                                   for name in ("_draft", "_debug", "_improve")}
        self._original_parse = agent.parse_exec_result
        for name in self._original_generate:
            setattr(agent, name, self._wrap_generation(name))
        agent.parse_exec_result = self._review

    def _save(self, phase: str, **metadata: Any) -> None:
        self.metadata.update(metadata)
        self.metadata["completed_evidence"] = dict(self.completed_evidence)
        self.store.save(journal=self.agent.journal, pending=self.pending,
                        preview=self.agent.data_preview, phase=phase, metadata=self.metadata)
        self.phase = phase

    def _wrap_generation(self, name: str) -> Callable[..., Any]:
        def generate(*args: Any) -> Any:
            if self.phase != "ready" or self.pending is not None:
                raise RuntimeError("new AIDE generation attempted with an unresolved phase")
            parent = args[0] if args else None
            self.metadata = {"operation": name, "parent_id": None if parent is None else parent.id,
                             "generation_call_start": self.calls.count}
            # Original parent selection and preview already happened. Capture
            # RNG here so a resume need not call either one again.
            self._save("generating")
            return self._generate_pending()
        return generate

    def _generate_pending(self) -> Any:
        name, parent_id = self.metadata["operation"], self.metadata["parent_id"]
        if name not in self._original_generate:
            raise ValueError("unknown persisted AIDE generation operation")
        parent = None if parent_id is None else next(
            (node for node in self.agent.journal.nodes if node.id == parent_id), None,
        )
        if (name == "_draft") != (parent_id is None) or (parent_id is not None and parent is None):
            raise ValueError("persisted AIDE parent/operation mismatch")
        with self.calls.phase(self.metadata["generation_call_start"]):
            node = self._original_generate[name](*(() if parent is None else (parent,)))
        if node.parent is not parent or node.step is not None:
            raise ValueError("original AIDE generation returned an unexpected Node")
        self.pending = node
        self._save("generated", code_sha256=hashlib.sha256(node.code.encode()).hexdigest())
        return node

    def _execution_phase(self, phase: str, payload: Mapping[str, Any]) -> None:
        if self.pending is None or payload.get("node_id") != self.pending.id:
            raise ValueError("AIDE execution phase belongs to a different pending node")
        self._save(phase, execution_phase=dict(payload))

    def _execute(self, code: str, reset_session: bool) -> Any:
        if self.pending is None or code != self.pending.code or reset_session is not True:
            raise ValueError("original AIDE callback changed code or reset semantics")
        if self.phase != "generated":
            raise RuntimeError("cannot start a second AIDE execution")
        bundle = self.executor.execute(node_id=self.pending.id, code=code,
                                       phase_callback=self._execution_phase)
        return self._accept_bundle(bundle)

    def _validate_bundle(self, bundle: Mapping[str, Any]) -> None:
        if self.pending is None or bundle["node_id"] != self.pending.id:
            raise ValueError("AIDE execution bundle has a different pending node")
        evaluation = bundle["evaluation"]
        # Technical grader failures must remain pending before any LLM bug
        # review or Journal insertion. Only explicit execution/output failure
        # may become a buggy-program node.
        if evaluation.get("valid") is not True:
            diagnostic = evaluation.get("public_diagnostic") or {}
            execution_failure = (evaluation.get("execution_valid") is False
                                 and evaluation.get("score_bearing_evaluation_attempted") is False
                                 and diagnostic.get("category") in {"candidate_execution", "candidate_output"})
            # A scored run whose output the official evaluator rejected, and which the
            # evaluator attributes to the candidate alone, is a buggy program too.
            scored_candidate_failure = (evaluation.get("execution_valid") is True
                                        and diagnostic.get("category") == "official_evaluation")
            if diagnostic.get("candidate_only") is not True or not (execution_failure or scored_candidate_failure):
                raise AideExecutionUnresolved("technical evaluation cannot label an AIDE program buggy")
        else:
            metric = evaluation.get("metrics", {}).get("aggregate_improvement")
            if (evaluation.get("execution_valid") is not True
                    or evaluation.get("score_bearing_evaluation_attempted") is not True
                    or isinstance(metric, bool) or not isinstance(metric, (int, float))
                    or not math.isfinite(metric)):
                raise AideExecutionUnresolved("AIDE valid execution has no authoritative finite score")

    def _accept_bundle(self, bundle: Mapping[str, Any]) -> Any:
        self._validate_bundle(bundle)
        self._save("executed", bundle_sha256=canonical(bundle))
        return self.execution_result_factory(**bundle["execution_result"])

    def _load_bundle(self) -> dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("AIDE execution bundle requires a pending node")
        bundle = self.executor.reconcile(node_id=self.pending.id, code=self.pending.code,
                                         phase_callback=self._execution_phase)
        expected = self.metadata.get("bundle_sha256")
        if expected is not None and canonical(bundle) != expected:
            raise ValueError("AIDE pending execution bundle changed")
        return bundle

    def _review(self, node: Any, exec_result: Any) -> None:
        if node is not self.pending or self.phase not in {"executed", "reviewing"}:
            raise RuntimeError("AIDE review is not bound to its pending execution")
        bundle = self._load_bundle()
        self._validate_bundle(bundle)
        if canonical({key: getattr(exec_result, key) for key in bundle["execution_result"]}) != canonical(bundle["execution_result"]):
            raise ValueError("AIDE review received a different original ExecutionResult")
        if self.phase != "reviewing":
            self._save("reviewing", review_call_start=self.calls.count, bundle_sha256=canonical(bundle))
        with self.calls.phase(self.metadata["review_call_start"]):
            self._original_parse(node=node, exec_result=exec_result)
        binding = apply_official_evaluation(
            node=node, evaluation=bundle["evaluation"],
            metric_value_factory=self.metric_factory, worst_metric_factory=self.worst_metric_factory,
        )
        atomic_json(self.executor.artifacts / node.id / "review_binding.json", seal(binding))
        self._save("reviewed", review_binding_sha256=canonical(binding))

    def _after_append(self) -> None:
        if (self.phase != "reviewed" or self.pending is None
                or not self.agent.journal.nodes or self.agent.journal.nodes[-1] is not self.pending):
            raise RuntimeError("original AIDE Journal did not append the reviewed pending Node")
        self.completed_evidence[self.pending.id] = {
            "bundle_sha256": self.metadata["bundle_sha256"],
            "review_binding_sha256": self.metadata["review_binding_sha256"],
        }
        self.pending = None
        self.metadata = {}
        self._save("ready")

    def _validate_reviewed(self) -> None:
        node = self.pending
        if node is None:
            raise ValueError("reviewed AIDE phase lost its node")
        binding_path = self.executor.artifacts / node.id / "review_binding.json"
        binding = unseal(json.loads(binding_path.read_text()))
        bundle = self._load_bundle()
        self._validate_bundle(bundle)
        self._validate_binding(node, bundle, binding, self.metadata["review_binding_sha256"])

    def _validate_binding(self, node: Any, bundle: Mapping[str, Any],
                          binding: Mapping[str, Any], expected_hash: str) -> None:
        evaluation_sha = hashlib.sha256(json.dumps(
            bundle["evaluation"], sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        if (canonical(binding) != expected_hash
                or binding["node_id"] != node.id
                or binding["parent_id"] != (None if node.parent is None else node.parent.id)
                or binding["code_sha256"] != hashlib.sha256(node.code.encode()).hexdigest()
                or binding["official_evaluation_sha256"] != evaluation_sha
                or binding["effective_metric"] != node.metric.value
                or binding["effective_is_buggy"] != node.is_buggy
                or binding["llm_review"]["analysis"] != node.analysis):
            raise ValueError("reviewed AIDE node no longer matches its evidence binding")

    def _validate_completed_history(self) -> None:
        nodes = self.agent.journal.nodes
        if set(self.completed_evidence) != {node.id for node in nodes}:
            raise ValueError("completed AIDE node lost its checkpoint-bound evidence anchors")
        for node in nodes:
            anchor = self.completed_evidence[node.id]
            # This executor API is explicitly read-only; missing score evidence
            # cannot be repaired by launching a fresh evaluation during restore.
            bundle = self.executor.verify_completed(node_id=node.id, code=node.code)
            if canonical(bundle) != anchor["bundle_sha256"]:
                raise ValueError("completed AIDE execution bundle changed")
            binding = unseal(json.loads((self.executor.artifacts / node.id / "review_binding.json").read_text()))
            self._validate_binding(node, bundle, binding, anchor["review_binding_sha256"])

    def step(self) -> None:
        if self._needs_reload:
            raise RuntimeError("reconstruct AIDE session from checkpoint after an exception")
        try:
            if self.phase == "ready":
                self.agent.step(exec_callback=self._execute)
                self._after_append()
                return
            if self.phase == "generating":
                self._generate_pending()
            if self.phase == "generated":
                self._execute(self.pending.code, True)
            elif self.phase in {"executing", "execution_complete", "evaluating"}:
                self._accept_bundle(self._load_bundle())
            if self.phase in {"executed", "reviewing"}:
                bundle = self._load_bundle()
                result = self.execution_result_factory(**bundle["execution_result"])
                self._review(self.pending, result)
            if self.phase != "reviewed":
                raise RuntimeError("AIDE pending phase was not resolved")
            self._validate_reviewed()
            self.agent.journal.append(self.pending)
            self._after_append()
        except BaseException:
            # The durable phase, not a partially mutated in-memory Node/RNG,
            # remains authoritative after an interrupted original method.
            self._needs_reload = True
            raise
