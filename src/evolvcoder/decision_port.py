"""A port for calibrated structured decisions, with no default backend.

The search loop makes the same few decisions thousands of times per campaign:
is this proposal well formed, is this failure worth repairing, does this program
plausibly satisfy the task. Today each of those costs a full backbone call that
returns prose we then parse, which is both the slowest and the least reliable way
to obtain a boolean.

A "System One" model (TypeSafe AI's Jev is the first) answers exactly that shape:
unstructured state in, a typed value plus a calibrated probability out, in
milliseconds. This module is the seam for one, deliberately written so that:

* **There is no default.** ``backend_from_env`` returns None unless a backend is
  named explicitly. Nothing in the product changes by importing this file.
* **The dependency is optional and lazy.** ``typesafe_sdk`` and
  ``system_one_adapter`` are imported inside the backend that needs them, so the
  package installs and the tests run without either.
* **Using it changes what a run may claim.** A second model inside the loop makes
  the system a two-model system. ``require_declared`` refuses to hand out a
  backend to a benchmark cell whose profile has not declared it, for the same
  reason the Terminal-Bench adapter refuses a package sealed for diagnostics: a
  claim has to be made on purpose, not acquired by an import.
* **Every decision is evidence.** ``Decision`` carries the backend identity, the
  model, the value and the confidence, so a decision the search acted on can be
  re-read afterwards rather than believed.

This module never grades a benchmark result. An official score comes from the
benchmark's own evaluator, never from a judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Protocol

SCHEMA = "evolvcoder-decision-port-v1"

# The env var the TypeSafe SDK itself reads. Named here so callers can check for
# presence without ever reading the value.
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"


class DecisionUnavailable(RuntimeError):
    """No decision backend is configured, or the configured one cannot be reached."""


@dataclass(frozen=True, slots=True)
class Binary:
    """A yes/no question. Maps to the SDK's Noul."""
    instructions: str

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)


@dataclass(frozen=True, slots=True)
class Pick:
    """A choice among named options. Maps to the SDK's Choice."""
    instructions: str
    options: dict[str, str]

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if not isinstance(self.options, dict) or len(self.options) < 2:
            raise ValueError("a Pick needs at least two named options")
        for name, meaning in self.options.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("option names must be nonempty strings")
            if not isinstance(meaning, str) or not meaning.strip():
                raise ValueError(f"option {name!r} needs a description of what it means")


@dataclass(frozen=True, slots=True)
class Rank:
    """A score against ordered criteria, lowest first. Maps to the SDK's Score."""
    instructions: str
    criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if not isinstance(self.criteria, tuple) or len(self.criteria) < 2:
            raise ValueError("a Rank needs at least two ordered criteria as a tuple")
        if any(not isinstance(c, str) or not c.strip() for c in self.criteria):
            raise ValueError("every criterion must be a nonempty string")


def _check_instructions(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a question needs nonempty instructions")


Question = Binary | Pick | Rank


@dataclass(frozen=True, slots=True)
class Decision:
    """One answer, with the identity of whatever produced it.

    ``confidence`` is None when a backend does not report one. A caller that
    gates on confidence must handle that rather than assume certainty.
    """
    name: str
    value: bool | str | float
    confidence: float | None
    backend: str
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a decision needs the question's name")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        for field_name in ("backend", "model"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name).strip():
                raise ValueError(f"a decision must record its {field_name}")

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "confidence": self.confidence,
                "backend": self.backend, "model": self.model}


class DecisionBackend(Protocol):
    name: str
    model: str

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Decision]: ...


def _validate_call(state: str, questions: dict[str, Question]) -> None:
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be nonempty text describing the situation")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one named question is required")
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("question names must be nonempty strings")
        if not isinstance(question, (Binary, Pick, Rank)):
            raise ValueError(f"question {name!r} is not a Binary, Pick or Rank")


# "jev" is not a model id. The service serves jev-1.13.0 behind the aliases
# jev-latest and jev-preview, and the SDK's own default is jev-latest.
DEFAULT_TYPESAFE_MODEL = "jev-latest"


@dataclass
class TypeSafeBackend:
    """The real System One service. Requires TYPESAFE_API_KEY in the environment.

    The key is never read here: the SDK reads it itself, and this class only
    checks that something is set so the failure is a clear message rather than a
    provider error deep in a search.
    """
    model: str = DEFAULT_TYPESAFE_MODEL
    name: str = field(default="typesafe", init=False)
    _client: Any = field(default=None, init=False, repr=False)

    def _sdk(self):
        try:
            import typesafe_sdk
        except ImportError as exc:
            raise DecisionUnavailable(
                "the typesafe-sdk package is not installed; `pip install typesafe-sdk`"
            ) from exc
        if not os.environ.get(TYPESAFE_KEY_ENV):
            raise DecisionUnavailable(
                f"{TYPESAFE_KEY_ENV} is not set; create a key in the TypeSafe console. "
                "Access was early-access gated as of 2026-09-20."
            )
        return typesafe_sdk

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Decision]:
        _validate_call(state, questions)
        sdk = self._sdk()
        if self._client is None:
            self._client = sdk.TypeSafeClient()
        payload = {name: _to_sdk(sdk, question) for name, question in questions.items()}
        # Pass the model explicitly. Letting the SDK fall back to its own default
        # would stamp every Decision with a model the request never named, which
        # is the one thing an evidence record must not do.
        response = self._client.system_one(state=state, questions=payload, model=self.model)
        return _from_sdk(response, questions, self.name, _answered_by(response, self.model))

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            client.close()


@dataclass
class AdapterBackend:
    """The MIT LLM-backed drop-in, for development and for cost comparison.

    This is not a System One model. It routes the same questions to an ordinary
    chat model, so it costs and behaves like one. It exists so the seam can be
    exercised offline and so a run can be priced against the real thing; a result
    produced through it must say so, which is why ``name`` differs.
    """
    provider: str
    model: str
    name: str = field(default="adapter", init=False)
    _client: Any = field(default=None, init=False, repr=False)

    def decide(self, state: str, questions: dict[str, Question]) -> dict[str, Decision]:
        _validate_call(state, questions)
        try:
            import system_one_adapter
            import typesafe_sdk
        except ImportError as exc:
            raise DecisionUnavailable(
                "the system-one-adapter package is not installed; "
                "`pip install 'system-one-adapter[openai]'`"
            ) from exc
        if self._client is None:
            self._client = system_one_adapter.SystemOneAdapterClient(
                structured_outputs=True, llm_answer_mode="probabilities",
                normalize_probabilities=True)
        payload = {name: _to_sdk(typesafe_sdk, question) for name, question in questions.items()}
        response = self._client.system_one(state=state, questions=payload,
                                           provider=self.provider, model=self.model)
        return _from_sdk(response, questions, self.name,
                         _answered_by(response, f"{self.provider}/{self.model}"))

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None and hasattr(client, "close"):
            client.close()


def _to_sdk(sdk, question: Question):
    if isinstance(question, Binary):
        return sdk.Noul(instructions=question.instructions)
    if isinstance(question, Pick):
        return sdk.Choice(instructions=question.instructions, criteria=dict(question.options))
    return sdk.Score(instructions=question.instructions, criteria=list(question.criteria))


def _answered_by(response, requested: str) -> str:
    """The model that actually answered, falling back to the one we asked for."""
    served = getattr(response, "model", None)
    return served if isinstance(served, str) and served.strip() else requested


def _probability(value):
    return float(value) if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 else None


def _confidence(entry):
    """Pull a calibrated probability if the answer carries one; never invent one."""
    for attribute in ("confidence", "probability", "calibrated_probability"):
        found = _probability(getattr(entry, attribute, None))
        if found is not None:
            return found
    return None


def _binary(entry) -> tuple[bool, float | None]:
    """Read a Noul answer.

    The service returns no separate confidence for a Noul: the ``noul`` field is
    itself the probability that the statement is true. Older/typed views may hand
    back a plain bool instead, so both shapes are accepted. ``confidence`` is
    reported for the value actually returned, so a False at p=0.05 reads as 0.95
    confident rather than as near-zero.
    """
    raw = getattr(entry, "noul", None)
    if isinstance(raw, bool):
        return raw, _confidence(entry)
    probability = _probability(raw)
    if probability is None:
        raise ValueError(f"a Noul answer must be a bool or a probability, got {raw!r}")
    value = probability >= 0.5
    return value, probability if value else 1.0 - probability


def _from_sdk(response, questions: dict[str, Question], backend: str, model: str) -> dict[str, Decision]:
    decisions: dict[str, Decision] = {}
    for name, question in questions.items():
        if isinstance(question, Binary):
            entry = response.nouls[name]
            value, confidence = _binary(entry)
        elif isinstance(question, Pick):
            entry = response.choices[name]
            value, confidence = entry.choice, _confidence(entry)
        else:
            entry = response.scores[name]
            value, confidence = entry.score, _confidence(entry)
        decisions[name] = Decision(name=name, value=value, confidence=confidence,
                                   backend=backend, model=model)
    return decisions


def backend_from_env(settings: dict | None = None) -> DecisionBackend | None:
    """Build the backend a caller explicitly asked for, or None.

    There is no discovery and no fallback: an absent or empty ``settings`` means
    the product runs exactly as it did before this module existed.
    """
    if not settings:
        return None
    if not isinstance(settings, dict):
        raise ValueError("decision settings must be a mapping")
    kind = settings.get("backend")
    if kind in (None, "", "none", "off"):
        return None
    if kind == "typesafe":
        return TypeSafeBackend(model=str(settings.get("model") or DEFAULT_TYPESAFE_MODEL))
    if kind == "adapter":
        for required in ("provider", "model"):
            if not settings.get(required):
                raise ValueError(f"the adapter backend requires an explicit {required}")
        return AdapterBackend(provider=str(settings["provider"]), model=str(settings["model"]))
    raise ValueError(f"unknown decision backend {kind!r}; use 'typesafe', 'adapter' or 'none'")


def require_declared(profile: dict, settings: dict | None) -> DecisionBackend | None:
    """Refuse a second model inside the loop unless the profile declares it.

    A judge in the search makes this a two-model system, which changes what a
    campaign may claim about its backbone. That belongs in the profile beside
    ``claim_scope``, where a reader will find it, not in a runtime flag.
    """
    backend = backend_from_env(settings)
    if backend is None:
        return None
    declared = (profile or {}).get("decision_model")
    if not isinstance(declared, dict) or not declared.get("declared"):
        raise ValueError(
            "a decision backend was configured but the profile does not declare one. "
            "Add a decision_model block stating the backend, the model and what it decides, "
            "and say in claim_scope that the system uses a second model, or the run's "
            "backbone comparison would be wrong."
        )
    if declared.get("backend") != backend.name or declared.get("model") != backend.model:
        raise ValueError(
            f"profile declares decision_model {declared.get('backend')}/{declared.get('model')} "
            f"but the configured backend is {backend.name}/{backend.model}"
        )
    return backend


def record(decisions: dict[str, Decision], *, state_sha256: str) -> dict:
    """The evidence row for a batch of decisions the search acted on."""
    return {"schema_version": SCHEMA, "state_sha256": state_sha256,
            "decisions": {name: decision.to_dict() for name, decision in sorted(decisions.items())}}
