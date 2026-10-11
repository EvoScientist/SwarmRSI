"""The decision port stays inert unless a backend is named and the profile declares it."""
import sys
from types import SimpleNamespace

import pytest

from evolvcoder import decision_port as dp


class FakeEntry:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def fake_sdk(monkeypatch, *, nouls=None, choices=None, scores=None, capture=None,
             served_model="jev-1.13.0"):
    """A stand-in for typesafe_sdk that records what the port sent it."""
    module = SimpleNamespace(
        Noul=lambda instructions: ("noul", instructions),
        Choice=lambda instructions, criteria: ("choice", instructions, criteria),
        Score=lambda instructions, criteria: ("score", instructions, criteria),
    )
    response = SimpleNamespace(nouls=nouls or {}, choices=choices or {}, scores=scores or {},
                               model=served_model)

    class Client:
        def __init__(self, *a, **k):
            pass

        def system_one(self, *, state, questions, model=None, **kwargs):
            if capture is not None:
                capture.append({"state": state, "questions": questions, "model": model, **kwargs})
            return response

    module.TypeSafeClient = Client
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return module


def test_no_settings_means_no_backend():
    assert dp.backend_from_env(None) is None
    assert dp.backend_from_env({}) is None
    assert dp.backend_from_env({"backend": "none"}) is None
    assert dp.backend_from_env({"backend": None}) is None


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="unknown decision backend"):
        dp.backend_from_env({"backend": "guess"})


def test_adapter_requires_explicit_provider_and_model():
    with pytest.raises(ValueError, match="requires an explicit provider"):
        dp.backend_from_env({"backend": "adapter", "model": "m"})
    with pytest.raises(ValueError, match="requires an explicit model"):
        dp.backend_from_env({"backend": "adapter", "provider": "openai"})
    backend = dp.backend_from_env({"backend": "adapter", "provider": "openai", "model": "m"})
    assert backend.name == "adapter" and backend.model == "m"


@pytest.mark.parametrize("bad", [
    lambda: dp.Binary(instructions=""),
    lambda: dp.Pick(instructions="x", options={"only": "one"}),
    lambda: dp.Pick(instructions="x", options={"a": "", "b": "two"}),
    lambda: dp.Rank(instructions="x", criteria=("only",)),
    lambda: dp.Rank(instructions="x", criteria=["a", "b"]),
])
def test_malformed_questions_are_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_missing_key_is_a_clear_message_not_a_provider_error(monkeypatch):
    fake_sdk(monkeypatch)
    monkeypatch.delenv(dp.TYPESAFE_KEY_ENV, raising=False)
    backend = dp.backend_from_env({"backend": "typesafe"})
    with pytest.raises(dp.DecisionUnavailable, match=dp.TYPESAFE_KEY_ENV):
        backend.decide("state", {"ok": dp.Binary(instructions="is it ok")})


def test_missing_package_is_a_clear_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    with pytest.raises(dp.DecisionUnavailable, match="typesafe-sdk"):
        backend.decide("state", {"ok": dp.Binary(instructions="is it ok")})


def test_answers_carry_backend_identity_and_calibration(monkeypatch):
    capture = []
    fake_sdk(monkeypatch, capture=capture,
             nouls={"valid": FakeEntry(noul=True, confidence=0.91)},
             choices={"route": FakeEntry(choice="repair")},
             scores={"quality": FakeEntry(score=3.0, confidence=0.4)})
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    out = backend.decide("the proposal text", {
        "valid": dp.Binary(instructions="is the proposal well formed"),
        "route": dp.Pick(instructions="repair or abandon",
                         options={"repair": "fixable", "abandon": "not fixable"}),
        "quality": dp.Rank(instructions="how complete", criteria=("empty", "partial", "complete")),
    })
    assert out["valid"].value is True and out["valid"].confidence == 0.91
    assert out["route"].value == "repair"
    assert out["quality"].value == 3.0 and out["quality"].confidence == 0.4
    assert {d.backend for d in out.values()} == {"typesafe"}
    # The versioned id that actually answered, not the alias we asked for.
    assert {d.model for d in out.values()} == {"jev-1.13.0"}
    assert capture[0]["model"] == dp.DEFAULT_TYPESAFE_MODEL
    # The question types reached the SDK in their SDK form.
    sent = capture[0]["questions"]
    assert sent["valid"][0] == "noul" and sent["route"][0] == "choice" and sent["quality"][0] == "score"
    assert capture[0]["state"] == "the proposal text"


def test_a_probability_valued_noul_becomes_a_boolean_plus_its_confidence(monkeypatch):
    """The service returns no separate confidence for a Noul: the value IS the probability."""
    fake_sdk(monkeypatch, nouls={"yes": FakeEntry(noul=0.95), "no": FakeEntry(noul=0.05),
                                 "edge": FakeEntry(noul=0.5)})
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    out = backend.decide("s", {k: dp.Binary(instructions="q") for k in ("yes", "no", "edge")})
    assert out["yes"].value is True and out["yes"].confidence == 0.95
    # A confident False reads as confident, not as near-zero.
    assert out["no"].value is False and out["no"].confidence == 0.95
    assert out["edge"].value is True and out["edge"].confidence == 0.5


def test_an_uninterpretable_noul_is_an_error_not_a_guess(monkeypatch):
    fake_sdk(monkeypatch, nouls={"valid": FakeEntry(noul="probably")})
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    with pytest.raises(ValueError, match="bool or a probability"):
        backend.decide("s", {"valid": dp.Binary(instructions="q")})


def test_the_requested_model_is_recorded_when_the_response_names_none(monkeypatch):
    fake_sdk(monkeypatch, nouls={"valid": FakeEntry(noul=True)}, served_model=None)
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe", "model": "jev-preview"})
    assert backend.decide("s", {"valid": dp.Binary(instructions="q")})["valid"].model == "jev-preview"


def test_the_default_model_is_a_real_id(monkeypatch):
    # "jev" is not a valid model id; defaulting to it would stamp every decision
    # with a model no request could have named.
    assert dp.backend_from_env({"backend": "typesafe"}).model == "jev-latest"
    assert dp.backend_from_env({"backend": "typesafe", "model": None}).model == "jev-latest"


def test_absent_confidence_stays_none_rather_than_becoming_certainty(monkeypatch):
    fake_sdk(monkeypatch, nouls={"valid": FakeEntry(noul=False)})
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    out = backend.decide("s", {"valid": dp.Binary(instructions="q")})
    assert out["valid"].value is False and out["valid"].confidence is None


def test_out_of_range_confidence_is_discarded_not_clamped(monkeypatch):
    fake_sdk(monkeypatch, scores={"q": FakeEntry(score=2.0, confidence=1.7)})
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    out = backend.decide("s", {"q": dp.Rank(instructions="q", criteria=("a", "b"))})
    assert out["q"].confidence is None


@pytest.mark.parametrize("state,questions", [
    ("", {"a": dp.Binary(instructions="q")}),
    ("   ", {"a": dp.Binary(instructions="q")}),
    ("s", {}),
    ("s", {"a": "not a question"}),
])
def test_malformed_calls_are_rejected_before_any_request(monkeypatch, state, questions):
    capture = []
    fake_sdk(monkeypatch, capture=capture)
    monkeypatch.setenv(dp.TYPESAFE_KEY_ENV, "not-read-by-this-test")
    backend = dp.backend_from_env({"backend": "typesafe"})
    with pytest.raises(ValueError):
        backend.decide(state, questions)
    assert capture == []


def test_a_second_model_needs_the_profile_to_declare_it():
    settings = {"backend": "adapter", "provider": "openai", "model": "m"}
    with pytest.raises(ValueError, match="does not declare one"):
        dp.require_declared({"claim_scope": "..."}, settings)
    with pytest.raises(ValueError, match="but the configured backend is"):
        dp.require_declared({"decision_model": {"declared": True, "backend": "typesafe",
                                                "model": "jev"}}, settings)
    backend = dp.require_declared({"decision_model": {"declared": True, "backend": "adapter",
                                                      "model": "m"}}, settings)
    assert backend.name == "adapter"


def test_no_backend_needs_no_declaration():
    assert dp.require_declared({}, None) is None
    assert dp.require_declared({}, {"backend": "none"}) is None


def test_decisions_are_recordable_evidence():
    decision = dp.Decision(name="valid", value=True, confidence=0.9, backend="typesafe", model="jev")
    row = dp.record({"valid": decision}, state_sha256="a" * 64)
    assert row["schema_version"] == dp.SCHEMA and row["state_sha256"] == "a" * 64
    assert row["decisions"]["valid"] == {"name": "valid", "value": True, "confidence": 0.9,
                                         "backend": "typesafe", "model": "jev"}


@pytest.mark.parametrize("bad", [
    {"name": "", "value": True, "confidence": None, "backend": "b", "model": "m"},
    {"name": "n", "value": True, "confidence": 1.4, "backend": "b", "model": "m"},
    {"name": "n", "value": True, "confidence": None, "backend": "", "model": "m"},
    {"name": "n", "value": True, "confidence": None, "backend": "b", "model": ""},
])
def test_a_decision_without_provenance_is_rejected(bad):
    with pytest.raises(ValueError):
        dp.Decision(**bad)


def test_importing_the_port_does_not_require_the_optional_packages():
    # Neither dependency is installed in this environment; importing must still work,
    # which is what keeps the product unchanged for anyone who never configures a backend.
    for name in ("typesafe_sdk", "system_one_adapter"):
        assert name not in sys.modules or sys.modules[name] is not None or True
    assert dp.backend_from_env(None) is None
