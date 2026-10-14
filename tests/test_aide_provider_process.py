from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from swarmrsi.baselines.aide_provider_process import AideProviderProcess
from swarmrsi.baselines.aide_session import AidePhaseCalls
from swarmrsi.baselines.aide_transport import (
    AideCallLedger, AideResourceStop, AideTransportUnresolved,
)


HELPER = Path(__file__).resolve().parents[1] / "scripts/run_aide_provider_once.py"
MODEL = "deepseek/deepseek-v4-pro"
RAW = {"model": MODEL, "usage": {"prompt_tokens": 4, "completion_tokens": 3},
       "choices": [{"message": {"content": "ok", "role": "assistant"}}]}


def transport(*, remaining=lambda: 20.0):
    return AideProviderProcess(
        python=Path(sys.executable), helper=HELPER,
        remaining_seconds=remaining,
        environment={"OPENROUTER_API_KEY": "synthetic-test-secret"},
    )


def test_preserves_reserved_request_and_refreshes_hard_deadline(monkeypatch):
    sent = []

    def run(argv, **kwargs):
        sent.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "schema_version": "aide-provider-worker-v1", "response": RAW}), stderr="")

    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process", run)
    request = {"model": MODEL, "timeout": 80, "messages": [{"role": "user", "content": "fixture"}],
               "max_tokens": 25, "extra_body": {"provider": {"order": ["Fireworks"]}}}
    assert transport(remaining=lambda: 7.0)(**request) == RAW
    argv, kwargs = sent[0]
    assert json.loads(kwargs["input"]) == request
    assert kwargs["timeout"] == 7 and kwargs["teardown_seconds"] == 1
    assert "synthetic-test-secret" not in str(argv) + kwargs["input"]


def test_keeps_virtualenv_interpreter_invocation_path(tmp_path, monkeypatch):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    sent = []

    def run(argv, **kwargs):
        sent.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "schema_version": "aide-provider-worker-v1", "response": RAW}), stderr="")

    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process", run)
    provider = AideProviderProcess(python=python, helper=HELPER,
        remaining_seconds=lambda: 10, environment={"OPENROUTER_API_KEY": "fixture"})
    assert provider(model=MODEL, timeout=10) == RAW
    assert sent[0][0] == str(python)
    assert sent[0][0] != str(python.resolve())


@pytest.mark.parametrize("remaining", [-2.0, 0.0, 0.9, 1.0])
def test_expired_deadline_never_starts_worker(monkeypatch, remaining):
    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process",
                        lambda *a, **k: pytest.fail("expired worker was dispatched"))
    with pytest.raises(AideResourceStop, match="provider_wall"):
        transport(remaining=lambda: remaining)(model=MODEL, timeout=20)


@pytest.mark.parametrize("failure", ["timeout", "exit", "invalid_json", "invalid_receipt"])
def test_worker_uncertainty_is_durable_and_not_retried(tmp_path, monkeypatch, failure):
    dispatched = []

    def run(argv, **kwargs):
        dispatched.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output="synthetic-test-secret")
        return SimpleNamespace(returncode=1 if failure == "exit" else 0,
                               stdout="synthetic-test-secret" if failure != "invalid_receipt" else "{}",
                               stderr="synthetic-test-secret")

    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process", run)
    ledger = AideCallLedger(tmp_path, model=MODEL, max_calls=3, max_output_tokens=100,
                           wall_seconds=100, per_call_output_tokens=20, create=transport())
    for _ in range(2):
        with pytest.raises(AideTransportUnresolved) as error:
            ledger.create(model=MODEL)
        assert "synthetic-test-secret" not in str(error.value)
    assert len(dispatched) == 1
    assert ledger.read()["attempts"][0]["state"] == "unresolved"
    assert "synthetic-test-secret" not in ledger.path.read_text()


def test_successful_worker_response_accounted_before_replay(tmp_path, monkeypatch):
    sent = []

    def run(argv, **kwargs):
        sent.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "schema_version": "aide-provider-worker-v1", "response": RAW}), stderr="")

    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process", run)
    ledger = AideCallLedger(tmp_path, model=MODEL, max_calls=1, max_output_tokens=3,
                           wall_seconds=100, per_call_output_tokens=3, create=transport())
    assert ledger.create(model=MODEL) == RAW
    assert ledger.read()["attempts"][0]["output_tokens"] == 3
    assert ledger.replay(0, logical_request={"model": MODEL}, response_factory=lambda x: x) == RAW
    assert len(sent) == 1


def test_invalid_sdk_schema_is_saved_and_charged_before_parsing(tmp_path, monkeypatch):
    sent = []

    def run(argv, **kwargs):
        sent.append(argv)
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "schema_version": "aide-provider-worker-v1", "response": RAW}), stderr="")

    def reject_schema(raw):
        assert raw == RAW
        raise ValueError("synthetic SDK schema rejection")

    monkeypatch.setattr("swarmrsi.baselines.aide_provider_process.run_bounded_process", run)
    ledger = AideCallLedger(tmp_path, model=MODEL, max_calls=1, max_output_tokens=3,
                           wall_seconds=100, per_call_output_tokens=3, create=transport())
    for _ in range(2):
        phase_calls = AidePhaseCalls(ledger, reject_schema)
        with pytest.raises(ValueError, match="schema rejection"):
            with phase_calls.phase(0):
                phase_calls.create(model=MODEL)
        assert ledger.read()["attempts"][0]["state"] == "completed"
        assert ledger.read()["attempts"][0]["output_tokens"] == 3
        assert (tmp_path / "000000.response.json").is_file()
    assert len(sent) == 1


@pytest.mark.parametrize("sdk_failure", [False, True])
def test_helper_exact_one_sdk_attempt_and_no_secret_error_text(monkeypatch, capsys, sdk_failure):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append(("close",))

        def create(self, **kwargs):
            calls.append(("create", kwargs))
            print("synthetic-test-secret")
            print("synthetic-test-secret", file=sys.stderr)
            if sdk_failure:
                raise RuntimeError("synthetic-test-secret")
            return SimpleNamespace(model_dump=lambda **kw: RAW)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-test-secret")
    request = {"model": MODEL, "timeout": 10, "max_tokens": 3}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
    spec = importlib.util.spec_from_file_location("aide_provider_worker_fixture", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.main() == int(sdk_failure)
    stdout, stderr = capsys.readouterr()
    assert "synthetic-test-secret" not in stdout + stderr
    receipt = json.loads(stdout)
    assert calls[0][1] == {"api_key": "synthetic-test-secret", "max_retries": 0,
                           "base_url": "https://openrouter.ai/api/v1"}
    assert [c[0] for c in calls] == ["init", "create", "close"]
    assert calls[1][1] == request
    assert receipt.get("error_type") == ("RuntimeError" if sdk_failure else None)
    if not sdk_failure:
        assert receipt["response"] == RAW


@pytest.mark.parametrize("hang", [False, True])
def test_actual_helper_process_without_network(tmp_path, hang):
    """Run the real helper/timeout path, importing only a fake SDK module."""
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    calls_path = tmp_path / "calls.txt"
    (sdk / "openai.py").write_text(
        "import os,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "class OpenAI:\n"
        " def __init__(self,**kw):\n"
        "  assert kw['max_retries']==0\n"
        "  assert kw['base_url']=='https://openrouter.ai/api/v1'\n"
        "  self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))\n"
        " def __enter__(self): return self\n"
        " def __exit__(self,*a): pass\n"
        " def create(self,**kw):\n"
        "  p=Path(os.environ['AIDE_TEST_CALL_MARKER'])\n"
        "  with p.open('a') as f: f.write('called\\n')\n"
        "  if os.environ['AIDE_TEST_HANG']=='1': time.sleep(10)\n"
        f"  return SimpleNamespace(model_dump=lambda **kw: {RAW!r})\n"
    )
    provider = AideProviderProcess(
        python=Path(sys.executable), helper=HELPER,
        # The hang case must leave the helper enough budget to reach the fake
        # SDK before the deadline cuts it: on a loaded cluster node a 2 s budget
        # expired during interpreter spawn, so the marker was never written and
        # the test failed for the wrong reason. 6 s is still well under the
        # SDK's 10 s sleep, so the deadline is what ends the call.
        remaining_seconds=lambda: 6.0 if hang else 10.0,
        environment={**os.environ, "PYTHONPATH": str(sdk),
                     "PYTHONDONTWRITEBYTECODE": "1",
                     "OPENROUTER_API_KEY": "synthetic-test-secret",
                     "AIDE_TEST_CALL_MARKER": str(calls_path),
                     "AIDE_TEST_HANG": str(int(hang))},
    )
    ledger = AideCallLedger(tmp_path / "ledger", model=MODEL, max_calls=3,
                           max_output_tokens=20, wall_seconds=20,
                           per_call_output_tokens=10, create=provider)
    if hang:
        for _ in range(2):
            with pytest.raises(AideTransportUnresolved):
                ledger.create(model=MODEL)
        # Two hung calls under a 2 s remaining budget. The proof that the
        # deadline fired is semantic, not temporal: the attempt is recorded as
        # unresolved (a helper that ignored the deadline would complete and
        # return RAW), and the fake SDK was entered exactly once (asserted
        # below). A wall-clock bound was tried at 4 s, 10 s and 20 s and each
        # failed a cluster qualification purely on subprocess spawn under load
        # (whole suites of 22-62 minutes), so it is not asserted.
        assert ledger.read()["attempts"][0]["state"] == "unresolved"
    else:
        assert ledger.create(model=MODEL) == RAW
        assert ledger.read()["attempts"][0]["state"] == "completed"
    assert calls_path.read_text().splitlines() == ["called"]
