from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("warning_stage", ["run", "cleanup"])
def test_upstream_warning_kept_off_receipt_stdout(tmp_path, monkeypatch, capsys, warning_stage):
    """Two logging paths must not corrupt or replace the actual result."""
    source = tmp_path / "aide/interpreter.py"
    source.parent.mkdir()
    source.write_text("# independently mocked helper logging test\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    @dataclass
    class Result:
        term_out: list[str]
        exec_time: float
        exc_type: str
        exc_info: dict
        exc_stack: list

    actual_result = Result(["before timeout\n", "original timeout details"], 2.01, "TimeoutError", {}, [])
    calls = []

    class Interpreter:
        def __init__(self, *args, **kwargs):
            calls.append(("init", kwargs))

        def run(self, code, reset_session):
            calls.append(("run", code, reset_session))
            if warning_stage == "run":
                print("WARNING Execution exceeded timeout of 2s")
            return actual_result

        def cleanup_session(self):
            calls.append(("cleanup",))
            if warning_stage == "cleanup":
                print("WARNING Process failed to terminate, killing immediately")

    monkeypatch.setitem(sys.modules, "aide.interpreter", SimpleNamespace(Interpreter=Interpreter))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "stdin", io.StringIO("print('fixture')"))
    monkeypatch.setattr(sys, "argv", ["helper", "--upstream-root", str(tmp_path),
        "--interpreter-sha256", digest, "--timeout-seconds", "2"])
    path = Path(__file__).resolve().parents[1] / "scripts/run_aide_interpreter_once.py"
    spec = importlib.util.spec_from_file_location("aide_interpreter_helper_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.main() == 0
    stdout, stderr = capsys.readouterr()
    receipt = json.loads(stdout)
    assert "WARNING" in stderr and "WARNING" not in stdout
    assert receipt["execution_result"]["term_out"] == actual_result.term_out
    assert receipt["execution_result"]["exc_type"] == "TimeoutError"
    assert receipt["code_sha256"] == hashlib.sha256(b"print('fixture')").hexdigest()
    assert [entry[0] for entry in calls] == ["init", "run", "cleanup"]
    assert calls[1][2] is True
