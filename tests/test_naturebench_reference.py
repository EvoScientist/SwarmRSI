"""The Claude Code thresholds come from a keyed lookup, and disagreement is fatal."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy/naturebench/reference_from_index.py"
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _index(tmp_path, rows):
    path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _trace(tmp_path, case, values):
    d = tmp_path / "traces" / case
    d.mkdir(parents=True)
    (d / "submissions.jsonl").write_text(
        "".join(json.dumps({"attempt": i + 1, "aggregate_improvement": v}) + "\n"
                for i, v in enumerate(values)))
    return tmp_path / "traces"


ROWS = [
    {"harness": "claude-code", "model": "deepseek-v4-pro", "case_id": "s1",
     "effective_improvement": 0.25, "judge_valid": True},
    {"harness": "codex", "model": "gpt-5.4", "case_id": "s1",
     "effective_improvement": -0.75, "judge_valid": True},
]


def _run(tmp_path, index, out, traces=None, model="deepseek-v4-pro"):
    cmd = [sys.executable, str(SCRIPT), "--index", str(index), "--model", model, "--out", str(out)]
    if traces:
        cmd += ["--traces", str(traces)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_selects_the_named_configuration_not_a_neighbouring_one(tmp_path):
    """The scrape this replaces returned Codex + GPT-5.4 under the DeepSeek name.

    Both configurations carry the same case id here, so a lookup that keys on the
    row rather than on position is the only thing that tells them apart.
    """
    out = tmp_path / "ref.json"
    assert _run(tmp_path, _index(tmp_path, ROWS), out).returncode == 0
    tasks = json.loads(out.read_text())["tasks"]
    assert tasks["s1"]["effective_improvement"] == 0.25


def test_disagreement_with_the_trace_is_fatal(tmp_path):
    """A threshold that its own trace does not support must stop the build."""
    traces = _trace(tmp_path, "s1", [-0.4, 0.10])          # best is 0.10, index says 0.25
    result = _run(tmp_path, _index(tmp_path, ROWS), tmp_path / "ref.json", traces)
    assert result.returncode != 0
    assert "disagree" in (result.stdout + result.stderr)
    assert not (tmp_path / "ref.json").exists()


def test_agreement_with_the_trace_passes(tmp_path):
    traces = _trace(tmp_path, "s1", [-0.4, 0.25])
    out = tmp_path / "ref.json"
    assert _run(tmp_path, _index(tmp_path, ROWS), out, traces).returncode == 0
    assert json.loads(out.read_text())["crosschecked_against_traces"] is True


def test_an_absent_configuration_is_refused(tmp_path):
    result = _run(tmp_path, _index(tmp_path, ROWS), tmp_path / "ref.json", model="no-such-model")
    assert result.returncode != 0 and not (tmp_path / "ref.json").exists()
