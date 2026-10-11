from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolvcoder.baselines.aide_openrouter import (
    apply_official_evaluation,
    assert_upstream_hashes,
    install_openrouter_query,
    make_openrouter_query,
)


class FunctionSpec:
    name = "submit_review"
    as_openai_tool_dict = {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }
    openai_tool_choice_dict = {"type": "function", "function": {"name": name}}


def completion(*, content=None, arguments=None):
    tool_calls = []
    if arguments is not None:
        tool_calls = [SimpleNamespace(function=SimpleNamespace(
            name="submit_review", arguments=json.dumps(arguments)
        ))]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=tool_calls
        ))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        system_fingerprint="mock", model="deepseek/deepseek-v4-pro", created=1,
    )


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def direct_create(create, _exceptions, **kwargs):
    return create(**kwargs)


def test_mocked_code_and_submit_review_calls_preserve_upstream_provider_policy():
    client = Client([
        completion(content="plan\n```python\nprint(1)\n```"),
        completion(arguments={
            "is_bug": False, "summary": "ok", "metric": 0.4,
            "lower_is_better": False,
        }),
    ])
    records = []
    query = make_openrouter_query(
        client=client, backoff_create=direct_create,
        timeout_exceptions=(), recorder=records.append,
    )
    code, *_ = query("system", "code", model="deepseek/deepseek-v4-pro")
    review, *_ = query(
        "system", "review", func_spec=FunctionSpec(),
        model="deepseek/deepseek-v4-pro",
    )
    assert "```python" in code
    assert review == {
        "is_bug": False, "summary": "ok", "metric": 0.4,
        "lower_is_better": False,
    }
    assert client.requests[0]["extra_body"]["provider"]["order"] == ["Fireworks"]
    assert client.requests[1]["tools"] == [FunctionSpec.as_openai_tool_dict]
    assert client.requests[1]["tool_choice"] == FunctionSpec.openai_tool_choice_dict
    assert [row["function"] for row in records] == [None, "submit_review"]
    assert sum(row["input_tokens"] for row in records) == 22
    assert sum(row["output_tokens"] for row in records) == 14


def test_installer_changes_only_openrouter_dispatch_slot():
    old = object()
    backend = SimpleNamespace(
        backend_openrouter=SimpleNamespace(query=old),
        provider_to_query_func={"openrouter": old, "openai": "unchanged"},
    )
    replacement = object()
    install_openrouter_query(backend, replacement)
    assert backend.backend_openrouter.query is replacement
    assert backend.provider_to_query_func == {
        "openrouter": replacement, "openai": "unchanged"
    }


def test_upstream_attestation_fails_closed(tmp_path: Path):
    source = tmp_path / "aide" / "agent.py"
    source.parent.mkdir()
    source.write_text("locked\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert_upstream_hashes(tmp_path, {"aide/agent.py": digest})
    source.write_text("changed\n")
    with pytest.raises(RuntimeError, match="attestation mismatch"):
        assert_upstream_hashes(tmp_path, {"aide/agent.py": digest})


def test_official_evaluator_overrides_llm_metric_but_retains_review():
    class Metric:
        def __init__(self, value, *, maximize=True):
            self.value = value
            self.maximize = maximize

    node = SimpleNamespace(
        id="node-1", parent=None, code="print(1)", analysis="LLM summary",
        is_buggy=True, metric=Metric(999.0), exec_time=1.2, exc_type=None,
        term_out="candidate output",
    )
    binding = apply_official_evaluation(
        node=node,
        evaluation={
            "valid": True,
            "execution_valid": True,
            "score_bearing_evaluation_attempted": True,
            "metrics": {"aggregate_improvement": 0.25},
        },
        metric_value_factory=Metric,
        worst_metric_factory=lambda: Metric(float("-inf")),
    )
    assert node.metric.value == 0.25
    assert node.is_buggy is False
    assert node.analysis == "LLM summary"
    assert binding["llm_review"]["metric"] == 999.0
    assert binding["llm_review"]["is_buggy"] is True
    assert binding["llm_analysis_preserved"] is True
    assert len(binding["official_evaluation_sha256"]) == 64
