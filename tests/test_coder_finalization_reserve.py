"""Controller deadline invariants; no model API calls and no score claims."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evolvcoder import coder_engine
from evolvcoder import CoderExitStatus, ContractBoundedCoder
from test_evolvcoder_coder import config, workspace


@pytest.mark.parametrize("reserve", [0, -1, float("nan"), float("inf"), 600, True])
def test_invalid_reserve_is_rejected(tmp_path, reserve):
    with pytest.raises(ValueError, match="finalization reserve"):
        replace(config(workspace(tmp_path)), finalization_reserve_seconds=reserve)


def test_default_config_has_no_new_identity_field(tmp_path):
    cfg = config(workspace(tmp_path))
    assert "finalization_reserve_seconds" not in cfg.canonical()
    assert replace(cfg, finalization_reserve_seconds=None).config_sha256 == cfg.config_sha256


@pytest.mark.parametrize("replacement,success", [("return 'good'", True), ("return ('good'", False)])
def test_reserve_validates_changed_program_without_another_model_call(tmp_path, monkeypatch,
                                                                    replacement, success):
    clock = [0.0]
    monkeypatch.setattr(coder_engine, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None))

    class DeadlineModel:
        request_total_timeout_seconds = 100.0
        calls = 0

        def respond(self, payload):
            self.calls += 1
            assert self.calls == 1
            assert self.request_total_timeout_seconds == 10.0
            assert payload["budget_remaining"]["wall_seconds"] == 10
            clock[0] = 10.0
            return {"tool": "edit", "path": "solution.py", "old": "return 'bad'",
                    "new": replacement}, {"input_tokens": 1, "output_tokens": 1}

    root = workspace(tmp_path)
    model = DeadlineModel()
    cfg = replace(config(root), max_wall_seconds=30, finalization_reserve_seconds=20,
                  config_sha256="")
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert (outcome.status == CoderExitStatus.SUCCESS) is success
    assert model.request_total_timeout_seconds == 100.0
    assert any(e["kind"] == "coder.finalization_started" for e in outcome.events)
    assert outcome.model_calls == 1
    assert outcome.parent_preserved
    if success:
        assert outcome.reason == "controller_validation_after_early_termination"
        assert outcome.final_sha256 != outcome.parent_sha256
    else:
        assert outcome.final_sha256 == outcome.parent_sha256
