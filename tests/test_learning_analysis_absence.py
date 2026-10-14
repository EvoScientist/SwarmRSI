"""Bounded negative-state snapshot race; synthetic local journals only."""
import pytest

from swarmrsi import learning_analysis as analysis
from test_learning_analysis_integrity import halted_chain
from test_naturebench_entrypoint import inputs  # noqa: F401


def test_final_appearing_between_unavailable_check_and_watch_is_rejected(inputs, tmp_path, monkeypatch):
    chain, _, trial = halted_chain(inputs, tmp_path)
    original_verify = analysis.verify_unavailable
    created = []

    def terminal_appears(record):
        identity = original_verify(record)
        final = trial / "run/final.json"
        final.parent.mkdir(parents=True, exist_ok=True)
        final.write_text("{}\n")
        created.append(final)
        return identity

    monkeypatch.setattr(analysis, "verify_unavailable", terminal_appears)
    with pytest.raises(ValueError):
        analysis.analyze_learning(chain_root=chain.root)
    assert created
