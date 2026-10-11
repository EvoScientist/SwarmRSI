"""Search and terminal sandboxes must see the same model cache, offline."""
import sys
from pathlib import Path

from evolvcoder import naturebench_evaluator as evaluator, naturebench_session as session


def test_search_and_terminal_sandboxes_share_the_cache_environment(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'; (cache / 'transformers').mkdir(parents=True)
    task = tmp_path / 'task'; (task / 'problem/data').mkdir(parents=True)
    monkeypatch.setattr(session.shutil, 'which', lambda name: '/synthetic/bwrap')
    monkeypatch.setattr(session, 'CANDIDATE_RUNTIME', None)
    monkeypatch.setattr(session, 'RUNTIME', Path(sys.executable))
    monkeypatch.setattr(session, 'MODEL_CACHE', cache)
    search = dict(session.coder_sandbox_config(task)['environment'])
    _, _, terminal, _ = evaluator._candidate_command(
        workspace=tmp_path, data_dir=task / 'problem/data', runtime_python=Path(sys.executable),
        require_sandbox=True, environment={}, inherit_model_cache=False, model_cache=cache)
    shared = dict(evaluator.MODEL_CACHE_ENVIRONMENT)
    # A TORCH_HOME alone let `import torchvision` die migrating the read-only
    # /model_cache/transformers; without the offline flags deberta never loaded.
    assert {'HF_HOME', 'TRANSFORMERS_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'} <= set(shared)
    assert {k: search[k] for k in shared} == shared == {k: terminal[k] for k in shared}
