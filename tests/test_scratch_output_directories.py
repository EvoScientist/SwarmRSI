"""Scratch output directories are ignored only when a run opts in, and never hide code by default."""
from types import SimpleNamespace

from swarmrsi import coder_engine

ENGINE = next(v for v in vars(coder_engine).values() if isinstance(v, type) and hasattr(v, '_ignored'))
BASE = (".git/**", ".swarmrsi/**", "output/**", "**/__pycache__/**", ".pytest_cache/**", "*.pyc")


def ignored(patterns, relative):
    return ENGINE._ignored(SimpleNamespace(config=SimpleNamespace(ignored_workspace_patterns=patterns)), relative)


def test_default_patterns_do_not_ignore_smoke_outputs():
    assert not ignored(BASE, 'smoke_out/submission.csv')
    assert ignored(BASE, 'output/submission.csv')


def test_opted_in_globs_ignore_named_scratch_directories():
    patterns = BASE + ('*smoke*/**', 'tmp*/**')
    for relative in ('smoke_out/submission.csv', '_smoke_out/a/b.csv', 'tmp_run/x.npy'):
        assert ignored(patterns, relative)
    # Files at the workspace root and unrelated directories are unaffected.
    for relative in ('run.py', 'smoke.py', 'src/model.py', 'features/smoke_out.csv'):
        assert not ignored(patterns, relative)
