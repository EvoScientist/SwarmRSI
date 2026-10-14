"""The coder is told which pretrained weights the offline sandbox can load."""
from pathlib import Path

from swarmrsi import naturebench_session as session


def cache(tmp_path):
    root = tmp_path / 'cache'
    (root / 'hub/checkpoints').mkdir(parents=True)
    (root / 'hub/checkpoints/resnet50-11ad3fa6.pth').write_bytes(b'')
    (root / 'hub/checkpoints/custom-weights.pth').write_bytes(b'')
    for repo in ('models--microsoft--deberta-v3-base', 'models--distilbert-base-uncased'):
        (root / 'transformers' / repo).mkdir(parents=True)
    return root


def test_inventory_lists_checkpoints_and_repo_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'torchvision_weight_calls', lambda: {
        'resnet50-11ad3fa6.pth': 'torchvision.models.resnet50(weights="IMAGENET1K_V2")'})
    inventory = session.offline_model_inventory(cache(tmp_path))
    assert 'custom-weights.pth' in inventory['torchvision']
    assert 'torchvision.models.resnet50(weights="IMAGENET1K_V2")' in inventory['torchvision']
    assert inventory['huggingface'] == ['distilbert-base-uncased', 'microsoft/deberta-v3-base']


def test_no_cache_means_no_inventory(tmp_path):
    assert session.offline_model_inventory(None) == {}
    assert session.offline_model_inventory(tmp_path / 'missing') == {}


def _task(tmp_path):
    root = tmp_path / 'task'
    (root / 'problem/data').mkdir(parents=True, exist_ok=True)
    (root / 'problem/data/train.csv').write_text('id,y\n1,0\n')
    return root


def test_task_context_carries_the_inventory_only_with_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'runtime_dependency_inventory', lambda: (['numpy'], []))
    monkeypatch.setattr(session, 'accelerator_inventory', lambda: {})
    monkeypatch.setattr(session, 'torchvision_weight_calls', lambda: {})
    monkeypatch.setattr(session, 'MODEL_CACHE', None)
    assert 'No pretrained model cache' in session.public_task_context(_task(tmp_path))
    monkeypatch.setattr(session, 'MODEL_CACHE', cache(tmp_path))
    text = session.public_task_context(_task(tmp_path))
    assert 'Frozen offline model inventory' in text and 'microsoft/deberta-v3-base' in text


def test_lock_and_blob_files_are_not_listed(tmp_path):
    root = cache(tmp_path)
    (root / 'transformers/.locks/models--x').mkdir(parents=True)
    (root / 'transformers/.locks/models--x/a.lock').write_text('')
    (root / 'transformers/models--distilbert-base-uncased/blobs').mkdir()
    (root / 'transformers/models--distilbert-base-uncased/blobs/abc').write_text('')
    (root / 'esm').mkdir()
    (root / 'esm/esm2_t6.pt').write_text('')
    inventory = session.offline_model_inventory(root)
    assert inventory['other'] == ['esm/esm2_t6.pt']


def test_weight_calls_come_from_the_candidate_runtime(monkeypatch):
    import subprocess
    monkeypatch.setattr(session, 'RUNTIME', 'python-with-torchvision')
    seen = []
    def probe(script):
        seen.append(script)
        return subprocess.CompletedProcess([], 0, 'warning line\n{"a.pth": "torchvision.models.a(weights=\\"X\\")"}\n', '')
    monkeypatch.setattr(session, 'runtime_probe', probe)
    assert session.torchvision_weight_calls() == {'a.pth': 'torchvision.models.a(weights="X")'}
    assert 'get_model_weights' in seen[0]


def test_weight_calls_fall_back_to_file_names(monkeypatch):
    monkeypatch.setattr(session, 'RUNTIME', 'python')
    def probe(script):
        raise RuntimeError('runtime is not serving imports')
    monkeypatch.setattr(session, 'runtime_probe', probe)
    assert session.torchvision_weight_calls() == {}
    monkeypatch.setattr(session, 'RUNTIME', None)
    monkeypatch.setattr(session, 'CANDIDATE_RUNTIME', None)
    assert session.torchvision_weight_calls() == {}
