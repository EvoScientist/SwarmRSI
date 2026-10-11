import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import json
import types
import numpy as np
import pandas as pd
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location('paired_pages', Path(__file__).parents[1] / 'deploy/mle/paired_pages.py')
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


def fixture(tmp_path):
    root = tmp_path / 'prepared'
    for directory in ['train', 'train_cleaned', 'test']:
        (root / 'public' / directory).mkdir(parents=True)
    for index in range(10):
        for directory, offset in [('train', 5), ('train_cleaned', 0)]:
            pixels = np.full((2 + index, 3 + index), index + offset, dtype=np.uint8)
            Image.fromarray(pixels).save(root / 'public' / directory / f'page_{index}.png')
    Image.fromarray(np.array([[3,4,5],[6,7,8]], dtype=np.uint8)).save(root / 'public/test/terminal.png')
    sample = pages.image_frame(root / 'public/test/terminal.png')
    sample['value'] = 1
    sample.to_csv(root / 'public/sampleSubmission.csv', index=False)
    return root


def test_split_bytes_visibility_and_repeat(tmp_path):
    source = fixture(tmp_path)
    out = tmp_path / 'out'
    split = pages.carve_pages(source, out, seed=0, holdout=0.2)
    custody = json.loads((out / 'evaluation/page_split.json').read_text())
    assert len(custody['fit_pages']) == 8 and len(custody['holdout_pages']) == 2
    assert set(custody['fit_pages']).isdisjoint(custody['holdout_pages'])
    for name in custody['fit_pages']:
        for folder in ('train','train_cleaned'):
            assert (out / 'problem/data' / folder / name).read_bytes() == (source / 'public' / folder / name).read_bytes()
    for name in custody['holdout_pages']:
        assert not (out / 'problem/data/train_cleaned' / name).exists()
        assert (out / 'problem/data/test' / name).read_bytes() == (source / 'public/train' / name).read_bytes()
    assert pages.carve_pages(source, tmp_path / 'again', seed=0, holdout=0.2) == split
    assert not (out / 'problem/data/answers.csv').exists()


def test_clean_template_duplicates_grouped_without_public_hashes(tmp_path):
    source = fixture(tmp_path)
    # Two distinct noisy observations of the same clean template, different encoding.
    with Image.open(source / 'public/train_cleaned/page_0.png') as image:
        image.save(source / 'public/train_cleaned/page_1.png', compress_level=0)
    Image.fromarray(np.full((2,3), 40, dtype=np.uint8)).save(source / 'public/train/page_1.png')
    out = tmp_path / 'out'
    metadata = pages.carve_pages(source, out, seed=0, holdout=0.2)
    audit = json.loads((out / 'evaluation/page_split.json').read_text())
    groups = audit['clean_template_by_page']
    assert groups['page_0.png'] == groups['page_1.png']
    assert ('page_0.png' in audit['fit_pages']) == ('page_1.png' in audit['fit_pages'])
    assert metadata['fit_group_count'] + metadata['holdout_group_count'] == 9
    assert metadata['fit_rows'] + metadata['holdout_rows'] == 10
    assert metadata['holdout_pixel_count'] == len(pd.read_csv(out / 'evaluation/answers.csv'))
    assert 'clean_template_by_page' not in metadata and 'fit_pages' not in metadata
    public_bytes = b''.join(p.read_bytes() for p in (out / 'problem').rglob('*') if p.is_file())
    for digest in groups.values():
        assert digest.encode() not in public_bytes
        assert digest not in json.dumps(metadata)
    again = tmp_path / 'again'
    assert pages.carve_pages(source, again, seed=0, holdout=0.2) == metadata
    assert (again / 'evaluation/page_split.json').read_bytes() == (out / 'evaluation/page_split.json').read_bytes()


def test_pixel_contract_and_order(tmp_path):
    image = tmp_path / 'a_b.png'
    Image.fromarray(np.array([[0,255,128],[64,32,16]], dtype=np.uint8)).save(image)
    expected = pages.image_frame(image)
    assert expected.id.tolist() == ['a_b_1_1','a_b_1_2','a_b_1_3','a_b_2_1','a_b_2_2','a_b_2_3']
    assert expected.value.tolist() == [0,1,128/255,64/255,32/255,16/255]
    assert pages.validate_pixels(expected.iloc[::-1], expected).id.tolist() == expected.id.tolist()


def test_decoded_duplicate_with_distinct_png_encoding(tmp_path):
    source = fixture(tmp_path)
    # Match both pair dimensions so rejection cannot be attributed to bad pairing.
    for directory in ('train', 'train_cleaned'):
        with Image.open(source / 'public' / directory / 'page_0.png') as image:
            image.save(source / 'public' / directory / 'page_1.png', compress_level=0)
    with pytest.raises(ValueError, match='duplicate decoded page content'):
        pages.carve_pages(source, tmp_path / 'out', seed=0, holdout=0.2)


@pytest.mark.parametrize('bad', ['pair', 'dimensions', 'duplicate', 'decoded_duplicate', 'symlink'])
def test_bad_pairs_failclosed(tmp_path, bad):
    source = fixture(tmp_path)
    path = source / 'public/train_cleaned/page_0.png'
    if bad == 'pair':
        path.unlink()
    elif bad == 'dimensions':
        Image.fromarray(np.zeros((3,3), dtype=np.uint8)).save(path)
    elif bad in ('duplicate', 'decoded_duplicate'):
        other = source / 'public/train_cleaned/page_1.png'
        if bad == 'duplicate':
            path.write_bytes(other.read_bytes())
        else:
            with Image.open(other) as im:
                im.save(path, compress_level=0)
    else:
        path.unlink()
        path.symlink_to(source / 'public/train_cleaned/page_1.png')
    with pytest.raises(ValueError):
        pages.carve_pages(source, tmp_path / 'out', seed=0, holdout=0.2)


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'extra', 'nan', 'inf', 'negative', 'large'])
def test_bad_pixels(tmp_path, bad):
    expected = pd.DataFrame({'id':['a_1_1','a_1_2'], 'value':[0.1,0.2]})
    submission = expected.copy()
    if bad == 'duplicate': submission.loc[1,'id'] = 'a_1_1'
    elif bad == 'missing': submission = submission.iloc[:1]
    elif bad == 'extra': submission.loc[1,'id'] = 'other'
    else: submission.loc[0,'value'] = {'nan':np.nan,'inf':np.inf,'negative':-0.1,'large':1.1}[bad]
    with pytest.raises(ValueError): pages.validate_pixels(submission, expected)


def test_identical_starter_carved_and_terminal(tmp_path):
    source = fixture(tmp_path)
    pages.carve_pages(source, tmp_path / 'out', seed=0, holdout=0.2)
    for index, data in enumerate([tmp_path / 'out/problem/data', source / 'public']):
        output = tmp_path / f'predictions{index}'
        subprocess.run([sys.executable,'-c',pages.STARTER], env={**os.environ,'DATA_DIR':str(data),'OUTPUT_DIR':str(output)}, check=True)
        submission = pd.read_csv(output / 'submission.csv')
        sample = pd.read_csv(data / 'sampleSubmission.csv')
        pages.validate_pixels(submission, sample)
        expected = pd.concat([pages.image_frame(p) for p in sorted((data / 'test').glob('*.png'))], ignore_index=True)
        aligned = pages.validate_pixels(submission, expected)
        np.testing.assert_allclose(aligned.value, expected.value)


def test_builder_main_seals_package_and_emits_summary(tmp_path, monkeypatch, capsys):
    """Exercise real main through final printing; only external registry/score stubbed."""
    prepared = fixture(tmp_path)
    (prepared / 'private').mkdir()
    (prepared / 'public/description.md').write_text('Synthetic paired image competition')
    official = tmp_path / 'official'
    (official / 'mlebench').mkdir(parents=True)
    (official / 'UPSTREAM_REVISION').write_text('frozen-synthetic-revision')
    (official / 'mlebench/__init__.py').write_text('')
    grader = types.SimpleNamespace(name='root-mean-squared-error',
        is_lower_better=lambda _: True,
        rank_score=lambda *_: {'gold_threshold': 0.1, 'silver_threshold': 0.2, 'bronze_threshold': 0.3})
    registry = types.SimpleNamespace(set_data_dir=lambda _: types.SimpleNamespace(
        get_competition=lambda _: types.SimpleNamespace(grader=grader)))
    monkeypatch.setitem(sys.modules, 'mlebench', types.ModuleType('mlebench'))
    monkeypatch.setitem(sys.modules, 'mlebench.grade', types.SimpleNamespace(get_leaderboard=lambda _: [1,2]))
    monkeypatch.setitem(sys.modules, 'mlebench.registry', types.SimpleNamespace(registry=registry))
    monkeypatch.setitem(sys.modules, 'paired_pages', pages)
    builder_spec = importlib.util.spec_from_file_location('page_builder_main', Path(__file__).parents[1] / 'deploy/mle/make_task_root.py')
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    def frozen_starter_smoke(python, root, metric):
        output = tmp_path / 'main-starter-output'
        subprocess.run([python, str(root / 'starter/run.py')], check=True,
            env={**os.environ, 'DATA_DIR':str(root / 'problem/data'), 'OUTPUT_DIR':str(output)})
        pages.validate_pixels(pd.read_csv(output / 'submission.csv'), pd.read_csv(root / 'evaluation/answers.csv'))
        return 0.9  # Explicit synthetic score stub, never scientific evidence.
    monkeypatch.setattr(builder, 'run_starter', frozen_starter_smoke)
    tasks = tmp_path / 'tasks'
    monkeypatch.setattr(sys, 'argv', ['make_task_root.py', 'denoising-dirty-documents',
        '--prepared', str(prepared), '--tasks-root', str(tasks), '--cache', str(tmp_path / 'cache'),
        '--official-source', str(official), '--python', sys.executable, '--seed', '0', '--holdout', '0.2'])
    assert builder.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['split'] == {'fit_rows':8, 'holdout_rows':2, 'stratified_on':None}
    root = tasks / 'denoising-dirty-documents'
    assert builder.verify_assets(root / 'custody.json')['task'] == 'denoising-dirty-documents'
    assert (root / 'evaluation/paired_pages.py').is_file()
    assert 'controller-only' in (root / 'problem/README.md').read_text()
