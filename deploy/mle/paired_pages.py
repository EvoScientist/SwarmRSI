"""Denoising page contract; no search logic or official-test feedback."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


def image_frame(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert('L'), dtype=float) / 255.0
    height, width = values.shape
    ids = [f'{path.stem}_{row+1}_{column+1}' for row in range(height) for column in range(width)]
    return pd.DataFrame({'id': ids, 'value': values.ravel()})


def validate_pixels(submission, answers):
    for frame in (submission, answers):
        if list(frame.columns) != ['id', 'value'] or frame.empty:
            raise ValueError('expected nonempty id,value pixel table')
        if frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError('missing or duplicate pixel IDs')
        values = pd.to_numeric(frame.value, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('pixel values must be finite and in [0,1]')
    if len(submission) != len(answers) or set(submission.id) != set(answers.id):
        raise ValueError('pixel ID set mismatch')
    return submission.set_index('id').loc[answers.id].reset_index()


def carve_pages(prepared, destination, *, seed, holdout):
    public = prepared / 'public'
    inventories = {}
    for name in ('train', 'train_cleaned'):
        directory = public / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('ordinary page directory required')
        pages = list(directory.iterdir())
        if not pages or any(p.is_symlink() or not p.is_file() or p.suffix.lower() != '.png' for p in pages):
            raise ValueError('ordinary PNG pages only')
        inventories[name] = {p.name: p for p in pages}
        digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in pages]
        if name == 'train' and len(set(digests)) != len(digests):
            raise ValueError('duplicate page content requires grouped protocol')
    noisy, cleaned = inventories['train'], inventories['train_cleaned']
    if set(noisy) != set(cleaned):
        raise ValueError('unpaired page names')
    seen = {'noisy': set(), 'cleaned': set()}
    groups, pixels = {}, {}
    for name in noisy:
        with Image.open(noisy[name]) as a, Image.open(cleaned[name]) as b:
            if a.size != b.size:
                raise ValueError('paired page dimensions differ')
            for key, image in [('noisy', a), ('cleaned', b)]:
                digest = hashlib.sha256(str(image.size).encode() + image.convert('L').tobytes()).hexdigest()
                if key == 'noisy' and digest in seen[key]:
                    raise ValueError('duplicate decoded page content requires grouped protocol')
                seen[key].add(digest)
                if key == 'cleaned':
                    groups[name] = digest
            pixels[name] = a.width * a.height
    names = sorted(noisy)
    fit_indices, held_indices = next(GroupShuffleSplit(n_splits=1, test_size=holdout,
        random_state=seed).split(names, groups=[groups[name] for name in names]))
    fit = sorted(names[index] for index in fit_indices)
    held = sorted(names[index] for index in held_indices)
    fit_groups, held_groups = {groups[name] for name in fit}, {groups[name] for name in held}
    if fit_groups & held_groups:
        raise ValueError('clean template group crossed split')
    data = destination / 'problem/data'
    evaluation = destination / 'evaluation'
    for folder in (data / 'train', data / 'train_cleaned', data / 'test', evaluation / 'held_cleaned'):
        folder.mkdir(parents=True, exist_ok=False)
    for name in fit:
        shutil.copyfile(noisy[name], data / 'train' / name)
        shutil.copyfile(cleaned[name], data / 'train_cleaned' / name)
    for name in held:
        shutil.copyfile(noisy[name], data / 'test' / name)
        shutil.copyfile(cleaned[name], evaluation / 'held_cleaned' / name)
    sample = pd.concat([image_frame(noisy[name]) for name in held], ignore_index=True)
    sample['value'] = 1.0
    answers = pd.concat([image_frame(cleaned[name]) for name in held], ignore_index=True)
    validate_pixels(sample, answers)
    sample.to_csv(data / 'sampleSubmission.csv', index=False)
    answers.to_csv(evaluation / 'answers.csv', index=False)
    split = {'algorithm': 'GroupShuffleSplit', 'seed': seed, 'holdout': holdout,
             'fraction_unit': 'clean_template_groups',
             'stratified_on': None, 'grouped_on': 'clean_template', 'label_column': None,
             'fit_rows': len(fit), 'holdout_rows': len(held),
             'fit_group_count': len(fit_groups), 'holdout_group_count': len(held_groups),
             'fit_pixel_count': sum(pixels[name] for name in fit),
             'holdout_pixel_count': sum(pixels[name] for name in held),
             'id_column': 'id', 'target_columns': ['value'], 'paired_pages': True,
             'filenames': {'train': None, 'test': None, 'sample': 'sampleSubmission.csv'}}
    (evaluation / 'page_split.json').write_text(json.dumps({**split, 'fit_pages': fit,
        'holdout_pages': held, 'clean_template_by_page': groups}, sort_keys=True))
    return split


STARTER = '''import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
def main():
    data = Path(os.environ['DATA_DIR'])
    sample = pd.read_csv(data / 'sampleSubmission.csv')
    images = {}
    result = []
    for identifier in sample.id:
        stem, row, column = identifier.rsplit('_', 2)
        if stem not in images:
            with Image.open(data / 'test' / (stem + '.png')) as image:
                images[stem] = np.asarray(image.convert('L'), dtype=float) / 255.0
        result.append(images[stem][int(row)-1, int(column)-1])
    sample['value'] = result
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / 'submission.csv', index=False)
if __name__ == '__main__':
    main()
'''
