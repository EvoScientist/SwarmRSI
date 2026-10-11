"""Build external OP packages for unmodified fd98216 run-naturebench.

Input manifest fields: task, role, metric_protocol, metric_audit_approved (must true),
datasets [{name, inputs {input|train|test|single_cell|spatial: source path}, gold: source path}].
Only approved public H5AD fields survive; no deployment assets belong in optimization packages.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

TASKS = ('denoise', 'label', 'batch', 'svg', 'decomp', 'visual')
ADAPTER = Path(__file__).resolve().parent


def sanitized(source, target, task, kind):
    import anndata as ad
    value = ad.read_h5ad(source)
    if task == 'svg':
        value.X = value.layers['counts'].copy()
    obs_keys = {'label': {'train': ['label', 'batch'], 'test': ['batch']},
                'batch': {'input': ['cell_type', 'batch']},
                'svg': {'input': ['annotation','leiden','time_point','n_genes_by_counts','log1p_n_genes_by_counts','total_counts','log1p_total_counts','n_genes','size_factors']},
                'decomp': {'single_cell': ['cell_type'], 'spatial': []}}.get(task, {}).get(kind, [])
    value.obs = value.obs[[k for k in obs_keys if k in value.obs]].copy()
    var_keys = ['feature_id','feature_name','orig_feature_name'] if task == 'svg' else []
    value.var = value.var[[k for k in var_keys if k in value.var]].copy()
    value.raw = None
    for key in list(value.layers):
        if key not in {None, 'counts', 'normalized'}:
            del value.layers[key]
    for key in list(value.obsm):
        if key not in {'spatial', 'X_spatial'}:
            del value.obsm[key]
    value.varm.clear(); value.varp.clear(); value.obsp.clear()
    keep_uns = {'cell_type_names'} if task == 'decomp' and kind == 'spatial' else set()
    value.uns = {k: value.uns[k] for k in keep_uns if k in value.uns}
    value.write_h5ad(target, compression='gzip')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def build(spec, destination, legacy_scripts, core_src):
    if spec['task'] not in TASKS or spec['role'] not in {'optimization', 'deployment'}:
        raise ValueError('unknown task or role')
    if spec.get('metric_audit_approved') is not True:
        raise ValueError('metric reconstruction requires independent approval before package release')
    task = spec['task']
    if destination.exists():
        raise ValueError('destination exists; immutable package cannot be overwritten')
    names = [r['name'] for r in spec['datasets']]
    if not names or len(set(names)) != len(names) or any(Path(n).name != n or n in {'.', '..'} for n in names):
        raise ValueError('dataset names invalid')
    # Validate all source paths before materializing anything.
    for row in spec['datasets']:
        for value in [*row['inputs'].values(), row['gold']]:
            if not Path(value).is_file():
                raise ValueError('missing source: ' + str(value))
    for sub in ('starter', 'problem/data', 'evaluation'):
        (destination / sub).mkdir(parents=True)
    data, evaluation = destination / 'problem/data', destination / 'evaluation'
    shutil.copyfile(ADAPTER / 'candidate_run.py', destination / 'starter/run.py')
    shutil.copyfile(ADAPTER / ('starter_' + task + '.py'), destination / 'starter/solution.py')
    shutil.copyfile(ADAPTER / 'output_evaluator.py', evaluation / 'evaluator.py')
    shutil.copyfile(ADAPTER / 'visual_historical_components.py', evaluation / 'visual_historical_components.py')
    shutil.copyfile(legacy_scripts / ('openproblems_' + task + '_container_runner.py'), data / 'runner.py')
    if task == 'denoise':
        runner = (data / 'runner.py').read_text().replace('np.random.seed(42)', 'np.random.seed(0)').replace('random.seed(42)', 'random.seed(0)')
        runner = runner.replace("    prediction = _dense(result.obsm[\"denoised\"])", "    if not result.obs_names.equals(source.obs_names) or not result.var_names.equals(source.var_names):\n        raise RuntimeError('denoise changed cell or gene identities/order')\n    prediction = _dense(result.obsm[\"denoised\"])")
        runner = runner.replace("np.savez_compressed(output, denoised=prediction)", "np.savez_compressed(output, denoised=prediction, obs_names=np.asarray(source.obs_names.astype(str), dtype=str), var_names=np.asarray(source.var_names.astype(str), dtype=str))")
        (data / 'runner.py').write_text(runner)
    for name in ('openproblems_paired_metrics.py', 'evaluate_openproblems_visual_workspace.py', 'preflight_openproblems_svg.py'):
        shutil.copyfile(legacy_scripts / name, evaluation / name)
    metric_path = evaluation / 'openproblems_paired_metrics.py'
    metric_text = metric_path.read_text()
    original_graph_call = 'connected_components(subgraph, directed=False)'
    if metric_text.count(original_graph_call) != 1:
        raise ValueError('unexpected legacy graph-connectivity source; audit required')
    metric_path.write_text(metric_text.replace(original_graph_call, "connected_components(subgraph, directed=True, connection='strong')"))
    public_rows, private_rows, sources = [], [], []
    for row in spec['datasets']:
        name = row['name']; arguments = ['candidate'] if task == 'denoise' else []
        for kind, source in row['inputs'].items():
            target = name + '__' + kind + '.h5ad'
            sanitized(Path(source), data / target, task, kind)
            arguments += ['--' + kind.replace('_', '-'), '{DATA}/' + target]
            sources.append({'path': str(Path(source).resolve()), 'sha256': digest(Path(source)), 'role': kind, 'dataset': name})
        if task != 'denoise':
            arguments += ['--seed', '0']
        output = name + ('.h5ad' if task == 'batch' else '.npz')
        arguments += ['--output', '{OUTPUT}/' + output]
        public_rows.append({'name': name, 'arguments': arguments, 'output': output})
        gold = name + '__gold.h5ad'
        shutil.copyfile(row['gold'], evaluation / gold)
        files = {'gold': gold}
        if task == 'batch':
            files['input'] = '../problem/data/' + name + '__input.h5ad'
        if task == 'decomp':
            reference = name + '__reference.h5ad'
            shutil.copyfile(row['inputs']['single_cell'], evaluation / reference)
            files['reference'] = reference
        if task == 'denoise':
            train = name + '__train.h5ad'
            shutil.copyfile(row['inputs']['train'], evaluation / train)
            files['train'] = train
        private_rows.append({'name': name, 'output': output, 'files': files})
        sources.append({'path': str(Path(row['gold']).resolve()), 'sha256': digest(Path(row['gold'])), 'role': 'gold', 'dataset': name})
    (data / 'inputs.json').write_text(json.dumps({'task': task, 'seed': 0, 'datasets': public_rows}, indent=2) + '\n')
    (evaluation / 'scoring.json').write_text(json.dumps({'task': task, 'role': spec['role'], 'metric_protocol': spec['metric_protocol'], 'datasets': private_rows}, indent=2) + '\n')
    metric = {'name': 'score', 'direction': 'maximize', 'role': 'public_validation'}
    metadata = {'task_id': destination.name, 'benchmark': 'OpenProblems', 'role': spec['role'],
                'search_metric': metric, 'search_target': None,
                'official_test_visible_to_candidate': False, 'source_records': sources,
                'metric_protocol': spec['metric_protocol'], 'metric_audit_approved': True,
                'metric_audit_reference': spec.get('metric_audit_reference'), 'paper_equivalence_verified': spec.get('paper_equivalence_verified', False),
                'claim_scope': spec.get('claim_scope', 'reconstructed protocol; paper equivalence unverified'),
                'resources': {'wall_seconds': 10800, 'cpus': 4, 'memory_gib': 50}, 'seed': 0}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from task_readmes import readme
    (destination / 'problem/README.md').write_text(readme(task, names))
    sys.path.insert(0, str(core_src))
    from evolvcoder.panel_tasks import seal, verify_assets
    visible = [str(p.relative_to(destination)) for sub in ('starter', 'problem') for p in (destination / sub).rglob('*') if p.is_file()]
    hidden = [str(p.relative_to(destination)) for p in evaluation.rglob('*') if p.is_file()]
    seal(destination, task_id=destination.name, metric=metric, candidate_visible=visible, controller_only=hidden)
    verify_assets(destination / 'custody.json')
    return metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--legacy-scripts', type=Path, default=ADAPTER / 'runners')
    p.add_argument('--core-src', type=Path, default=ADAPTER.parents[1] / 'src')
    a = p.parse_args()
    result = build(json.loads(a.manifest.read_text()), a.destination, a.legacy_scripts, a.core_src)
    print(json.dumps({'task': result['task_id'], 'model_calls': 0, 'package_built': True, 'runtime_smoke_passed': False}))

if __name__ == '__main__':
    main()
