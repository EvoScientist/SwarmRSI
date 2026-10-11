"""Controller-only scoring: never imports or executes candidate source."""
import json
import os
from pathlib import Path
import anndata as ad
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy import sparse
from openproblems_paired_metrics import (denoise_control_normalized, label_dataset_score,
    batch_components, batch_dataset_score, decomp_control_normalized, visual_dataset_score, equal_mean)

ROOT = Path(__file__).resolve().parent

def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def check_names(actual, expected, label):
    if not np.array_equal(np.asarray(actual).astype(str), np.asarray(expected).astype(str)):
        raise ValueError(label + ' identities differ')

def log_cp10k(matrix):
    x = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    x = x.astype(np.float64, copy=True)
    totals = x.sum(axis=1, keepdims=True)
    np.divide(x * 10000., totals, out=x, where=totals != 0)
    x[totals[:, 0] == 0] = 0
    return np.log1p(x)

def score(task, output, files, metric_protocol):
    gold = ad.read_h5ad(ROOT / files['gold'])
    if task == 'batch':
        value = ad.read_h5ad(output)
        check_names(value.obs_names, gold.obs_names, 'batch obs')
        source = ad.read_h5ad(ROOT / files['input'], backed='r')
        check_names(value.obs_names, source.obs_names, 'batch input obs')
        if 'corrected_counts' in value.layers:
            check_names(value.var_names, source.var_names, 'batch input var')
            corrected = value.layers['corrected_counts']
            if corrected.shape != source.shape or not np.isfinite(corrected.data if sparse.issparse(corrected) else corrected).all():
                raise ValueError('invalid corrected_counts matrix')
        source.file.close()
        graph_keys = set(value.obsp) & {'connectivities', 'distances'}
        if graph_keys and graph_keys != {'connectivities', 'distances'}:
            raise ValueError('both graph matrices required')
        if 'X_emb' not in value.obsm and 'corrected_counts' in value.layers:
            import scanpy as sc
            n_comps = min(50, min(value.layers['corrected_counts'].shape) - 1)
            value.obsm['X_emb'] = sc.pp.pca(value.layers['corrected_counts'], n_comps=n_comps, use_highly_variable=False, svd_solver='arpack', return_info=False)
        if 'X_emb' not in value.obsm:
            raise ValueError('batch output has no embedding or corrected_counts; graph-only ASW semantics unresolved')
        if 'connectivities' not in value.obsp:
            import scanpy as sc
            sc.pp.neighbors(value, use_rep='X_emb')
        embedding = value.obsm['X_emb']
        if embedding.ndim != 2 or embedding.shape[0]!=gold.n_obs or embedding.shape[1]<1 or not np.isfinite(embedding).all():
            raise ValueError('invalid batch embedding')
        for key in ('connectivities', 'distances'):
            if key in value.obsp:
                graph = value.obsp[key]
                if graph.shape != (gold.n_obs, gold.n_obs) or not np.isfinite(graph.data if sparse.issparse(graph) else graph).all():
                    raise ValueError('invalid batch graph')
        components = batch_components(value.obsm['X_emb'], gold.obs['cell_type'], gold.obs['batch'], graph=value.obsp.get('connectivities'))
        return batch_dataset_score(components), components
    value = load_npz(output)
    if task == 'denoise':
        train = ad.read_h5ad(ROOT / files['train'])
        check_names(value['obs_names'], gold.obs_names, 'denoise obs')
        check_names(value['var_names'], gold.var_names, 'denoise var')
        check_names(train.obs_names, gold.obs_names, 'denoise train/gold obs')
        check_names(train.var_names, gold.var_names, 'denoise train/gold var')
        prediction = value['denoised']
        if prediction.shape != gold.shape or train.shape != gold.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('invalid denoise prediction')
        mse = float(np.mean((log_cp10k(prediction) - log_cp10k(gold.layers['counts'])) ** 2))
        control = float(np.mean((log_cp10k(train.layers['counts']) - log_cp10k(gold.layers['counts'])) ** 2))
        return denoise_control_normalized(method_mse=mse, no_denoise_mse=control), {'mse': mse, 'no_denoise_mse': control}
    if task == 'label':
        check_names(value['obs_names'], gold.obs_names, 'label obs')
        actual, prediction = gold.obs['label'].astype(str), value['prediction'].astype(str)
        if prediction.ndim != 1 or len(prediction)!=gold.n_obs or (np.char.str_len(prediction)==0).any():
            raise ValueError('invalid label prediction')
        components = {'accuracy': float(accuracy_score(actual, prediction)), **{
            'f1_' + k: float(f1_score(actual, prediction, average=k)) for k in ('macro', 'micro', 'weighted')}}
        return label_dataset_score(components), components
    if task == 'svg':
        if metric_protocol != 'released_tusoai_svg_classification':
            raise ValueError('SVG protocol unapproved or unsupported; no metric substitution')
        from preflight_openproblems_svg import evaluate_released_tusoai
        check_names(value['var_names'], gold.var_names, 'SVG var')
        from preflight_openproblems_svg import validate_prediction_artifact
        errors = validate_prediction_artifact(value['prediction'], gold.n_vars)
        if errors:
            raise ValueError('; '.join(errors))
        predicted = gold.var['true_spatial_var_score'].copy()
        predicted.iloc[:] = value['prediction']
        components = evaluate_released_tusoai(gold.var['true_spatial_var_score'], predicted)
        components['literal_accuracy'] = float(accuracy_score(gold.var['true_spatial_var_score'].astype(int), predicted.astype(int)))
        return components['score'], components
    if task == 'decomp':
        check_names(value['obs_names'], gold.obs_names, 'decomp obs')
        if 'cell_type_names' in gold.uns:
            check_names(value['cell_type_names'], gold.uns['cell_type_names'], 'decomp cell types')
        prediction = value['prediction']
        truth = np.asarray(gold.obsm['proportions_true'], dtype=float)
        if prediction.shape != truth.shape or not np.isfinite(prediction).all() or (prediction < -1e-12).any() or not np.allclose(prediction.sum(axis=1), 1., atol=1e-6, rtol=1e-6):
            raise ValueError('invalid decomposition prediction')
        method = float(r2_score(truth, prediction, multioutput='uniform_average'))
        reference = ad.read_h5ad(ROOT / files['reference'])
        labels = reference.obs['cell_type'].astype(str)
        counts = labels.value_counts().reindex(value['cell_type_names'].astype(str)).to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts <= 0).any():
            raise ValueError('random reference control has missing cell types')
        control_prediction = np.random.RandomState(42).dirichlet(counts, size=truth.shape[0])
        control = float(r2_score(truth, control_prediction, multioutput='uniform_average'))
        return decomp_control_normalized(method_r2=method, random_control_r2=control), {'r2': method, 'random_control_r2': control}
    if task == 'visual':
        from visual_historical_components import components as historical_components, aggregate as historical_aggregate
        check_names(value['obs_names'], gold.obs_names, 'visual obs')
        embedding = value['embedding']
        if embedding.ndim != 2 or embedding.shape != (gold.n_obs, 2) or not np.isfinite(embedding).all():
            raise ValueError('invalid visualization embedding: expected exactly two dimensions')
        if metric_protocol != 'historical_op_dr_ab268cc_svd500':
            raise ValueError('unapproved Visual metric protocol')
        components = historical_components(embedding, gold)
        return historical_aggregate(components), components
    raise ValueError('unknown task')

def main():
    spec = json.loads((ROOT / 'scoring.json').read_text())
    output = Path(os.environ['OUTPUT_DIR'])
    expected = {r['output'] for r in spec['datasets']}
    found = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if found != expected or any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('output set differs from complete frozen dataset denominator')
    rows = {}
    for row in spec['datasets']:
        number, components = score(spec['task'], output / row['output'], row['files'], spec['metric_protocol'])
        if not np.isfinite(number):
            raise ValueError('nonfinite score')
        rows[row['name']] = {'score': number, 'components': components}
    Path('score.json').write_text(json.dumps({'score': equal_mean((r['score'] for r in rows.values()), name=spec['task']),
        'per_dataset': rows, 'metric_protocol': spec['metric_protocol']}, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
