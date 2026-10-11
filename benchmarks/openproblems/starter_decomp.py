import numpy as np
from scipy import sparse
from scipy.optimize import nnls

def matrix(a):
    return a.layers['counts'] if 'counts' in a.layers else a.X

def tuso_model(single_cell_adata, spatial_adata):
    genes = spatial_adata.var_names.intersection(single_cell_adata.var_names)
    sc = single_cell_adata[:, genes]
    sp = spatial_adata[:, genes]
    signatures = []
    labels = sc.obs['cell_type'].astype(str).to_numpy()
    for label in spatial_adata.uns['cell_type_names']:
        selected = matrix(sc)[labels == str(label)]
        if not selected.shape[0]:
            raise ValueError('reference cell type absent')
        signatures.append(np.asarray(selected.mean(axis=0)).ravel())
    design = np.asarray(signatures).T
    x = matrix(sp)
    prediction = []
    for i in range(x.shape[0]):
        row = x[i].toarray().ravel() if sparse.issparse(x) else x[i]
        weights = nnls(design, row)[0]
        prediction.append(weights / weights.sum() if weights.sum() else np.ones(len(weights)) / len(weights))
    spatial_adata.obsm['proportions_pred'] = np.asarray(prediction)
    return spatial_adata
