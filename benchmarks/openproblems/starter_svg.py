import numpy as np
from scipy import sparse

def tuso_model(adata):
    x = adata.layers['counts'] if 'counts' in adata.layers else adata.X
    if sparse.issparse(x):
        variance = np.asarray(x.multiply(x).mean(axis=0)).ravel() - np.asarray(x.mean(axis=0)).ravel() ** 2
    else:
        variance = np.var(x, axis=0)
    adata.var['pred_spatial_var_score'] = (variance >= np.median(variance)).astype(int)
    return adata
