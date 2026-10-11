from sklearn.decomposition import TruncatedSVD

def tuso_model(adata):
    x = adata.layers['normalized'] if 'normalized' in adata.layers else adata.layers['counts'] if 'counts' in adata.layers else adata.X
    adata.obsm['X_emb'] = TruncatedSVD(n_components=max(1, min(20, min(x.shape) - 1)), random_state=0).fit_transform(x)
    return adata
