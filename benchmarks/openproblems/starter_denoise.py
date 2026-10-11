def tuso_model(adata):
    adata.obsm['denoised'] = adata.obsm['train'].copy()
    return adata
