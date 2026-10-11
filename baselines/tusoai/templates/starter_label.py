import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import normalize

def features(a):
    x = a.layers['normalized'] if 'normalized' in a.layers else a.layers['counts'] if 'counts' in a.layers else a.X
    return normalize(x, norm='l1')

def tuso_model(train_adata, test_adata):
    model = NearestCentroid().fit(features(train_adata), train_adata.obs['label'].astype(str))
    test_adata.obs['label_pred'] = model.predict(features(test_adata))
    return test_adata
