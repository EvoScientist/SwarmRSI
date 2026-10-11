import json
import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
rng=np.random.default_rng(0);rows=[]
for dtype in (np.float32,np.float64):
 for kind in ('dense','sparse','tied'):
  x=rng.normal(size=(100,80)).astype(dtype)
  if kind=='sparse':x=sparse.csr_matrix(x*(rng.random(x.shape)<.15))
  if kind=='tied':x[1::2]=x[::2]
  distances=pairwise_distances(x,metric='euclidean')
  for embedding_kind in ('random','tied'):
   emb=rng.normal(size=(100,2))
   if embedding_kind=='tied':emb[1::2]=emb[::2]
   direct=trustworthiness(x,emb,n_neighbors=15,metric='euclidean')
   cached=trustworthiness(distances.copy(),emb,n_neighbors=15,metric='precomputed')
   assert direct==cached,(kind,dtype,embedding_kind,direct,cached)
   rows.append({'dtype':str(dtype),'input':kind,'embedding':embedding_kind,'exact_equal':True})
print(json.dumps({'model_calls':0,'cases':rows},indent=2))
