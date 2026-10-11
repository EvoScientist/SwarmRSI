import json,tempfile,time
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse
import visual_historical_components as m
import visual_reference_cache as c
rng=np.random.default_rng(0)
with tempfile.TemporaryDirectory() as td:
 root=Path(td);gold=ad.AnnData(X=sparse.csr_matrix(rng.poisson(.3,size=(512,501)).astype(np.float32)));gold.layers['normalized']=gold.X.copy();p=root/'gold.h5ad';gold.write_h5ad(p)
 c.create(root/'cache',p,Path(m.__file__))
 rows=[]
 for case in ('random','tied'):
  emb=rng.normal(size=(512,2))
  if case=='tied':emb[1::2]=emb[::2]
  direct=m.components(emb,gold);cached=c.components(emb,gold,root/'cache',p,Path(m.__file__))
  assert direct==cached,(direct,cached)
  rows.append({'case':case,'exact_equal':True,'components':direct})
 # Same shape but changed source must be refused.
 with p.open('ab') as f:f.write(b'changed')
 try:c.components(emb,gold,root/'cache',p,Path(m.__file__))
 except ValueError:changed_source_rejected=True
 else:raise AssertionError('changed source accepted')
 print(json.dumps({'model_calls':0,'cases':rows,'changed_source_rejected':changed_source_rejected},indent=2))
