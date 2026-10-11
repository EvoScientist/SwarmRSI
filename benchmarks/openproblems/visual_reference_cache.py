"""Prototype exact controller-only historical Visual reference cache.

Never a candidate interface. No candidate-dependent quantities are cached.
The uncached sealed helper remains the scientific definition.
"""
import hashlib, importlib.metadata, json
from pathlib import Path
import numpy as np
from sklearn.metrics import pairwise_distances

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def binding(gold_path,helper_path,gold):
 import evaluate_openproblems_visual_workspace as legacy
 return {'schema':'visual-exact-reference-v1','gold_sha256':sha(gold_path),
  'helper_sha256':sha(helper_path),'density_helper_sha256':sha(legacy.__file__),
  'obs_names_sha256':hashlib.sha256(json.dumps(list(gold.obs_names.astype(str)),ensure_ascii=True).encode()).hexdigest(),
  'shape':list(gold.shape),'svd_components':500,'svd_seed':0,'density_neighbors':30,'density_seed':42,
  'versions':{k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-learn','umap-learn','pynndescent','numba','anndata')}}

def create(directory,gold_path,helper_path):
 import anndata as ad
 import visual_historical_components as m
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
 gold=ad.read_h5ad(gold_path);original=gold.layers['normalized']
 if min(original.shape)<500:raise ValueError('SVD500 requires at least500cells/features')
 meta=binding(gold_path,helper_path,gold)
 trust_distances=pairwise_distances(original,metric='euclidean')
 np.save(directory/'trust_distances.npy',trust_distances,allow_pickle=False)
 del trust_distances
 high=m.pdist(m.TruncatedSVD(n_components=500,random_state=0).fit_transform(original))
 np.save(directory/'high_distances.npy',high,allow_pickle=False)
 del high
 radii=m._calculate_radii(original,30,42)
 np.save(directory/'original_radii.npy',radii,allow_pickle=False)
 meta['files']={name:sha(directory/name) for name in ('trust_distances.npy','high_distances.npy','original_radii.npy')}
 (directory/'reference.json').write_text(json.dumps(meta,indent=2)+'\n')
 return meta

def components(embedding,gold,directory,gold_path,helper_path):
 import visual_historical_components as m
 directory=Path(directory);meta=json.loads((directory/'reference.json').read_text())
 if set(meta.get('files',{}))!={'trust_distances.npy','high_distances.npy','original_radii.npy'}:raise ValueError('reference cache file set mismatch')
 if sha(m.__file__)!=sha(helper_path):raise ValueError('imported historical helper differs from bound implementation')
 expected=binding(gold_path,helper_path,gold)
 if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('reference cache source/runtime binding mismatch')
 if any(sha(directory/k)!=v for k,v in meta['files'].items()):raise ValueError('reference cache bytes drift')
 if embedding.shape!=(gold.n_obs,2) or not np.isfinite(embedding).all():raise ValueError('invalid2Dembedding')
 high=np.load(directory/'high_distances.npy',allow_pickle=False,mmap_mode='r')
 original_radii=np.load(directory/'original_radii.npy',allow_pickle=False)
 if high.shape!=(gold.n_obs*(gold.n_obs-1)//2,) or original_radii.shape!=(gold.n_obs,):raise ValueError('reference shape mismatch')
 trust_distances=np.load(directory/'trust_distances.npy',allow_pickle=False)
 if trust_distances.shape!=(gold.n_obs,gold.n_obs):raise ValueError('trust reference shape mismatch')
 trust=float(m.trustworthiness(trust_distances,embedding,n_neighbors=15,metric='precomputed'))
 del trust_distances
 distance=float(m.spearmanr(m.pdist(embedding),high).correlation)
 embedded_radii=m._calculate_radii(embedding,30,42)
 density=float(m.pearsonr(original_radii,embedded_radii)[0])
 result={'trustworthiness':trust,'distance_correlation':distance,'density_preservation':density}
 if not all(np.isfinite(v) for v in result.values()):raise ValueError('nonfinite metric')
 return result
