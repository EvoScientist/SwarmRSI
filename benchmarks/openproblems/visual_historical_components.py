"""Source-bound historical OpenProblems DR components, no label references.

Official commit ab268cc0d6781cee7140a73b3943761104719ad8:
Trustworthiness(k15), distance_correlation(SVD500 + pdist Spearman), density(k30).
Only SVD's unspecified RNG is pinned to algorithm seed 0. Density retains 42.
The equal-weight normalized aggregate is an explicitly declared reconstruction.
"""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import trustworthiness
from evaluate_openproblems_visual_workspace import _calculate_radii


def components(embedding, gold):
    if 'normalized' not in gold.layers:
        raise ValueError('historical Visual requires normalized expression')
    original=gold.layers['normalized']
    if min(original.shape)<500:
        raise ValueError('historical Visual SVD500 requires at least 500 cells and features')
    if embedding.shape!=(gold.n_obs,2) or not np.isfinite(embedding).all():
        raise ValueError('Visual requires finite exactly-2D embedding')
    trust=float(trustworthiness(original,embedding,n_neighbors=15,metric='euclidean'))
    reduced=TruncatedSVD(n_components=500,random_state=0).fit_transform(original)
    high=pdist(reduced); low=pdist(embedding)
    distance=float(spearmanr(low,high).correlation)
    del high,low,reduced
    original_radii=_calculate_radii(original,30,42)
    embedded_radii=_calculate_radii(embedding,30,42)
    density=float(pearsonr(original_radii,embedded_radii)[0])
    result={'trustworthiness':trust,'distance_correlation':distance,'density_preservation':density}
    if not all(np.isfinite(v) for v in result.values()):raise ValueError('nonfinite historical Visual metric')
    return result


def aggregate(values):
    return float(np.mean([values['trustworthiness'],(values['distance_correlation']+1)/2,(values['density_preservation']+1)/2]))
