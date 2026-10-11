"""Official strong-component regression against two asymmetric graph cases."""
from pathlib import Path
import importlib.util
import numpy as np
from scipy import sparse
import os,tempfile

legacy=Path(os.environ.get('OP_LEGACY_SCRIPTS',str(Path(__file__).resolve().parent / 'runners')))/'openproblems_paired_metrics.py'
s=legacy.read_text();assert s.count('connected_components(subgraph, directed=False)')==1
s=s.replace('connected_components(subgraph, directed=False)',"connected_components(subgraph, directed=True, connection='strong')")
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'metrics.py';p.write_text(s);spec=importlib.util.spec_from_file_location('metrics',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 chain=sparse.csr_matrix(np.array([[0,1,0],[0,0,1],[0,0,0]]))
 assert m.graph_connectivity_score(chain,['a','a','a'])==1/3
 partial=sparse.csr_matrix(np.array([[0,1,0,0],[1,0,1,0],[0,0,0,1],[0,0,0,0]]))
 assert m.graph_connectivity_score(partial,['a']*4)==0.5
 assert m.graph_connectivity_score(chain.maximum(chain.T),['a']*3)==1.
print('3 graph cases PASS: asymmetric chain=1/3, partial strong component=1/2, symmetric connected=1')
