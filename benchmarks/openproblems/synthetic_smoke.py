"""Zero-model synthetic interface/negative-artifact checks; not scientific results."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.metrics import pairwise_distances
from build_package import build, TASKS

HERE = Path(__file__).resolve().parent
CORE = Path(os.environ.get('OP_CORE_SRC', str(HERE.parents[1] / 'src')))
LEGACY = Path(os.environ.get('OP_LEGACY_SCRIPTS', str(Path(__file__).resolve().parent / 'runners')))

def execute(root, workspace):
    env = {**os.environ, 'DATA_DIR': str(root / 'problem/data'), 'OUTPUT_DIR': str(workspace / 'output'),
           'NUMBA_NUM_THREADS':'2','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    subprocess.run([sys.executable, str(root / 'starter/run.py')], env=env, cwd=workspace, check=True, capture_output=True)
    return subprocess.run([sys.executable, str(root / 'evaluation/evaluator.py')], env=env, cwd=workspace, capture_output=True, text=True)

def main():
    report=[]
    with tempfile.TemporaryDirectory(prefix='op_adapter_synthetic_') as d:
        top=Path(d); rng=np.random.default_rng(0)
        n,g=512,501
        base=ad.AnnData(rng.poisson(3,(n,g)).astype(float), obs=pd.DataFrame(index=[f'c{i}' for i in range(n)]),var=pd.DataFrame(index=[f'g{i}' for i in range(g)]))
        base.uns['hidden_canary']='NOT_CANDIDATE_VISIBLE'; base.var['hidden_canary']=1
        base.layers['counts']=base.X.copy(); base.layers['normalized']=np.log1p(base.X)
        base.obs['cell_type']=[f't{i%3}' for i in range(n)]; base.obs['batch']=[f'b{i%2}' for i in range(n)]
        base.obs['label']=base.obs['cell_type'].copy(); base.obsm['spatial']=rng.normal(size=(n,2))
        base.var['true_spatial_var_score']=np.arange(g)%2
        base.uns['cell_type_names']=np.array(['t0','t1','t2'])
        base.obsm['proportions_true']=rng.dirichlet(np.ones(3),size=n)
        base.obs['is_waypoint']=np.arange(n)<20
        original=base.layers['normalized']; centroids=np.stack([original[base.obs['cell_type']==t].mean(axis=0) for t in ['t0','t1','t2']])
        for key,a,b in [('between_waypoint_distances',original[:20],original[:20]),('waypoint_centroid_distances',original[:20],centroids),('between_centroid_distances',centroids,centroids)]:
            distance=pairwise_distances(a,b);np.fill_diagonal(distance,0);base.uns[key]=distance
        source=top/'source.h5ad';base.write_h5ad(source)
        different=base.copy();different.layers['counts']=rng.poisson(3,(n,g)).astype(float); gold=top/'gold.h5ad';different.write_h5ad(gold)
        for task in TASKS:
            kinds={'denoise':['train'],'label':['train','test'],'decomp':['single_cell','spatial']}.get(task,['input'])
            spec={'task':task,'role':'optimization','metric_protocol':'released_tusoai_svg_classification' if task=='svg' else 'historical_op_dr_ab268cc_svd500' if task=='visual' else 'legacy_reconstruction_synthetic_only','metric_audit_approved':True,'claim_scope':'synthetic smoke only','datasets':[{'name':'fixture','inputs':{k:str(source) for k in kinds},'gold':str(gold if task=='denoise' else source)}]}
            root=top/task;build(spec,root,LEGACY,CORE);workspace=top/(task+'_work');workspace.mkdir()
            sanitized_files=list((root/'problem/data').glob('*.h5ad'))
            for visible_path in sanitized_files:
                visible=ad.read_h5ad(visible_path)
                assert 'hidden_canary' not in visible.uns and 'hidden_canary' not in visible.var and visible.raw is None
                assert 'true_spatial_var_score' not in visible.var and 'proportions_true' not in visible.obsm
                if task=='visual':assert 'cell_type' not in visible.obs and 'is_waypoint' not in visible.obs
            try:
                result=execute(root,workspace)
                positive=result.returncode==0
                score=json.loads((workspace/'score.json').read_text()) if positive else None
                if task=='batch' and positive:
                    (root/'starter/solution.py').write_text("def tuso_model(a):\n    a.layers['corrected_counts']=a.layers['counts'].copy()\n    return a\n")
                    alternative=execute(root,workspace)
                    assert alternative.returncode==0, alternative.stderr
                # Missing declared artifact must be rejected without silently dropping its dataset.
                expected=root/'evaluation/scoring.json'; out=json.loads(expected.read_text())['datasets'][0]['output']
                (workspace/'output'/out).unlink(missing_ok=True)
                env={**os.environ,'OUTPUT_DIR':str(workspace/'output')}
                negative=subprocess.run([sys.executable,str(root/'evaluation/evaluator.py')],cwd=workspace,env=env,capture_output=True,text=True)
                report.append({'task':task,'positive':positive,'missing_artifact_rejected':negative.returncode!=0,'hidden_field_canary_removed':True,'score':score,'stderr':result.stderr[-1500:]})
            except Exception as e:
                report.append({'task':task,'positive':False,'error':str(e)})
    (HERE/'synthetic_smoke_result.json').write_text(json.dumps({'model_calls':0,'scope':'synthetic interface only, not official scientific results','tasks':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if len(report)==6 and all(r.get('positive') and r.get('missing_artifact_rejected') and r.get('hidden_field_canary_removed') for r in report) else 1

if __name__=='__main__':raise SystemExit(main())
