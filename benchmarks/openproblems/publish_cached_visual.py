"""Create a separate sealed Visual package with exact controller-only references."""
import argparse,json,shutil,sys
from pathlib import Path
from visual_reference_cache import sha
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--core-src',type=Path,required=True);a=p.parse_args()
if a.destination.exists():raise ValueError('destination immutable and already exists')
sys.path.insert(0,str(a.core_src));from swarmrsi.panel_tasks import seal,verify_assets
verify_assets(a.source/'custody.json')
# Full copy: future writes can never affect live uncached package bytes.
shutil.copytree(a.source,a.destination)
e=a.destination/'evaluation';spec=json.loads((e/'scoring.json').read_text())
for row in spec['datasets']:
 source=a.references if len(spec['datasets'])==1 and (a.references/'reference.json').is_file() else a.references/row['name'];meta=json.loads((source/'reference.json').read_text())
 if meta['gold_sha256']!=sha(e/row['files']['gold']):raise ValueError('cache gold source differs')
 dest=e/'references'/row['name'];shutil.copytree(source,dest)
 row['files']['reference_cache']=str(dest.relative_to(e))
(e/'scoring.json').write_text(json.dumps(spec,indent=2)+'\n')
shutil.copy2(Path(__file__).with_name('visual_reference_cache.py'),e/'visual_reference_cache.py')
p=e/'evaluator.py';text=p.read_text();old='components = historical_components(embedding, gold)'
if text.count(old)!=1:raise ValueError('unexpected source scorer')
text=text.replace(old,"from visual_reference_cache import components as cached_components\n        components = cached_components(embedding, gold, ROOT / files['reference_cache'], ROOT / files['gold'], ROOT / 'visual_historical_components.py')")
p.write_text(text)
meta=json.loads((a.destination/'metadata.json').read_text());meta['exact_reference_cache']={'uncached_custody_sha256':sha(a.source/'custody.json'),'implementation_sha256':sha(e/'visual_reference_cache.py'),'claim':'same literal historical metric; candidate-independent controller reference precompute'};(a.destination/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
visible=[str(x.relative_to(a.destination)) for sub in ('starter','problem') for x in (a.destination/sub).rglob('*') if x.is_file()]
hidden=[str(x.relative_to(a.destination)) for x in e.rglob('*') if x.is_file()]
seal(a.destination,task_id='visual',metric=meta['search_metric'],candidate_visible=visible,controller_only=hidden);verify_assets(a.destination/'custody.json')
print(json.dumps({'package':str(a.destination),'sealed':True,'model_calls':0,'original_package_preserved':True}))
