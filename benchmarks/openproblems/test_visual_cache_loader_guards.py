import json,tempfile
from pathlib import Path
import visual_reference_cache as c
import visual_historical_components as m
with tempfile.TemporaryDirectory() as td:
 p=Path(td);meta=p/'reference.json'
 meta.write_text(json.dumps({'files':{}}))
 try:c.components(None,None,p,p/'gold',Path(m.__file__))
 except ValueError as e:assert str(e)=='reference cache file set mismatch'
 else:raise AssertionError('missing file manifest accepted')
 meta.write_text(json.dumps({'files':{k:'unused' for k in ('trust_distances.npy','high_distances.npy','original_radii.npy')}}))
 try:c.components(None,None,p,p/'gold',Path(c.__file__))
 except ValueError as e:assert str(e)=='imported historical helper differs from bound implementation'
 else:raise AssertionError('wrong imported helper accepted')
print('PASS exact file-set and actual imported-helper guards')
