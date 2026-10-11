#!/usr/bin/env python3
"""One sandboxed execution, followed by attested output-only scoring."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--driver',type=Path,required=True)
    p.add_argument('--runtime-python',required=True)
    p.add_argument('--execution-timeout-seconds','--timeout-seconds',dest='execution_timeout_seconds',type=int,default=1500)
    p.add_argument('--grader-timeout-seconds',type=int,default=300)
    p.add_argument('--candidate-runtime-file',type=Path)
    a=p.parse_args();w=a.output_dir/'workspace';w.mkdir()
    shutil.copy2(a.candidate,w/'solution.py');shutil.copy2(a.driver,w/'run.py')
    evidence=w/'.evolvcoder/execution_validation.json';evidence.parent.mkdir()
    result=a.output_dir/'external-result.json'
    common=[sys.executable,'-m','evolvcoder.naturebench_evaluator','--workspace',str(w),'--task-root',str(a.task_root),
         '--runtime-python',a.runtime_python,'--require-candidate-sandbox']
    if a.candidate_runtime_file:common+=['--candidate-runtime-json',a.candidate_runtime_file.read_text()]
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN','SECRET'))
         and k not in {'EVOLVCODER_TRIAL_STARTED_AT_UNIX','OP_TRIAL_DEADLINE_UNIX','BASELINE_CELL_DEADLINE_MONOTONIC'}}
    deadline=float(os.environ.get('BASELINE_CELL_DEADLINE_MONOTONIC','inf'))
    def phase(mode,destination,ceiling):
        remaining=deadline-time.monotonic()
        if remaining<1:raise TimeoutError('all-stage cell deadline exhausted')
        timeout=min(ceiling,int(remaining)) if remaining!=float('inf') else ceiling
        cmd=common+[mode,'--result',str(destination),'--timeout-seconds',str(timeout)]
        (a.output_dir/(mode[2:]+'.command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
        r=subprocess.run(cmd,env=env,capture_output=True,text=True)
        (a.output_dir/(mode[2:]+'.stdout')).write_text(r.stdout)
        (a.output_dir/(mode[2:]+'.stderr')).write_text(r.stderr)
        if not destination.is_file():raise RuntimeError('core evaluator produced no result: '+r.stderr[-1500:])
        return json.loads(destination.read_text())
    data=phase('--execution-only',evidence,a.execution_timeout_seconds)
    if data.get('valid') is True:
        data=phase('--score-existing-output',result,a.grader_timeout_seconds)
    else:
        result.write_text(json.dumps(data,indent=2)+'\n')
    valid=data.get('valid') is True
    if valid and data.get('details',{}).get('candidate_execution_reused') is not True:
        raise RuntimeError('grading did not reuse attested execution')
    print(json.dumps({'valid':valid,'score':data.get('metrics',{}).get('score'),
       'error':None if valid else data.get('details',data),
       'model_info':str(data.get('details',{}).get('candidate_execution_log') or '')[:2000],
       'result_path':str(result),'execution_evidence_path':str(evidence)}))

if __name__=='__main__':main()
