#!/usr/bin/env python3
"""Research-release TusoAI, unchanged search plus explicit transport/execution adapters (v3).

v3 restores TusoAI's native transport behaviour: the upstream client is the
OpenAI SDK with its default two retries, and a call that still fails is raised
to TusoAI's own `except Exception` handlers (which return an empty reply and
continue), instead of ending the whole cell on the first failure as v1 did.
No call without --execute. Bridge must enforce candidate isolation; this runner
never executes candidate Python in its own privileged controller namespace.
"""
import argparse, hashlib, json, math, os, random, shutil, signal, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

class StopCell(BaseException):
    pass

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + '\n')

class Ledger:
    def __init__(self, work, limits, *, start_unix=None, deadline_unix=None,
                 wall_clock=time.time, monotonic_clock=time.monotonic):
        self.work, self.limits = work, limits
        self._monotonic = monotonic_clock
        now, mono = wall_clock(), monotonic_clock()
        self.started_at_unix = now if start_unix is None else float(start_unix)
        self.deadline_unix = self.started_at_unix + limits['wall_seconds'] if deadline_unix is None else float(deadline_unix)
        if not all(math.isfinite(x) for x in (self.started_at_unix,self.deadline_unix)):
            raise ValueError('nonfinite shared clock')
        if self.started_at_unix > now+1 or not 30 < self.deadline_unix-self.started_at_unix <= limits['wall_seconds']+1e-6:
            raise ValueError('invalid shared trial interval')
        self.finalization_reserve = 30
        self.start = mono - (now-self.started_at_unix)
        self.final_deadline = mono + self.deadline_unix-now
        self.search_deadline = self.final_deadline-self.finalization_reserve
        self.calls = self.input_tokens = self.output_tokens = self.evaluations = 0
        self.ids = set()
    def event(self, kind, **data):
        with (self.work / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'event':kind,'elapsed_seconds':self._monotonic()-self.start,**data},default=str)+'\n')
    def remaining(self):
        return self.search_deadline-self._monotonic()
    def final_remaining(self):
        return self.final_deadline-self._monotonic()
    def check(self):
        if self.remaining() <= 0: raise StopCell('wall_budget')
    def usage(self):
        return dict(calls=self.calls,input_tokens=self.input_tokens,output_tokens=self.output_tokens,
                    evaluations=self.evaluations,wall_seconds=self._monotonic()-self.start,
                    trial_started_at_unix=self.started_at_unix,trial_deadline_unix=self.deadline_unix,
                    finalization_reserve_seconds=self.finalization_reserve)

class CompletionProxy:
    def __init__(self, client, ledger, config):
        self.client, self.ledger, self.config = client, ledger, config
    def create(self, **kw):
        l=self.ledger; l.check()
        control=self.config.get('transport',{})
        kw.setdefault('model',self.config['model'])
        if kw['model'] != self.config['model'] or kw['model'] != 'deepseek-flash':
            raise StopCell('transport_model_mismatch')
        if kw.get('stream',False) is not False: raise StopCell('streaming_not_qualified')
        ceiling=self.config['limits']['max_output_tokens_per_call']
        out=kw.setdefault('max_tokens',ceiling)
        if type(out) is not int or not 0 < out <= ceiling:
            raise StopCell('upstream_max_tokens_exceeds_transport_cap')
        kw.setdefault('reasoning_effort',control.get('reasoning_effort','medium'))
        extra=dict(kw.get('extra_body') or {})
        extra.setdefault('thinking',control.get('thinking',{'type':'enabled'}))
        if 'provider' in extra or 'provider' in kw: raise StopCell('nonofficial_provider_parameter')
        kw['extra_body']=extra
        timeout_ceiling=control.get('request_timeout_seconds',600)
        requested_timeout=kw.get('timeout',timeout_ceiling)
        if not isinstance(requested_timeout,(int,float)) or not 0 < requested_timeout <= timeout_ceiling:
            raise StopCell('upstream_timeout_exceeds_transport_cap')
        # The absolute cell boundary is always authoritative, including mid-call.
        kw['timeout']=min(requested_timeout,l.remaining())
        reserve=len(json.dumps({k:v for k,v in kw.items() if k!='timeout'},ensure_ascii=False).encode())+1024
        if l.calls >= l.limits['model_calls']: raise StopCell('model_call_budget')
        if l.input_tokens+reserve > l.limits['input_tokens']: raise StopCell('input_token_reservation')
        if l.output_tokens+out > l.limits['output_tokens']: raise StopCell('output_token_reservation')
        l.calls+=1
        prefix=l.work/'requests'/('%06d'%l.calls)
        write(str(prefix)+'.request.json',kw)
        l.event('request_reserved',call=l.calls,input_reserved=reserve,output_reserved=out)
        try: response=self.client.chat.completions.create(**kw)
        except Exception as e:
            # Usage stays unresolved in the guard ledger; TusoAI handles the failure natively.
            l.event('transport_failed_after_native_retries',call=l.calls,error=repr(e)[:500])
            raise
        raw=response.model_dump()
        write(str(prefix)+'.response.json',raw)
        usage=raw.get('usage') or {}
        it,ot=usage.get('prompt_tokens'),usage.get('completion_tokens')
        rid=raw.get('id')
        if not rid or rid in l.ids or type(it) is not int or type(ot) is not int:
            raise StopCell('missing_or_duplicate_response_usage')
        l.ids.add(rid); l.input_tokens+=it; l.output_tokens+=ot
        choice=(raw.get('choices') or [{}])[0]
        message=choice.get('message') or {}
        l.event('response',call=l.calls,id=rid,input_tokens=it,output_tokens=ot,
                reported_cost=usage.get('cost'),provider=raw.get('provider'),
                finish_reason=choice.get('finish_reason'),visible_content_chars=len(message.get('content') or ''),
                reasoning_content_chars=len(message.get('reasoning_content') or ''))
        if it>reserve or ot>out: raise StopCell('reservation_exceeded')
        return response

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args(); c=json.loads(a.config.read_text())
    upstream=Path(c['upstream_root']).resolve(); template=Path(c['template']).resolve()
    required=['model','limits','task_root','bridge_command','literature','task_description','data_available']
    for k in required:
        if k not in c: raise ValueError('missing '+k)
    if not c['bridge_command']: raise ValueError('isolated bridge required')
    if c['model'] != 'deepseek-flash': raise ValueError('campaign requires official deepseek-flash')
    if not c.get('base_url', '').endswith('/api.deepseek.com'): raise ValueError('campaign requires local guard route /api.deepseek.com')
    identity={'upstream_revision':'097ac9a9430d5045d7b4e391b03f095084f9c3a3',
              'upstream_files':{str(x.relative_to(upstream)):sha(x) for x in sorted((upstream/'tusoai').glob('*.py'))},
              'template_sha256':sha(template),'config_sha256':sha(a.config), 'config':c}
    if not a.execute:
        print(json.dumps({'dry_run':True,'identity':identity},indent=2));return
    public_contract_path=Path(c['task_root'])/'problem/README.md'
    public_contract_text=public_contract_path.read_text()
    work=Path(c['work']).resolve(); work.mkdir(parents=True,exist_ok=False)
    (work/'requests').mkdir(); (work/'evaluations').mkdir()
    write(work/'identity.json',identity)
    os.chdir(work);sys.path.insert(0,str(upstream))
    l=Ledger(work,c['limits'],start_unix=float(os.environ['SWARMRSI_TRIAL_STARTED_AT_UNIX']),
             deadline_unix=float(os.environ['OP_TRIAL_DEADLINE_UNIX'])); random.seed(c.get('seed',0))
    # Hard total wall includes literature and all initialization, unlike upstream TIME_LIMIT.
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(StopCell('wall_budget')))
    signal.setitimer(signal.ITIMER_REAL,max(.001,l.remaining()))
    status='failed';reason=None
    try:
        l.check()
        import numpy as np
        np.random.seed(c.get('seed',0))
        from openai import OpenAI
        from tusoai import tusoai, optimizer, extract_literature, summarize_literature
        key=os.environ['DEEPSEEK_API_KEY']
        rawclient=OpenAI(api_key=key,base_url=c['base_url'],max_retries=2)  # SDK default, as upstream TusoAI
        client=SimpleNamespace(chat=SimpleNamespace(completions=CompletionProxy(rawclient,l,c)))
        reference=work/'initial.py';shutil.copy2(template,reference)
        best=None
        def evaluate(script_name,timeout=1500,val_limit=1.0):
            nonlocal best
            l.check()
            if l.evaluations>=l.limits['evaluations']: raise StopCell('evaluation_budget')
            l.evaluations+=1
            out=work/'evaluations'/('%06d'%l.evaluations);out.mkdir()
            candidate=out/'candidate.py';shutil.copy2(script_name,candidate)
            cmd=[v.replace('{candidate}',str(candidate)).replace('{task_root}',c['task_root']).replace('{output_dir}',str(out)) for v in c['bridge_command']]
            l.event('evaluation_started',attempt=l.evaluations,candidate_sha256=sha(candidate),command=cmd)
            proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,
                env={**os.environ,'BASELINE_CELL_DEADLINE_MONOTONIC':str(l.search_deadline)})
            try:
                stdout,stderr=proc.communicate(timeout=min(timeout+c.get('grader_timeout',300)+30,l.remaining()))
            except BaseException as exc:
                os.killpg(proc.pid,signal.SIGKILL);proc.communicate()
                if isinstance(exc,subprocess.TimeoutExpired):
                    l.event('evaluation_timeout',attempt=l.evaluations);return 'Error: timed out'
                raise
            (out/'bridge.stdout').write_text(stdout);(out/'bridge.stderr').write_text(stderr)
            if proc.returncode: return 'Error: isolated executor failed: '+stderr[-2000:]
            result=json.loads(stdout);write(out/'score.json',result)
            if result.get('valid') is not True: return 'Error: invalid output: '+str(result.get('error',''))[-2000:]
            score=float(result['score'])
            if not math.isfinite(score): return 'Error: nonfinite evaluator result'
            utility=1-score if c['metric_direction']=='minimize' else score
            l.event('evaluation_completed',attempt=l.evaluations,score=score,utility=utility)
            if best is None or utility>best['utility']:
                best={'score':score,'utility':utility,'candidate_sha256':sha(candidate),'candidate':str(candidate)}
                shutil.copy2(candidate,work/'best.py');write(work/'best.json',best)
            return {'evaluation':utility,'model_info':str(result.get('model_info',''))[:2000]}
        optimizer.run_and_evaluate=evaluate
        lit=c['literature'];mode=lit['mode']
        if mode=='upstream_live':
            # Upstream run_extraction assigns a local variable; set its actual module global.
            extract_literature.SEMANTIC_SCHOLAR_API_KEY=os.environ.get('SEMANTIC_SCHOLAR_API_KEY','')
            papers=extract_literature.run_extraction(c['task_description'],api_key=extract_literature.SEMANTIC_SCHOLAR_API_KEY,top_n=lit.get('top_n',10))
            write(work/'literature_papers.json',papers)
            if not papers: raise StopCell('literature_empty')
            summaries=summarize_literature.summarise_papers(papers,client,c['model'])
        elif mode=='cached_extracted_papers':
            path=Path(lit['path']);assert sha(path)==lit['sha256']
            papers=json.loads(path.read_text())
            if not papers or not lit.get('provenance'): raise StopCell('literature_provenance_missing')
            summaries=summarize_literature.summarise_papers(papers,client,c['model'])
        else: raise StopCell('unsupported_literature_mode')
        if not summaries or not all(summaries.values()):raise StopCell('literature_summary_empty')
        write(work/'summaries.json',summaries)
        task=c['task_description'];data=public_contract_text;model=c['model']
        write(work/'public_task_contract.json',{'text':data,'sha256':sha(Path(c['task_root'])/'problem/README.md')})
        categories=tusoai.make_categories(task,data,num_cat=10,summaries=summaries,client=client,model=model)
        instructions=tusoai.make_instructions(task,data,categories,summaries,10,client=client,model=model)
        solutions=tusoai.make_solutions(task,data,5,summaries,client=client,model=model)
        probabilities=tusoai.make_probabilities(task,data,categories,solutions,client=client,model=model)
        write(work/'preparation.json',dict(categories=categories,instructions=instructions,solutions=solutions,probabilities=probabilities))
        selected, history = tusoai.discover_method(llm_model=model,temperature=.5,client=client,prompts=instructions,probabilities=probabilities,
          reference_filename=str(reference),initialisations=solutions,n_generations=10000,children_per_model=1,
          bug_retries=3,initial_bug_fix_attempts=5,timeout=c.get('candidate_timeout',1500),n_feedback_buffer=5,
          skip_timeout=True,drop_island_iter=2,prompt_samples=3,alter_info_samples=3,prompt_decay=1.1,
          hints=c.get('hints',['Keep the function header, input, output the same.']),filename=str(work/'search'),
          use_initial=False,TIME_LIMIT=max(1,int(l.remaining())),task_description=task,val_limit=float('inf'),debug=False,
          alter_info_prompts=json.loads((upstream/'tusoai/diagnostic_prompts.json').read_text()),history_path=str(work/'history.json'))
        selected_path=Path(selected.file)
        selected_hash=sha(selected_path)
        retained=[p for p in (work/'evaluations').glob('*/candidate.py') if sha(p)==selected_hash and (p.parent/'score.json').is_file()]
        if not retained: raise StopCell('upstream_final_candidate_not_attested')
        valid_retained=[p for p in retained if json.loads((p.parent/'score.json').read_text()).get('valid') is True]
        if not valid_retained: raise StopCell('upstream_final_candidate_invalid')
        shutil.copy2(valid_retained[-1],work/'upstream_selected.py')
        write(work/'upstream_selected.json',{'candidate_sha256':selected_hash,'source':str(valid_retained[-1]),
             'accuracy':getattr(selected,'accuracy',None),'lineage':getattr(selected,'lineage',None)})
        status='completed'
    except StopCell as e: status='stopped';reason=str(e)
    except Exception as e: reason=repr(e);raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        remaining=l.final_remaining()
        if remaining<=0: raise TimeoutError('shared finalization deadline exhausted')
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('shared finalization deadline exhausted')))
        signal.setitimer(signal.ITIMER_REAL,remaining)
        try:
            write(work/'terminal.json',dict(status=status,reason=reason,usage=l.usage()))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
