"""Deployment-only wrapper; upstream MLEvolve sources remain read-only.

Candidate namespace contains public inputs and its workspace, never API credentials,
controller logs, official answers, Docker socket or the host filesystem.
"""
import fcntl
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import types
import uuid

sys.path.insert(0, '/upstream')
LEDGER = Path('/output/usage.jsonl')
ledger_lock = threading.Lock()


def record(**data):
    data['time'] = time.time()
    with ledger_lock:
        with LEDGER.open('a') as stream:
            stream.write(json.dumps(data) + '\n')


def sandbox_command(cmd, cwd, gpu_uuid):
    # Parent controller is trusted; generated candidates are not.
    work = next(p for p in Path(cwd).parents if p.name == 'workspace') if Path(cwd).name != 'workspace' else Path(cwd)
    args = ['bwrap', '--die-with-parent', '--new-session', '--unshare-pid',
            '--unshare-net', '--unshare-ipc', '--unshare-uts', '--cap-drop', 'ALL',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp']
    for path in ('/usr', '/opt', '/lib', '/lib64', '/bin', '/sbin', '/etc', '/public', '/models'):
        if Path(path).exists():
            args += ['--ro-bind', path, path]
    for path in ([] if os.environ.get('CPU_ONLY')=='1' else glob.glob('/dev/nvidia*')):
        args += ['--dev-bind', path, path]
    args += ['--bind', str(work), str(work), '--chdir', str(cwd)]
    clean = dict(PATH='/opt/conda/envs/agent/bin:/usr/local/bin:/usr/bin:/bin',
                 HOME=str(work / 'working'), TMPDIR='/tmp', PYTHONUNBUFFERED='1',
                 PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=gpu_uuid,
                 HF_HOME='/models/hf', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                 TRANSFORMERS_CACHE='/models/transformers',
                 TORCH_HOME='/models/torch', USE_TF='0', USE_TORCH='1',
                 OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    for key, value in clean.items():
        args += ['--setenv', key, value]
    return args + ['--'] + list(cmd)


class GuardedProcess:
    def __init__(self, cmd, **kwargs):
        self.lock = None
        while self.lock is None:
            cpu_only = os.environ.get('CPU_ONLY')=='1'
            resources = ['']*6 if cpu_only else os.environ['CAMPAIGN_GPUS'].split(',')
            for slot, gpu in enumerate(resources):
                prefix = 'cpu' if cpu_only else 'gpu'
                stream = open(f'/slots/{prefix}-{slot}.lock', 'a')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.lock = stream
                    self.gpu = gpu
                    break
                except BlockingIOError:
                    stream.close()
            if self.lock is None:
                time.sleep(1)
        try:
            kwargs['env'] = {'PATH': '/usr/bin:/bin', 'HOME': '/tmp'}
            self.proc = subprocess.Popen(sandbox_command(cmd, kwargs['cwd'], self.gpu), **kwargs)
            record(event='candidate_started', pid=self.proc.pid, gpu=self.gpu,
                   resource_kind='cpu' if cpu_only else 'gpu')
            threading.Thread(target=self.release_after_exit, daemon=True).start()
        except BaseException:
            self.lock.close()
            raise

    def release_after_exit(self):
        code = self.proc.wait()
        self.lock.close()
        record(event='candidate_exited', pid=self.proc.pid, exit_code=code)

    def __getattr__(self, name):
        return getattr(self.proc, name)


def install_accounting():
    import llm.openai as backend
    original = backend.OpenAI

    def factory(*args, **kwargs):
        client = original(*args, **kwargs)
        create = client.chat.completions.create

        def tracked(**params):
            call = uuid.uuid4().hex
            if params.get('model') == 'deepseek-flash':
                params['max_tokens'] = max(int(params.get('max_tokens') or 0), 65536)
            if params.get('stream'):
                params['stream_options'] = {'include_usage': True}
            record(event='request_started', call=call, model=params['model'], max_tokens=params.get('max_tokens'))
            try:
                result = create(**params)
            except BaseException as exc:
                record(event='request_error', call=call, error_type=type(exc).__name__, usage_unknown=True)
                raise
            if not params.get('stream'):
                record(event='response', call=call, usage=result.usage.model_dump() if result.usage else None,
                       model=result.model)
                return result

            def chunks():
                seen = False
                try:
                    for chunk in result:
                        if chunk.usage:
                            record(event='response', call=call, model=chunk.model, usage=chunk.usage.model_dump())
                            seen = True
                        yield chunk
                finally:
                    result.close()
                    if not seen:
                        record(event='stream_ended_without_usage', call=call, usage_unknown=True)
            return chunks()
        client.chat.completions.create = tracked
        return client
    backend.OpenAI = factory


def preflight():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('/models/bge_ready', device='cpu')
    assert model.encode(['deployment preflight']).shape == (1, 768)
    work = Path('/output/preflight/workspace/working')
    work.mkdir(parents=True, exist_ok=True)
    cpu_only = os.environ.get('CPU_ONLY')=='1'
    script = "import os,socket,torch; assert not os.path.exists('/run/secrets/ds.key'); assert not os.path.exists('/output/usage.jsonl'); assert not os.path.exists('/official'); assert not os.path.exists('/var/run/docker.sock'); assert not os.path.exists('/data'); assert len(os.listdir('/proc'))>0; s=socket.socket(); s.settimeout(1); assert s.connect_ex(('127.0.0.1',3128))!=0; "
    script += "assert not torch.cuda.is_available(); assert not any(p.startswith('nvidia') for p in os.listdir('/dev')); print('candidate_isolation_cpu_ok')" if cpu_only else "assert torch.cuda.is_available(); print('candidate_isolation_gpu_ok')"
    result = subprocess.run(sandbox_command([sys.executable, '-c', script], work,
                                            '' if cpu_only else os.environ['CAMPAIGN_GPUS'].split(',')[0]),
                            capture_output=True, text=True, timeout=90,
                            env={'PATH': '/usr/bin:/bin', 'HOME': '/tmp'})
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    print('ZERO_MODEL_CALL_PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        preflight()
    else:
        sys.argv=[argument.replace('agent.memory_embedding_model_path=/models/bge',
                                   'agent.memory_embedding_model_path=/models/bge_ready')
                  if argument=='agent.memory_embedding_model_path=/models/bge' else argument
                  for argument in sys.argv]
        physical_gpus=list(dict.fromkeys(os.environ['CAMPAIGN_GPUS'].split(',')))
        if len(physical_gpus)==2:
            os.environ['CAMPAIGN_GPUS']=','.join(physical_gpus+physical_gpus)
        record(event='runtime_resource_configuration', candidate_slots=4,
               max_candidates_per_card=2, model='deepseek-flash', max_tokens=65536)
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver('ds_secret', lambda: Path('/run/secrets/ds.key').read_text().strip())
        import engine.executor as executor
        executor.subprocess = types.SimpleNamespace(**{name: getattr(subprocess, name) for name in dir(subprocess)})
        executor.subprocess.Popen = GuardedProcess
        install_accounting()
        import run
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        run.run()
