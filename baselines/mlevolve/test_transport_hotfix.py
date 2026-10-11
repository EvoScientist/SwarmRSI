import pathlib
import sys
import types

sent=[]
events=[]

def original_factory(**kwargs):
    def create(**params):
        sent.append(params)
        chunk=types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason='stop')])
        return iter([chunk]) if params.get('stream') else chunk
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))

backend=types.SimpleNamespace(OpenAI=original_factory)
sys.modules['llm.openai']=backend
existing_client=original_factory()
source=pathlib.Path(__file__).with_name('transport_hotfix.py').read_text()
namespace={'record':lambda **data:events.append(data)}
exec(compile(source,'transport_hotfix.py','exec'),namespace)
fixed_factory=backend.OpenAI
exec(compile(source,'transport_hotfix.py','exec'),namespace)
assert backend.OpenAI is fixed_factory
client=backend.OpenAI()
result=list(client.chat.completions.create(model='deepseek-flash',max_tokens=16384,
    stream=True,messages=[{'role':'user','content':'test'}],extra_body={'thinking':{'type':'enabled'},'reasoning_effort':'high'}))
assert len(result)==1 and result[0].choices[0].finish_reason=='stop'
assert sent[-1]['max_tokens']==65536
assert sent[-1]['extra_body']['reasoning_effort']=='high'
client.chat.completions.create(model='deepseek-flash',max_tokens=131072)
assert sent[-1]['max_tokens']==131072
client.chat.completions.create(model='other',max_tokens=16384)
assert sent[-1]['max_tokens']==16384
existing_client.chat.completions.create(model='deepseek-flash',max_tokens=16384)
assert sent[-1]['max_tokens']==16384
assert sum(e['event']=='output_cap_live_fix_installed' for e in events)==1
print('PASS: cap correction, unchanged request fields/results, larger caps, idempotence, existing clients preserved; zero network')
