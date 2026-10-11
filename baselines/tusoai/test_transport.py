import importlib.util, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run_research_baseline.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def setup_proxy(self,root,reply):
  (root/'requests').mkdir()
  lim=dict(wall_seconds=10800,model_calls=2,input_tokens=10000,output_tokens=1000,max_output_tokens_per_call=200)
  ledger=m.Ledger(root,lim)
  fake=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:SimpleNamespace(model_dump=lambda:reply))))
  proxy=m.CompletionProxy(fake,ledger,dict(limits=lim,model='deepseek-flash'))
  return proxy,ledger
 def test_usage_and_raw_retained(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{'id':'a','usage':{'prompt_tokens':12,'completion_tokens':3}})
   p.create(messages=[{'role':'user','content':'test'}])
   self.assertEqual((l.calls,l.input_tokens,l.output_tokens),(1,12,3))
   self.assertTrue((Path(d)/'requests/000001.response.json').is_file())
   with self.assertRaises(m.StopCell):p.create(messages=[])
 def test_missing_usage_stops(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{'id':'a'})
   with self.assertRaises(m.StopCell):p.create(messages=[])
 def test_cap_prevents_transport(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{})
   l.calls=2
   with self.assertRaises(m.StopCell):p.create(messages=[])
   self.assertEqual(len(list((Path(d)/'requests').iterdir())),0)
 def test_shared_clock_counts_launcher_delay_and_reserve(self):
  now=[50.0]
  l=m.Ledger(Path('.'),{'wall_seconds':10800},start_unix=900,deadline_unix=11700,
             wall_clock=lambda:1000,monotonic_clock=lambda:now[0])
  self.assertEqual(l.usage()['wall_seconds'],100)
  self.assertEqual(l.remaining(),10670)
  self.assertEqual(l.final_remaining(),10700)
  now[0]=l.search_deadline-.001
  l.check()
  now[0]=l.search_deadline
  with self.assertRaises(m.StopCell):l.check()
  self.assertEqual(l.final_remaining(),30)
  now[0]=l.final_deadline
  self.assertEqual(l.final_remaining(),0)
 def test_shared_clock_cannot_extend_envelope(self):
  with self.assertRaises(ValueError):
   m.Ledger(Path('.'),{'wall_seconds':10800},start_unix=900,deadline_unix=11701,
            wall_clock=lambda:1000,monotonic_clock=lambda:50)
 def test_no_transport_at_search_deadline(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{})
   l._monotonic=lambda:l.search_deadline
   with self.assertRaises(m.StopCell):p.create(messages=[{'role':'user','content':'test'}])
   self.assertEqual(l.calls,0)
   self.assertEqual(len(list((Path(d)/'requests').iterdir())),0)
 def test_explicit_defaults_and_response_identity(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{})
   captured={}
   response=SimpleNamespace(model_dump=lambda:{'id':'default','usage':{'prompt_tokens':8,'completion_tokens':4},'choices':[{'finish_reason':'length','message':{'content':'','reasoning_content':'think'}}]})
   def create(**kw):captured.update(kw);return response
   p.client.chat.completions.create=create
   result=p.create(messages=[{'role':'user','content':'literal prompt'}],temperature=.5)
   self.assertIs(result,response)
   self.assertEqual(captured['reasoning_effort'],'medium')
   self.assertEqual(captured['extra_body'],{'thinking':{'type':'enabled'}})
   self.assertEqual(captured['max_tokens'],200)
   self.assertEqual(captured['timeout'],600)
   self.assertEqual(captured['temperature'],.5)
   self.assertEqual(l.calls,1)  # truncated visible output never triggers wrapper retry
 def test_explicit_arguments_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{'id':'explicit','usage':{'prompt_tokens':5,'completion_tokens':2}})
   captured={};original=p.client.chat.completions.create
   def create(**kw):captured.update(kw);return original(**kw)
   p.client.chat.completions.create=create
   messages=[{'role':'user','content':'unchanged'}]
   body={'thinking':{'type':'disabled'},'top_p':.99}
   p.create(model='deepseek-flash',messages=messages,temperature=.37,max_tokens=50,timeout=120,reasoning_effort='low',extra_body=body,response_format={'type':'text'})
   self.assertIs(captured['messages'],messages)
   self.assertEqual(captured['temperature'],.37)
   self.assertEqual(captured['max_tokens'],50)
   self.assertEqual(captured['timeout'],120)
   self.assertEqual(captured['reasoning_effort'],'low')
   self.assertEqual(captured['extra_body'],body)
   self.assertEqual(body,{'thinking':{'type':'disabled'},'top_p':.99})
   self.assertEqual(captured['response_format'],{'type':'text'})
 def test_oversize_refused_before_transport(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{})
   for extra in ({'max_tokens':201},{'timeout':601}):
    with self.assertRaises(m.StopCell):p.create(messages=[{'role':'user','content':'a'}],**extra)
   self.assertEqual(l.calls,0)
 def test_wrong_model_and_stream_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p,l=self.setup_proxy(Path(d),{})
   for extra in ({'model':'other'},{'stream':True}):
    with self.assertRaises(m.StopCell):p.create(messages=[{'role':'user','content':'a'}],**extra)
   self.assertEqual(l.calls,0)
if __name__=='__main__':unittest.main()
