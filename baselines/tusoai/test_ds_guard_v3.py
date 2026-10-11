import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from ds_guard_v3 import (Guard, Refused, UpstreamUnresolved, handler_class,
                         MAX_IDENTICAL_ATTEMPTS)

RAW = b'{ "model" : "deepseek-flash", "messages":[{"role":"user","content":"hello"}], "max_tokens":100,"stream":false }'
def response(id='r1', inp=20, out=10):
    return json.dumps(dict(id=id, model='deepseek-flash', choices=[{'message': {'content': 'ok'}}],
                          usage=dict(prompt_tokens=inp, completion_tokens=out, total_tokens=inp+out))).encode()

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)/'cell'; self.guard = None
    def tearDown(self):
        if self.guard: self.guard.close()
        self.tmp.cleanup()
    def make(self, **kw):
        self.guard = Guard(self.root, **kw); return self.guard

    def test_no_dollar_cap_by_default(self):
        g = self.make()
        self.assertIsNone(g.state['limits']['cap_nanodollars'])

    def test_body_unchanged_and_cost_attested(self):
        g = self.make(); seen = []
        out = g.dispatch(RAW, lambda raw, rem: (seen.append(raw), (200, response()))[1])
        self.assertEqual(seen, [RAW]); self.assertEqual(out, response())
        self.assertEqual(g.state['cost_upper_bound_nanodollars'], 20*300 + 10*1200)

    def test_hang_is_unresolved_not_poison_and_identical_retry_settles(self):
        g = self.make(upstream_timeout=7); seen = []
        def hang(raw, remaining):
            seen.append(remaining); raise TimeoutError()
        with self.assertRaises(UpstreamUnresolved) as ctx:
            g.dispatch(RAW, hang)
        self.assertEqual(ctx.exception.status, 504)
        self.assertLessEqual(seen[0], 7)
        self.assertEqual(g.state['status'], 'ready')
        self.assertEqual(len(g.state['unresolved']), 1); self.assertEqual(g.state['reservations'], {})
        g.dispatch(RAW, lambda *_: (200, response()))
        self.assertEqual(len(g.state['responses']), 1)
        self.assertEqual(g.state['calls'], 2)
        with self.assertRaisesRegex(Refused, 'settled'):
            g.dispatch(RAW, lambda *_: (200, response('r2')))

    def test_unresolved_charge_counts_against_limits(self):
        g = self.make(max_output=150)
        with self.assertRaises(UpstreamUnresolved):
            g.dispatch(RAW, lambda *_: (_ for _ in ()).throw(ConnectionResetError()))
        with self.assertRaisesRegex(Refused, 'output token limit'):
            g.reserve(RAW.replace(b'hello', b'other'))

    def test_identical_attempt_limit(self):
        g = self.make()
        for _ in range(MAX_IDENTICAL_ATTEMPTS):
            with self.assertRaises(UpstreamUnresolved):
                g.dispatch(RAW, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        with self.assertRaisesRegex(Refused, 'attempt limit'):
            g.reserve(RAW)

    def test_provider_error_status_and_body_pass_through(self):
        g = self.make(); body = b'{"error":{"message":"context too long"}}'
        with self.assertRaises(UpstreamUnresolved) as ctx:
            g.dispatch(RAW, lambda *_: (400, body))
        self.assertEqual((ctx.exception.status, ctx.exception.body), (400, body))
        self.assertEqual(g.state['status'], 'ready')

    def test_usage_exceeding_reservation_is_unresolved(self):
        g = self.make()
        with self.assertRaises(UpstreamUnresolved):
            g.dispatch(RAW, lambda *_: (200, response(out=101)))
        self.assertEqual(g.state['cost_upper_bound_nanodollars'], 0)
        self.assertEqual(len(g.state['unresolved']), 1)

    def test_restart_forbidden(self):
        self.make(); self.guard.close(); self.guard = None
        with self.assertRaisesRegex(Refused, 'existing cell state'):
            Guard(self.root)

    def test_http_client_sees_retryable_504_then_success(self):
        g = self.make(); attempts = []
        def transport(raw, remaining):
            attempts.append(1)
            if len(attempts) == 1: raise TimeoutError()
            return 200, response()
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_class(g, transport, 'tok'))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f'http://127.0.0.1:{server.server_port}/api.deepseek.com/chat/completions'
        def post():
            req = urllib.request.Request(url, data=RAW, method='POST',
                                         headers={'Authorization': 'Bearer tok', 'Content-Type': 'application/json'})
            return urllib.request.urlopen(req, timeout=10).read()
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                post()
            # Exactly the marker the frozen core classifies as transient.
            self.assertIn('HTTP Error 504', str(ctx.exception))
            self.assertEqual(post(), response())
        finally:
            server.shutdown(); server.server_close()

if __name__ == '__main__':
    unittest.main()
