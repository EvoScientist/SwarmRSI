#!/usr/bin/env python3
"""One-cell recording DeepSeek transport (v3): no dollar cutoff, no poisoning on upstream failure.

The v1/v2 guards poisoned the whole cell on the first upstream hang, which
refused every retry the method itself would make; five of twelve v1 searches
ended after 8-30 calls that way. v3 treats a failed upstream attempt as an
unresolved charge (its full reservation stays in the token/cost upper bound)
and returns 504, so each method recovers with its own native retry policy,
as in the MLE-Bench and NatureBench runs. An identical request may be re-sent
only while no attempt at it has settled. Dollars are conservative rate
estimates, not provider billing. Restart of a cell state is still forbidden.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import hmac
import secrets
import json
import os
from pathlib import Path
import stat
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = 'https://api.deepseek.com/chat/completions'
INPUT_NANODOLLARS = 300  # $0.30 / million; cache misses, peak rate
OUTPUT_NANODOLLARS = 1200  # $1.20 / million; peak rate
PRICING_SOURCE = 'https://api-docs.deepseek.com/quick_start/pricing'

class Refused(Exception):
    pass

class Poisoned(Exception):
    pass

class UpstreamUnresolved(Exception):
    def __init__(self, status, reason, body=None):
        super().__init__(reason); self.status = status; self.body = body

MAX_IDENTICAL_ATTEMPTS = 13  # the method's first send plus its native retry budget

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def strict_json(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def atomic(path, raw):
    tmp = path.with_name(path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if tmp.exists(): tmp.unlink()

class Guard:
    def __init__(self, root, *, cap_nanodollars=None, max_calls=600, upstream_timeout=300,
                 max_input=10_000_000, max_output=2_000_000, deadline_seconds=10800,
                 clock=time.time):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.path = self.root / 'state.json'
        self.lock = threading.RLock()
        self.clock = clock
        self.upstream_timeout = upstream_timeout
        self.fd = os.open(self.root / 'cell.lock', os.O_RDWR | os.O_CREAT, 0o600)
        # Hold the OS lock for lifetime: prevents concurrent guard processes.
        try: fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd); raise Refused('another process owns cell')
        if self.path.exists() or any(self.root.glob('*.raw')) or any(self.root.glob('*.tmp')):
            self.close()
            raise Refused('existing cell state: audit required; restart/replay forbidden')
        if min(max_calls, max_input, max_output, deadline_seconds) <= 0 or (cap_nanodollars is not None and cap_nanodollars <= 0):
            self.close(); raise ValueError('positive limits required')
        now = clock()
        self.state = dict(schema='ds-guard-v3', status='ready', created_at=now,
                          deadline_at=now+deadline_seconds,
                          limits=dict(cap_nanodollars=cap_nanodollars, max_calls=max_calls,
                                      max_input=max_input, max_output=max_output),
                          pricing=dict(input_nanodollars=INPUT_NANODOLLARS,
                                       output_nanodollars=OUTPUT_NANODOLLARS,
                                       source=PRICING_SOURCE, classification='cost_upper_bound'),
                          calls=0, input_tokens=0, output_tokens=0,
                          cost_upper_bound_nanodollars=0, reservations={}, responses=[],
                          unresolved=[], upstream_timeout_seconds=upstream_timeout,
                          poison_reason=None)
        self.save()

    def close(self):
        if getattr(self, 'fd', None) is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd); self.fd = None

    def save(self):
        atomic(self.path, json.dumps(self.state, sort_keys=True, indent=2).encode())

    def poison(self, reason):
        with self.lock:
            self.state['status'] = 'poisoned'
            self.state['poison_reason'] = reason
            self.save()

    def reserve(self, raw):
        # Validation failures before dispatch do not poison a cell.
        try:
            body = strict_json(raw.decode('utf-8'))
            assert isinstance(body, dict)
            assert body.get('model') == 'deepseek-flash'
            assert body.get('stream', False) is False
            assert isinstance(body.get('messages'), list) and body['messages']
            out = body['max_tokens']
            assert type(out) is int and out > 0
            assert body.get('n', 1) == 1 and type(body.get('n', 1)) is int
            assert 'max_completion_tokens' not in body
            assert 'provider' not in body  # OpenRouter-only option must be omitted.
        except (ValueError, UnicodeError, AssertionError, KeyError, TypeError):
            raise Refused('invalid request: deepseek-flash/nonstream/messages/max_tokens required')
        inp = len(raw) + 2048
        cost = inp * INPUT_NANODOLLARS + out * OUTPUT_NANODOLLARS
        with self.lock:
            s = self.state; limits = s['limits']
            if s['status'] == 'poisoned': raise Poisoned(s['poison_reason'])
            if self.clock() >= s['deadline_at']: raise Refused('all-stage deadline reached')
            pending = list(s['reservations'].values())
            charged = pending + s['unresolved']
            request_hash = digest(raw)
            if any(x['request_sha256'] == request_hash for x in s['responses']):
                raise Refused('duplicate of a settled request; replay forbidden')
            if sum(x['request_sha256'] == request_hash for x in pending + s['unresolved']) >= MAX_IDENTICAL_ATTEMPTS:
                raise Refused('identical request attempt limit')
            if s['calls'] >= limits['max_calls']: raise Refused('call limit')
            if s['input_tokens'] + sum(x['input_reserved'] for x in charged) + inp > limits['max_input']:
                raise Refused('input token limit')
            if s['output_tokens'] + sum(x['output_reserved'] for x in charged) + out > limits['max_output']:
                raise Refused('output token limit')
            if limits['cap_nanodollars'] is not None and s['cost_upper_bound_nanodollars'] + sum(x['cost_reserved_nanodollars'] for x in charged) + cost > limits['cap_nanodollars']:
                raise Refused('cost limit')
            index = str(s['calls'] + 1)
            s['reservations'][index] = dict(input_reserved=inp, output_reserved=out,
                cost_reserved_nanodollars=cost, request_sha256=digest(raw), started_at=self.clock())
            s['calls'] += 1
            try:
                self.save()
                atomic(self.root / (index + '.request.raw'), raw)
            except BaseException:
                self.poison('request artifact persistence failed'); raise
            return index

    def settle(self, index, raw, status=200):
        with self.lock:
            s = self.state
            try:
                reservation = s['reservations'][index]
                atomic(self.root / (index + '.response.raw'), raw)
                if not 200 <= status < 300: raise ValueError('upstream HTTP failure')
                body = strict_json(raw.decode('utf-8'))
                if not isinstance(body, dict) or body.get('error'): raise ValueError('upstream error')
                response_id = body.get('id')
                if not isinstance(response_id, str) or not response_id: raise ValueError('missing response id')
                if response_id in [x['id'] for x in s['responses']]: raise ValueError('duplicate response id')
                usage = body['usage']; inp = usage['prompt_tokens']; out = usage['completion_tokens']
                if type(inp) is not int or type(out) is not int or inp < 0 or out < 0:
                    raise ValueError('invalid usage')
                total = usage.get('total_tokens')
                if type(total) is not int or total != inp + out: raise ValueError('inconsistent usage')
                if inp > reservation['input_reserved'] or out > reservation['output_reserved']:
                    raise ValueError('usage exceeds reservation')
                if body.get('model') not in ('deepseek-flash', 'deepseek-v4.1-flash', 'DeepSeek-V4.1-Flash'):
                    raise ValueError('response model drift')
                if not isinstance(body.get('choices'), list) or not body['choices']:
                    raise ValueError('missing choices')
                if self.clock() > s['deadline_at']: raise ValueError('response past deadline')
                cost = inp * INPUT_NANODOLLARS + out * OUTPUT_NANODOLLARS
                s['input_tokens'] += inp; s['output_tokens'] += out
                s['cost_upper_bound_nanodollars'] += cost
                s['responses'].append(dict(id=response_id, request_index=index,
                    request_sha256=reservation['request_sha256'], response_sha256=digest(raw),
                    usage=usage, response_model=body['model'], cost_upper_bound_nanodollars=cost,
                    provider_billed_dollars=None, completed_at=self.clock()))
                del s['reservations'][index]
                self.save()
            except Exception as exc:
                # Keep the full reservation as an unresolved charge; never release it.
                reason = ('unresolved response: ' + type(exc).__name__ + ': ' + str(exc)
                          if isinstance(exc, ValueError) else 'unresolved response: ' + type(exc).__name__)
                self.unresolve(index, reason, status)
                if 400 <= status < 600:
                    # Provider refusal: pass status and body through unchanged so the
                    # method sees what the provider said (retryable or not).
                    raise UpstreamUnresolved(status, reason, raw) from None
                raise UpstreamUnresolved(502, reason) from None

    def unresolve(self, index, reason, status=None):
        with self.lock:
            s = self.state
            item = s['reservations'].pop(index)
            item.update(request_index=index, reason=reason, upstream_status=status,
                        failed_at=self.clock())
            s['unresolved'].append(item)
            try:
                self.save()
            except BaseException:
                self.poison('unresolved artifact persistence failed'); raise

    def dispatch(self, raw, transport):
        index = self.reserve(raw)
        try:
            status, response = transport(raw, max(0.01, min(self.upstream_timeout,
                                                             self.state['deadline_at'] - self.clock())))
        except BaseException as exc:
            self.unresolve(index, 'upstream transport unresolved: ' + type(exc).__name__)
            raise UpstreamUnresolved(504, 'upstream transport unresolved: ' + type(exc).__name__) from None
        self.settle(index, response, status)
        return response

    def health(self):
        with self.lock:
            return dict(status=self.state['status'], calls=self.state['calls'],
                        pending=len(self.state['reservations']), unresolved=len(self.state['unresolved']), deadline_at=self.state['deadline_at'],
                        cost_upper_bound_nanodollars=self.state['cost_upper_bound_nanodollars'],
                        poison_reason=self.state['poison_reason'])


def read_secret(key_file=None, key_env=None):
    if bool(key_file) == bool(key_env): raise ValueError('one secret source required')
    if key_file:
        path = Path(key_file)
        if path.is_symlink(): raise ValueError('secret symlink forbidden')
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077 or st.st_uid != os.getuid():
            raise ValueError('private owner-only secret file required')
        value = path.read_text().strip()
    else:
        value = os.environ.get(key_env, '').strip()
    if not value or '\n' in value or '\r' in value: raise ValueError('invalid secret source')
    return value


def make_transport(secret):
    def transport(raw, remaining):
        request = urllib.request.Request(UPSTREAM, data=raw, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret})
        # urllib has no automatic retries. No alternate URLs or redirects allowed.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl): return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            # Total deadline, not only socket inactivity: keep-alive bytes must not
            # hold a hung request open past the attempt timeout.
            give_up = time.monotonic() + remaining
            with opener.open(request, timeout=remaining) as response:
                chunks = []
                while True:
                    if time.monotonic() > give_up:
                        raise TimeoutError('upstream total attempt deadline exceeded')
                    block = response.read1(65536)
                    if not block:
                        return response.status, b''.join(chunks)
                    chunks.append(block)
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
    return transport


def handler_class(guard, transport, client_token):
    if not client_token:
        raise ValueError("downstream authentication token required")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass  # Never log headers or body.
        def respond(self, status, raw):
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            if self.path != '/health': return self.respond(404, b'{"error":"route forbidden"}')
            self.respond(200, json.dumps(guard.health()).encode())
        def do_POST(self):
            auth = self.headers.get_all('Authorization', [])
            if len(auth) != 1 or not hmac.compare_digest(auth[0], 'Bearer ' + client_token):
                return self.respond(401, b'{"error":"unauthorized"}')
            if self.path not in ('/chat/completions', '/api.deepseek.com/chat/completions'):
                return self.respond(404, b'{"error":"route forbidden"}')
            try:
                if self.headers.get('Transfer-Encoding'): raise Refused('chunked body forbidden')
                values = self.headers.get_all('Content-Length', [])
                if len(values) != 1: raise Refused('one content length required')
                size = int(values[0])
                if not 0 < size <= 16 * 1024 * 1024: raise Refused('body size invalid')
                self.connection.settimeout(30)
                raw = self.rfile.read(size)
                if len(raw) != size: raise Refused('body incomplete')
                response = guard.dispatch(raw, transport)
                self.respond(200, response)
            except (Refused, ValueError) as exc:
                self.respond(400, json.dumps({'error': str(exc)}).encode())
            except UpstreamUnresolved as exc:
                # A retryable gateway status: the method's own retry policy decides.
                self.respond(exc.status, exc.body if exc.body is not None else
                             json.dumps({'error': str(exc), 'charge': 'unresolved_upper_bound'}).encode())
            except Poisoned as exc:
                self.respond(502, json.dumps({'error': str(exc), 'retry_allowed': False}).encode())
            except (BrokenPipeError, ConnectionResetError):
                pass  # Usage already settled; no upstream replay.
            except Exception:
                self.respond(500, b'{"error":"guard internal failure; inspect state","retry_allowed":false}')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--port', type=int, default=0)
    secret = parser.add_mutually_exclusive_group(required=True)
    secret.add_argument('--key-file'); secret.add_argument('--key-env')
    parser.add_argument('--cap-usd', type=str, default=None)
    parser.add_argument('--max-calls', type=int, default=600)
    parser.add_argument('--max-input', type=int, default=10_000_000)
    parser.add_argument('--max-output', type=int, default=2_000_000)
    parser.add_argument('--deadline-seconds', type=int, default=10800)
    parser.add_argument('--upstream-timeout', type=int, default=300)
    args = parser.parse_args()
    from decimal import Decimal
    cap = None if args.cap_usd is None else int(Decimal(args.cap_usd) * 1_000_000_000)
    transport = make_transport(read_secret(args.key_file, args.key_env))
    guard = Guard(args.state_dir, cap_nanodollars=cap, max_calls=args.max_calls,
                  max_input=args.max_input, max_output=args.max_output,
                  deadline_seconds=args.deadline_seconds, upstream_timeout=args.upstream_timeout)
    client_token = secrets.token_urlsafe(32)
    atomic(guard.root / 'client_token', client_token.encode())
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_class(guard, transport, client_token))
    server.daemon_threads = False
    atomic(guard.root / 'endpoint.json', json.dumps(dict(
        base_url=f'http://127.0.0.1:{server.server_port}/api.deepseek.com',
        health_url=f'http://127.0.0.1:{server.server_port}/health', pid=os.getpid())).encode())
    try: server.serve_forever()
    finally:
        server.server_close(); guard.close()

if __name__ == '__main__': main()
