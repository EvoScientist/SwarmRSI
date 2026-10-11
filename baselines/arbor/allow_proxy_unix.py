"""HTTP CONNECT proxy on a unix socket that only lets api.deepseek.com:443 through (direct egress).

Arbor runs in a network-less bwrap sandbox; this socket is its only way out, for the model API.
usage: allow_proxy_unix.py <socket path>
"""
import asyncio, sys
SOCK, ALLOWED = sys.argv[1], {("api.deepseek.com", 443)}

async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data); await w.drain()
    except Exception:
        pass
    finally:
        w.close()

async def handle(r, w):
    try:
        head = await r.readuntil(b"\r\n\r\n")
    except Exception:
        w.close(); return
    parts = head.split(b"\r\n")[0].decode(errors="replace").split()
    if len(parts) < 2 or parts[0] != "CONNECT":
        w.write(b"HTTP/1.1 403 Forbidden\r\n\r\n"); w.close(); return
    host, _, port = parts[1].rpartition(":")
    if (host, int(port or 0)) not in ALLOWED:
        print("blocked", parts[1], flush=True)
        w.write(b"HTTP/1.1 403 Forbidden\r\n\r\n"); w.close(); return
    try:
        ur, uw = await asyncio.open_connection(host, int(port))
    except Exception as e:
        print("upstream failed", host, e, flush=True)
        w.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n"); w.close(); return
    w.write(b"HTTP/1.1 200 Connection established\r\n\r\n"); await w.drain()
    await asyncio.gather(pipe(r, uw), pipe(ur, w))

async def main():
    server = await asyncio.start_unix_server(handle, SOCK)
    async with server:
        await server.serve_forever()
asyncio.run(main())
