"""Inside the network-less sandbox: forward 127.0.0.1:<port> to a unix socket bound in from the host.
usage: sandbox_fwd.py <port>=<socket> [<port>=<socket> ...]"""
import asyncio, sys

async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data); await w.drain()
    except Exception:
        pass
    finally:
        w.close()

def handler(sock):
    async def handle(r, w):
        try:
            ur, uw = await asyncio.open_unix_connection(sock)
        except Exception:
            w.close(); return
        await asyncio.gather(pipe(r, uw), pipe(ur, w))
    return handle

async def main():
    servers = []
    for spec in sys.argv[1:]:
        port, sock = spec.split("=", 1)
        servers.append(await asyncio.start_server(handler(sock), "127.0.0.1", int(port)))
    await asyncio.gather(*(s.serve_forever() for s in servers))
asyncio.run(main())
