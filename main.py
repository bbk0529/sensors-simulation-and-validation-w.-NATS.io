import json
import asyncio
from nats.aio.client import Client as NATS

def run(loop):
    nc = NATS()
    yield from nc.connect()
    resp = yield from nc.request("hello", json.dumps({"name": "Py"}).encode(), 0.50)
    print(json.loads(resp.data))
    yield from nc.close()

if __name__ == '__main__':
  loop = asyncio.get_event_loop()
  loop.run_until_complete(run(loop))
  loop.close()