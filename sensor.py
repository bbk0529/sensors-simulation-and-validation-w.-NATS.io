import time
import numpy as np
import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers

import json
import pickle

df = pickle.load(open('temp_dwd_data.p', 'rb'))
data = df.values

async def run(loop):
    nc = NATS()
    await nc.connect()
        
    async def request_handler(msg):
        print(msg)
        subject = msg.subject
        reply = msg.reply
        data = msg.data.decode()
        print("sent a message on '{subject} {reply}': {data}".format(
            subject=subject, reply=reply, data=data))


    async def send_data() : 
        # data = np.random.randn(10)
        size = 20
        idx_station = np.random.choice(range(data.shape[0]), size=10)
        idx_timesteps = np.random.choice(range(0, data.shape[1]-size))
        ts_data = data [idx_station, idx_timesteps: idx_timesteps + size]
        print("\n",idx_station[0], idx_timesteps)
        print(ts_data)
        message = json.dumps({"temperature" : ts_data.tolist()})
        # print(message)
        await nc.request(
                "sensor" + "." + str(idx_station[0]) + "." + str(idx_timesteps),
                message.encode(), 
                expected=100, 
                cb=request_handler
        )

    while True : 
        await asyncio.sleep(3)
        await send_data()


    




if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    loop.close()