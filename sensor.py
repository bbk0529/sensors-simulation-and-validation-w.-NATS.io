import time
import numpy as np
import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers

import json
import pickle

df = pickle.load(open('temp_dwd_data.p', 'rb')).values

async def run(loop):
    nc = NATS()
    await nc.connect()
        

    def pick_data(df) : 
        size_timesteps = 30
        size_stations = 10 
        idx_station = np.random.choice(range(df.shape[0]), size=size_stations)
        idx_timesteps = np.random.choice(range(0, df.shape[1]-size_timesteps))
        ts_data = df[idx_station, idx_timesteps: idx_timesteps + size_timesteps]
        print("\n",idx_station[0], idx_timesteps)
        print(ts_data)
        return ts_data.tolist(), idx_station[0], idx_timesteps

    async def request_handler(msg):
        print(msg)
        subject = msg.subject
        reply = msg.reply
        data = msg.data.decode()
        print("sent a message on '{subject} {reply}': {data}".format(
            subject=subject, reply=reply, data=data))


    async def send_data() :         
        
        message = dict()        
        message['temperature'], idx_station, idx_timesteps= pick_data(df)
        message = json.dumps(message)        
        await nc.request(                
                "sensor.processed" + "." + str(idx_station) + "." + str(idx_timesteps),
                message.encode(),  #message, should be encoded
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