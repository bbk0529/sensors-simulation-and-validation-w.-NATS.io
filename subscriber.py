import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
import numpy as np
import json
from new_predictor import *

smoothpredictor = SmoothingAndPredict()

async def run(loop):
    nc = NATS()
    try:        
        await nc.connect()
    except ErrNoServers as e:
        print(e)
        return

    async def message_handler(msg):        
        data = json.loads(msg.data.decode())        
        ts_data = np.array(data['temperature'])
        y = ts_data[0]
        # noise =  np.random.randn(len(ts_data[0])) * 10
        noise = np.zeros(len(y)) 
        noise_idx =  np.random.choice(len(noise))
        noise_value = np.random.choice(range(5,10)) * np.random.choice([-1,1])
        noise[noise_idx] += noise_value
        print('='*100)
        print(msg.subject)
        print("Random noise added at idx {} with value {}".format(noise_idx, noise_value))        
                        
        
        X = ts_data[1:]
        y += noise
        corrected_y = smoothpredictor.correct(X, y)
        idx_detected = np.where(np.array(abs(y -  corrected_y)) > 3)
        print("Found noise by model", idx_detected[0], idx_detected[0] == noise_idx)       
        print(y-corrected_y)                        
        print("\n")
        
        # print(ts_data)
        # print(ts_data > 0 )
        

    await nc.subscribe(">", cb=message_handler)    
    



    try:        
        await nc.flush(1)
    except ErrTimeout:
        print("Flush timeout")
    await asyncio.sleep(100000)    
    await nc.drain()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    loop.close()