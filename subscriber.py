import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
import numpy as np
import json
from new_predictor import *

smoothpredictor = SmoothingAndPredict()
smoothpredictor = SpatialComparision()

async def run(loop):
    nc = NATS()
    try:        
        await nc.connect()
    except ErrNoServers as e:
        print(e)
        return

    def noise_creator(data) : 
        noise = np.zeros(len(data)) 
        noise_idx =  np.random.choice(len(noise))
        noise_value = np.random.choice(range(5,10)) * np.random.choice([-1,1])
        noise[noise_idx] += noise_value
        print("Random noise added at idx {} with value {}".format(noise_idx, noise_value))        
        return noise

    def validate(X,y, func) :
        corrected_y = func.correct(X, y)
        idx_detected = np.where(np.array(abs(y -  corrected_y)) > 3)
        print("Found noise by model", idx_detected[0])       
        print(y-corrected_y)                        
        print("\n")      

    async def message_handler(msg):               
        df = json.loads(msg.data.decode())
        ts_data = np.array(df['temperature'])
        y = ts_data[0]               
        
        print('='*100)
        print(msg)            
                
        X = ts_data[1:]
        y += noise_creator(y)

        validate(X,y, smoothpredictor)        
        

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