import numpy as np
import asyncio
from nats.aio.client import Client as NATS
import json
import pickle
import pandas as pd

df = pickle.load(open('temp_dwd_data.p', 'rb')).values

def data_creator(df, size_timesteps, size_stations ) :         
    idx_stations = np.random.choice(range(df.shape[0]), size=size_stations)
    idx_timesteps = np.random.choice(range(0, df.shape[1]-size_timesteps))
    ts_data = df[idx_stations, idx_timesteps: idx_timesteps + size_timesteps]    
    return ts_data, idx_stations, idx_timesteps

def print_data(ts_data,idx_stations,idx_timesteps) : 
    print(idx_stations)
    print(idx_timesteps)
    print(pd.DataFrame(ts_data).describe())
    print("\n")


def prepare_sending(ts_data,idx_stations,idx_timesteps) : 
    data_for_trans = dict()
    data_for_trans['temperature'] = ts_data.tolist()
    message = json.dumps(data_for_trans)
    subject = "sensor.processed" + "." + str(idx_stations[0]) + "." + str(idx_timesteps)

    return subject, message


async def run(loop):
    nc = NATS()
    await nc.connect()
        
    async def send_data() :                 
        ts_data, idx_stations, idx_timesteps  = data_creator(df, size_timesteps=30, size_stations=10)        
        print_data(ts_data,idx_stations,idx_timesteps)
        subject, message = prepare_sending(ts_data,idx_stations,idx_timesteps)

        await nc.publish(
            subject,
            message.encode()
        )

    while True : 
        await asyncio.sleep(3)
        await send_data()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    loop.close()