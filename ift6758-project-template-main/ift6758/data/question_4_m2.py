import pandas as pd
import numpy as np
import os

data = "tidy_df.csv"

def get_df(data):
    df = pd.read_csv(data)
    return df

def get_new_features():
    df = get_df(data)
    df['game_seconds'] = df['period_time']
    df1 = df['game_seconds'].str.split(':',expand=True).astype(int)
    df['game_seconds'] = df1[0]*60+df1[1]
    df['game_seconds'] = ((df['period']-1)*(60*20) + df['game_seconds'])
    df['previous_event_game_seconds'] = df['previous_event_period_time']
    df2 = df['previous_event_game_seconds'].str.split(':',expand=True).astype(int)
    df['previous_event_game_seconds'] = df2[0]*60+df2[1]
    df['previous_event_game_seconds'] = ((df['previous_event_period']-1)*(60*20) + df['previous_event_game_seconds'])
    df['time_since_last_event'] = df['game_seconds']-df['previous_event_game_seconds']
    df['distance_from_last_event'] = np.sqrt((df['x_coordinates']**2 - df['previous_event_x_coordinates'])**2 + (df['y_coordinates'] - df['previous_event_y_coordinates'])**2)
    df['distance_from_last_event'] = df['distance_from_last_event'].round(decimals=2)
    df['distance_from_net'] = df['distance_from_net'].round(decimals=2)
    rebound = np.empty(len(df.index),dtype=bool)
    for index, row in df.iterrows():
        if row['previous_event_type']=='Shot':
            rebound[index]=True
        else:
            rebound[index]=False
    df['rebound'] = rebound.tolist()
    
    df['speed'] = (df['distance_from_last_event']/df['time_since_last_event']).round(decimals=2)
    return df
    