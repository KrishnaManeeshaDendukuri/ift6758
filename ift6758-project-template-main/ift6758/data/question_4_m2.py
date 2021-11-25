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
    df['rebound_same_team'] = np.where(rebound & (df['attacking_team'] == df['previous_attacking_team']), 1, 0)
    df['home_team_attacking'] = np.where(df['attacking_team'] == df['home_team'], 1, 0)
    df['overtime'] = np.where(df['period']>3, 1, 0)
    
    df['speed'] = (df['distance_from_last_event']/df['time_since_last_event']).round(decimals=2)

    # bonus: penalty_features
    df['5v5'] = np.where((df['home_players'] == 5)&(df['away_players'] == 5), 1, 0)
    df['4v4'] = np.where((df['home_players'] == 4)&(df['away_players'] == 4), 1, 0)
    df['3v3'] = np.where((df['home_players'] == 3)&(df['away_players'] == 3), 1, 0)
    df['5v4'] = np.where((df['home_players'] == 5)&(df['away_players'] == 4)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 4)&(df['away_players'] == 5)&(df['attacking_team'] == df['away_team']), 1, 0)
    df['5v3'] = np.where((df['home_players'] == 5)&(df['away_players'] == 3)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 3)&(df['away_players'] == 5)&(df['attacking_team'] == df['away_team']), 1, 0)
    df['4v3'] = np.where((df['home_players'] == 4)&(df['away_players'] == 3)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 3)&(df['away_players'] == 4)&(df['attacking_team'] == df['away_team']), 1, 0)
    df['4v5'] = np.where((df['home_players'] == 4)&(df['away_players'] == 5)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 5)&(df['away_players'] == 4)&(df['attacking_team'] == df['away_team']), 1, 0)
    df['3v5'] = np.where((df['home_players'] == 3)&(df['away_players'] == 5)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 5)&(df['away_players'] == 3)&(df['attacking_team'] == df['away_team']), 1, 0)
    df['3v4'] = np.where((df['home_players'] == 3)&(df['away_players'] == 4)&(df['attacking_team'] == df['home_team']), 1, 0) +\
                np.where((df['home_players'] == 4)&(df['away_players'] == 3)&(df['attacking_team'] == df['away_team']), 1, 0)
    
    df['power_play'] = df['5v4'] + df['4v3'] + df['5v3'] 
    df['penalty_kill'] = df['4v5'] + df['3v4'] + df['3v5'] 
    
    return df
    