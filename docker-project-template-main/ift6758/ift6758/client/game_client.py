import json
import requests
import pandas as pd
import logging
import os
import numpy as np
import math 

class GameClient:
    def __init__(self,game_id):
        self.game_id = game_id
        self.tracker = 0
        self.game = None
        
    def get_game(self):
        file_path = './' + str(self.game_id) + '.json'
        file = str(self.game_id) + '.json'
        data = requests.get(f'https://statsapi.web.nhl.com/api/v1/game/{self.game_id}/feed/live/')
        if (data.status_code == 404):
            return None
        with open(file_path, 'w') as f:
            json.dump(data.json(), f)
        
        return file_path


    def update_events(self):
        self.get_game()
        file_path = './' + str(self.game_id) + '.json'
        attributes = []
        with open(file_path,'r') as f:
            game = json.load(f)
        events = game['liveData']['plays']['allPlays']
        previous_event = None
        home_team = game['gameData']['teams']['home']['name']
        away_team = game['gameData']['teams']['away']['name']
        
        home_players = 5
        away_players = 5
        time_since_powerplay_started = 0
        penalties = {
            'home_minor2_penalty_stack' : [],
            'home_minor4_penalty_stack' : [],
            'home_major_penalty_stack' : [],
            'away_minor2_penalty_stack' : [],
            'away_minor4_penalty_stack' : [],
            'away_major_penalty_stack' : []
        }
        
        for event in events:
            event['time_seconds'] = (event['about']['period']-1)*20*60 + int(event['about']['periodTime'][0:2])*60 + int(event['about']['periodTime'][3:5])
            
            #update time remaining on penalties, remove penalties if they are over
            if previous_event is not None: 
                time_since_last_event = event['time_seconds'] - previous_event.get('time_seconds')

                for key, penalty_stack in penalties.items():
                    for penalty in range(len(penalty_stack)):
                        penalty_stack[penalty] -= time_since_last_event
                    penalties[key] = [penalty for penalty in penalty_stack if penalty>0]

                # recompute the number of players on the ice
                home_players = max(5 - len(penalties['home_minor2_penalty_stack']) - len(penalties['home_minor4_penalty_stack']) - len(penalties['home_major_penalty_stack']), 3)
                away_players = max(5 - len(penalties['away_minor2_penalty_stack']) - len(penalties['away_minor4_penalty_stack']) - len(penalties['away_major_penalty_stack']), 3)
                
                if (home_players == away_players):
                    time_since_powerplay_started = 0
                else:
                    time_since_powerplay_started += time_since_last_event
                    
            if event['result']['event'] == 'Shot' or event['result']['event'] == 'Goal':
                x_coordinates = event['coordinates'].get('x')
                y_coordinates = event['coordinates'].get('y')
                attacking_team = event['team']['name']
                period = event['about']['period']
                period_time = event['about']['periodTime']
                time_left = 20*60 - period_time*60
                game_seconds = (period-1)*20*60 + period_time*60
                shot_type = event['result'].get('secondaryType')
                if previous_event is None:
                    time_since_last_event = 0
                    previous_event_type = None
                    previous_event_x_coordinates = 0
                    previous_event_y_coordinates = 0
                    previous_event_period_time = 0
                    previous_event_period = 0
                else:
                    previous_event_x_coordinates = previous_event['coordinates'].get('x')
                    previous_event_y_coordinates = previous_event['coordinates'].get('y')
                    previous_event_period_time = previous_event['about']['periodTime']
                    previous_event_period = previous_event['about']['period']
                    time_since_last_event = game_seconds - ((previous_event_period-1)*20*60 + previous_event_period_time*60)previous_event_type = previous_event['result']['event']
                distance_from_last_event = np.sqrt((x_coordinates**2 - previous_event_x_coordinates**2)+(y_coordinates**2 - previous_event_y_coordinates**2))
                speed = distance_from_last_event / time_since_last_event
                if previous_event['result']['event'] == 'Shot' or previous_event['result']['event'] == 'Goal':
                        rebound = 1
                    else: 
                        rebound = 0

                attributes.append([x_coordinates, y_coordinates,period,period_time,previous_event_type,previous_event_x_coordinates, previous_event_y_coordinates,previous_event_period_time,previous_event_period,rebound,home_team,away_team,shot_type,attacking_team,game_seconds,speed,distance_from_last_event,home_players,away_players,time_since_powerplay_started,time_left])
                
            if event['result']['event'] == 'Goal':
                if attacking_team == home_team:
                    if (len(penalties['away_minor2_penalty_stack'])>0 and len(penalties['away_minor4_penalty_stack'])>0) :
                        # when there is both a 2min minor and a 4min minor, we will assume the time is taken away from the penalty with the least time left
                        if(penalties['away_minor2_penalty_stack'][0] < penalties['away_minor4_penalty_stack'][0]):
                            penalties['away_minor2_penalty_stack'][0] = 0
                        else:
                            if penalties['away_minor4_penalty_stack'][0] > 120:
                                penalties['away_minor4_penalty_stack'][0] = 120
                            else:
                                penalties['away_minor4_penalty_stack'][0] = 0
                    elif len(penalties['away_minor2_penalty_stack'])>0:
                        penalties['away_minor2_penalty_stack'][0] = 0
                    elif len(penalties['away_minor4_penalty_stack'])>0:
                        if penalties['away_minor4_penalty_stack'][0] > 120:
                            penalties['away_minor4_penalty_stack'][0] = 120
                        else:
                            penalties['away_minor4_penalty_stack'][0] = 0
                else:
                    if (len(penalties['home_minor2_penalty_stack'])>0 and len(penalties['home_minor4_penalty_stack'])>0) :
                        # when there is both a 2min minor and a 4min minor, we will assume the time is taken away from the penalty with the least time left
                        if(penalties['home_minor2_penalty_stack'][0] < penalties['home_minor4_penalty_stack'][0]):
                            penalties['home_minor2_penalty_stack'][0] = 0
                        else:
                            if penalties['home_minor4_penalty_stack'][0] > 120:
                                penalties['home_minor4_penalty_stack'][0] = 120
                            else:
                                penalties['home_minor4_penalty_stack'][0] = 0
                    elif len(penalties['home_minor2_penalty_stack'])>0:
                        penalties['home_minor2_penalty_stack'][0] = 0
                    elif len(penalties['home_minor4_penalty_stack'])>0:
                        if penalties['home_minor4_penalty_stack'][0] > 120:
                            penalties['home_minor4_penalty_stack'][0] = 120
                        else:
                            penalties['home_minor4_penalty_stack'][0] = 0
            
            elif event['result']['event'] == 'Penalty':
                if event['result']['penaltySeverity'] == 'Minor':
                    if event['result']['penaltyMinutes'] == 2: 
                            penalties['home_minor2_penalty_stack'].append(120)
                        else:
                            penalties['home_minor4_penalty_stack'].append(240)
                    else:
                        penalties['home_major_penalty_stack'].append(300)

                else:
                    if event['result']['penaltySeverity'] == 'Minor':
                        if event['result']['penaltyMinutes'] == 2: 
                            penalties['away_minor2_penalty_stack'].append(120)
                        else:
                            penalties['away_minor4_penalty_stack'].append(240)
                    else:
                        penalties['away_major_penalty_stack'].append(300)
                        
            previous_event = event
        game = pd.DataFrame(attributes, columns=['x_coordinates, y_coordinates,period,period_time,previous_event_type,previous_event_x_coordinates, previous_event_y_coordinates,previous_event_period_time,previous_event_period,rebound,home_team,away_team,shot_type,attacking_team,game_seconds,speed,distance_from_last_event,home_players,away_players,time_since_powerplay_started'])
        return game
    
    def update_tracker(self):
        self.tracker = self.game.shape[0]
        
    def features_for_models(self,model):
        if model == 'lrd':
            temp = self.game['distance_from_net']
            return temp.loc[self.tracker:,:]
        
        if model == 'lrda':
            temp = self.game[['distance_from_net','angle_from_net']]
            return temp.loc[self.tracker:,:]
        
        if model == 'lda':
            temp = self.game['angle_from_net']
            return temp.loc[self.tracker:,:]
        
        if model == 'xgboost-tuning':
            feat_xgb = ['game_seconds', 'period', 'x_coordinates', 'y_coordinates','distance_from_net', 'angle_from_net', 
                        'previous_event_type','previous_event_x_coordinates', 'previous_event_y_coordinates',
                        'time_since_last_event', 'distance_from_last_event', 'rebound','change_in_angle', 'speed', 'time_since_powerplay_started', 
                        '5v5','4v4', '3v3', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4','shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Slap Shot',
                        'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around','shot_type_Wrist Shot']
                    
            le_name_mapping = {'Blocked Shot': 0, 'Faceoff': 1, 'Game Official': 2, 'Giveaway': 3, 'Goal': 4, 'Hit': 5, 'Missed Shot': 6, 'Official Challenge': 7, 'Penalty': 8, 'Period End': 9, 'Period Official': 10, 'Period Ready': 11, 'Period Start': 12, 'Shootout Complete': 13, 'Shot': 14, 'Stoppage': 15, 'Takeaway': 16}
    
            temp = self.game
            temp = self.geat_angle_change(temp)
            temp = self.get_penalties()
            temp = self.preprocess(temp[feat_xgb])
            temp['previous_event_type'] = temp['previous_event_type'].replace(le_name_mapping)
            return temp.loc[self.tracker:,:] 
            
    
    def preprocess(self,df):
        le = LabelEncoder()
        df["previous_event_type"] = le.fit_transform(df["previous_event_type"])
        return df

    def ping_game(self,model):
        self.game = self.update_events()
        self.game = self.get_distance()
        self.game = self.get_angle()
        df = self.features_for_models(model)
        self.update_tracker()
        tracker = self.tracker
        return df, tracker
    
    def get_distance(self):
        df = self.game
        list_distance = []
        for i in range(0,df.shape[0]):
            list_distance.append(min(np.sqrt((89-df.x_coordinates[i])**2+df.y_coordinates[i]**2),np.sqrt((-89-df.x_coordinates[i])**2+df.y_coordinates[i]**2)))
        df['distance_from_net'] = list_distance
        return df
            
    def get_angle(self):
        df = self.game
        list_angle = []
        for i in range(0,df.shape[0]):
            if df.y_coordinates[i] == 0:
                list_angle.append(0)
            elif df.y_coordinates[i] > 0:
                list_angle.append(min(np.arcsin(df.y_coordinates[i]/df.distance_from_net[i])*-180/math.pi,np.arcsin(df.y_coordinates[i]/df.distance_from_net[i])*180/math.pi))
            else:
                list_angle.append(min(np.arcsin(df.y_coordinates[i]/df.distance_from_net[i])*-180/math.pi,np.arcsin(df.y_coordinates[i]/df.distance_from_net[i])*180/math.pi))
        df['angle_from_net'] = list_angle
        return df
    
    def get_angle_change(self,df):
    df = df
    list_angle = []
    for index, row in df.iterrows():
        if row['previous_event_type']=='Shot':
            if df.attacking_team_side[index] == "right":
                if df.previous_event_y_coordinates[index] == 0:
                    list_angle.append(np.absolute(df.angle_from_net[index]))
                else:
                    distance_from_net = np.minimum(np.sqrt(df.previous_event_y_coordinates[index]**2+(df.previous_event_x_coordinates[index]+89)**2),np.sqrt(df.previous_event_y_coordinates[index]**2+(df.previous_event_x_coordinates[index]-89)**2))
                    angle = np.arcsin(df.previous_event_y_coordinates[index]/distance_from_net)*-180/math.pi
                    sign = np.sign([angle,df.angle_from_net[index]])
                    change_in_angle = 0
                    if sign[0]!=sign[1]:
                        change_in_angle = np.absolute(angle) + np.absolute(df.angle_from_net[index])
                    else:
                        change_in_angle = np.absolute(angle-df.angle_from_net[index])
                    list_angle.append(change_in_angle)
                    
            if df.attacking_team_side[index] == "left":
                if df.previous_event_y_coordinates[index] == 0:
                    list_angle.append(np.absolute(df.angle_from_net[index]))
                else:
                    distance_from_net = np.minimum(np.sqrt(df.previous_event_y_coordinates[index]**2+(df.previous_event_x_coordinates[index]+89)**2),np.sqrt(df.previous_event_y_coordinates[index]**2+(df.previous_event_x_coordinates[index]-89)**2))
                    angle = np.arcsin(df.previous_event_y_coordinates[index]/distance_from_net)*180/math.pi
                    sign = np.sign([angle,df.angle_from_net[index]])
                    change_in_angle = 0
                    if sign[0]!=sign[1]:
                        change_in_angle = np.absolute(angle) + np.absolute(df.angle_from_net[index])
                    else:
                        change_in_angle = np.absolute(angle-df.angle_from_net[index])
                    list_angle.append(change_in_angle)  
        else:
            list_angle.append(0)
            
    df['change_in_angle'] = list_angle
    df['change_in_angle'] = df['change_in_angle'].round(decimals=2)
    
    return df

    def get_penalties(self):
        df = self.game
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
        
        return self.game