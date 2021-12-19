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
        self.game = pd.DataFrame(columns=['x_coordinates', 'y_coordinates', 'previous_event_x_coordinates', 'previous_event_y_coordinates'])
        
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
        file_path = './' + str(self.game_id) + '.json'
        game = []
        with open(file_path,'r') as f:
            events = json.load(f)
        events = events['liveData']['plays']['allPlays']
        previous_event = None
        for event in events:
            if event['result']['event'] == 'Shot' or event['result']['event'] == 'Goal':
                x_coordinates = event['coordinates'].get('x')
                y_coordinates = event['coordinates'].get('y')
                if previous_event == None:
                    previous_event_x_coordinates = 0
                    previous_event_y_coordinates = 0
                else:
                    previous_event_x_coordinates = previous_event['coordinates'].get('x')
                    previous_event_y_coordinates = previous_event['coordinates'].get('y')
                game.append([x_coordinates, y_coordinates,previous_event_x_coordinates, previous_event_y_coordinates])
                previous_event = event
        game = pd.DataFrame(game, columns=['x_coordinates', 'y_coordinates', 'previous_event_x_coordinates', 'previous_event_y_coordinates'])
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