import pandas as pd
import numpy as np
import requests
import json
import os

data_folder_path = './'


def get_file(game_id, folder_path):
    """
    Check if file is already downloaded.
        If yes, return the file path
        If not, check if it exists on the nhl api.
            If it does, save it locally and return the path
            If it doesn't, return None
    """

    file_path = folder_path + str(game_id) + '.json'
    if not os.path.isfile(file_path):
        data = requests.get(f'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/')
        if (data.status_code == 404):
            return None
        with open(file_path, 'w') as f:
            json.dump(data.json(), f)

    return file_path


def get_season_files(seasons = [2016, 2017, 2018, 2019, 2020], folder_path = data_folder_path, clear_existing_files = False):

    """download all regular season + playoff files from specified seasons into directory.
     Combine into one file, which is returned and saved at top of directory

     Note: might have to do one year at a time due to memory limitations"""

    if clear_existing_files:
        filelist = [ f for f in os.listdir(folder_path)  if f.endswith('.json') ]
        for f in filelist:
            os.remove(os.path.join(folder_path, f))

    result = list()

    for season in seasons:

        # get regular season games
        file_exists = True
        game_id = season * 1000000 + 20000 + 1 #id of first game in reg season
        while(file_exists):
            file = get_file(game_id, folder_path)
            if file is not None:
                with open(file, 'r') as f:
                    result.append(json.load(f))
            else:
                file_exists = False
            game_id +=1


        # get playoff games

        if season == 2019: # in 2019-2020 there was a playoff round 0 due to covid
            first_round = 0
        else:
            first_round = 1

        for playoff_round in range(first_round, 5):
            for matchup in range(1, 2**(4-playoff_round-1*(playoff_round==0))+1):
                file_exists = True
                game_id = season * 1000000 + 30000 + playoff_round * 100 + matchup * 10 + 1   # id of first game in playoff round & matchup
                while(file_exists):
                    file = get_file(game_id, folder_path)
                    if file is not None:
                        with open(file, 'r') as f:
                            result.append(json.load(f))
                    else:
                        file_exists = False
                    game_id +=1

    output_file_name = '0_all_games_' + "".join([str(season) for season in seasons]) + '.json'
    with open(folder_path + output_file_name, 'w') as f:
            json.dump(result, f)

    return result


def get_player_name(event, role):

    assert role in ['Goalie','Shooter','Scorer'], "selected role not supported"
    for player in event['players']:
        if (player['playerType'] == role):
            return player['player']['fullName']

def return_list_value_if_exists(lst, pos):
    if len(lst) > pos :
        return lst[pos]
    else:
        return None


def get_distance_from_net_and_side(df, net_distance_from_center = 89):

    """
    Compute the distance from where a shot was taken to the goal line



    source: https://community.rapidminer.com/discussion/44904/using-the-nhl-api-to-analyze-pro-ice-hockey-data-part-1
    """

    df['coordinates_multiplier'] = 1 -\
                                   2 * (df['home_team_side_1st_period']=='right') * (((df['attacking_team'] == df['home_team']) & df['period'].isin([1,3,5,7,9])) | ((df['attacking_team'] == df['away_team']) & df['period'].isin([2,4,6,8]))) -\
                                   2 * (df['home_team_side_1st_period']=='left') * (((df['attacking_team'] == df['home_team']) & df['period'].isin([2,4,6,8])) | ((df['attacking_team'] == df['away_team']) & df['period'].isin([1,3,5,7,9])))
    df['distance_from_net'] = df['home_team_side_1st_period'].isnull() * \
                                  np.minimum(
                                             ((df['x_coordinates'] - net_distance_from_center)**2 + (df['y_coordinates'] - 0)**2)**0.5,
                                             ((df['x_coordinates'] + net_distance_from_center)**2 + (df['y_coordinates'] - 0)**2)**0.5
                                  ) + \
                              ~df['home_team_side_1st_period'].isnull() * \
                                  ((df['x_coordinates'] - net_distance_from_center * df['coordinates_multiplier'])**2 + (df['y_coordinates'] - 0)**2)**0.5
    df['attacking_team_side'] = np.where(df['coordinates_multiplier'] == 1, 'left', 'right')
    del df['coordinates_multiplier']

    return df


def return_tidy_df(games, clear_games = True):

    data = []

    for game in games:

        game_id = game['gameData']['game']['pk']
        season = game['gameData']['game']['season']
        season_type = game['gameData']['game']['type']
        home_team = game['gameData']['teams']['home']['name']
        away_team = game['gameData']['teams']['away']['name']
        d = return_list_value_if_exists(game['liveData']['linescore']['periods'], 0)
        if d is not None:
            home_team_side_1st_period =  d['home'].get('rinkSide')
        else:
            home_team_side_1st_period = None

        events = game['liveData']['plays']['allPlays']

        for event in events:

            event_type = event['result']['event']

            if event_type == 'Shot':
                event_id = event['about']['eventIdx']
                attacking_team = event['team']['name']
                attacking_player = get_player_name(event, 'Shooter')
                goalie = get_player_name(event, 'Goalie')
                period = event['about']['period']
                period_time = event['about']['periodTime']
                goal_ind = 0
                shot_ind = 1
                x_coordinates = event['coordinates'].get('x')
                y_coordinates = event['coordinates'].get('y')
                shot_type = event['result'].get('secondaryType')
                empty_net = None
                strength = None
                gwg = None

                data.append([game_id, season, season_type, event_id, home_team, away_team, home_team_side_1st_period,
                             attacking_team, attacking_player, goalie, period, period_time, goal_ind,
                             shot_ind, x_coordinates, y_coordinates, shot_type, empty_net, strength, gwg])

            elif event_type == 'Goal':
                event_id = event['about']['eventIdx']
                attacking_team = event['team']['name']
                attacking_player = get_player_name(event, 'Scorer')
                goalie = get_player_name(event, 'Goalie')
                period = event['about']['period']
                period_time = event['about']['periodTime']
                goal_ind = 1
                shot_ind = 0
                x_coordinates = event['coordinates'].get('x')
                y_coordinates = event['coordinates'].get('y')
                shot_type = event['result'].get('secondaryType')
                empty_net = event['result'].get('emptyNet')
                strength = event['result']['strength']['name']
                gwg = event['result']['gameWinningGoal']

                data.append([game_id, season, season_type, event_id, home_team, away_team, home_team_side_1st_period,
                             attacking_team, attacking_player, goalie, period, period_time, goal_ind,
                             shot_ind, x_coordinates, y_coordinates, shot_type, empty_net, strength, gwg])

    df = pd.DataFrame(data,
                      columns=['game_id', 'season', 'season_type', 'event_id', 'home_team', 'away_team', 'home_team_side_1st_period',
                               'attacking_team', 'attacking_player', 'goalie', 'period', 'period_time','goal_ind',
                               'shot_ind', 'x_coordinates', 'y_coordinates', 'shot_type', 'empty_net', 'strength', 'gwg'])

    df = get_distance_from_net_and_side(df, net_distance_from_center = 89)

    if clear_games:
        del games

    return df


def tidy_df_loop(seasons = [2016, 2017, 2018, 2019, 2020]):
    """loop over all season. for each season, get file, create tidy df from it, and clear the memory.
       append the dfs together"""
    i = 0
    for season in seasons:
        data_files = get_season_files(seasons = [season])

        if i == 0:
            df = return_tidy_df(data_files)
        else:
            sub_df = return_tidy_df(data_files)
            df = pd.concat([df, sub_df])
            del sub_df
        i += 1
        print(f"finished {season} season")

    return df
