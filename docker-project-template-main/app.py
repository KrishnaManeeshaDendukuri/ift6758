"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app
gunicorn can be installed via:
    $ pip install gunicorn
"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import numpy as np
import joblib
from comet_ml import Experiment
from comet_ml import API
import pickle
import xgboost


import ift6758


#LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
LOG_FILE = "flask.log"
key = "PeZE4vLrXZR7BuQTNrIYRlRF9"


app = Flask(__name__)

def get_features(model_name):
    
    if model_name == 'kleitoun_lrd_1.0.0':
        features = ['distance_from_net']
        
    if model_name == 'kleitoun_lrda_1.0.0':
        features = ['distance_from_net', 'angle_from_net']
        
    if model_name == 'kleitoun_lda_1.0.0':
        features = ['angle_from_net']
        
    if model_name == 'kleitoun_xgboost-tuning_1.0.0':
        features = ['game_seconds', 'period', 'x_coordinates', 'y_coordinates',
           'distance_from_net', 'angle_from_net', 'previous_event_type',
           'previous_event_x_coordinates', 'previous_event_y_coordinates',
           'time_since_last_event', 'distance_from_last_event', 'rebound',
           'change_in_angle', 'speed', 'time_since_powerplay_started', '5v5',
           '4v4', '3v3', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4',
           'shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Slap Shot',
           'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around',
           'shot_type_Wrist Shot'
        ]
        
    if model_name == 'chief_moth_9230_1.0.0':
        features = ['home_players', 'away_players', 'time_since_powerplay_started', 
            'distance_from_net', 'angle_from_net', 'game_seconds',
            'previous_event_game_seconds', 'time_since_last_event', 'distance_from_last_event',
            'rebound', 'rebound_same_team', 'home_team_attacking', 'overtime', 'speed', '5v5', 
            '4v4', '3v3', '5v4', '5v3', '4v3', '4v5', '3v5', '3v4', 'power_play',
            'penalty_kill', 'change_in_angle', 'shot_type_Backhand', 'shot_type_Deflected',
            'shot_type_Slap Shot', 'shot_type_Snap Shot', 'shot_type_Tip-In',
            'shot_type_Wrap-around'
        ]
    
    return features

def rename_model_file(model_name):
    
    if model_name == 'kleitoun_lrd_1.0.0':
        os.rename(f'LogisticRegression_distance', model_name)
        
    if model_name == 'kleitoun_lrda_1.0.0':
        os.rename(f'LogisticRegression_distance+angle', model_name)
        
    if model_name == 'kleitoun_lra_1.0.0':
        os.rename(f'LogisticRegression_angle', model_name)
    
    if model_name == 'kleitoun_xgboost-tuning_1.0.0':
        os.rename(f'XGBoost_hmtuning_model_v2.pickle', model_name)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    
    # clear log file 
    with open(LOG_FILE, 'w'):
        pass

    # TODO: any other initialization before the first request (e.g. load default model)
    api = API(key)
    
    # download default model if it is not already downloaded
    json = {'workspace': 'kleitoun', 'model': 'lrd', 'version': '1.0.0'}
    global model_name
    model_name = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
    model_file = Path(f"{model_name}")
    
    if not model_file.is_file():  
        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
        # rename model file to the same name as in the api.get() call
        # this will allow us to detect if a model has already been downloaded by looking at the file name
        rename_model_file(model_name)
        
    global model
    model = pickle.load(open(model_name, 'rb'))

    app.logger.info('succesfully loaded default model')
    pass


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    with open(LOG_FILE) as f:
        response = f.readlines()
        
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    Recommend (but not required) json with the schema:
        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO: check to see if the model you are querying for is already downloaded

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    global model_name
    model_name = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
    model_file = Path(f"{model_name}")
    
    if not model_file.is_file():
        api = API(key)
        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
        rename_model_file(model_name)
        
    global model
    model = pickle.load(open(model_name, 'rb'))
    
    response = None

    app.logger.info(response)
    return jsonify(response) 


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # TODO:
    

    features = get_features(model_name)
    #X = np.array([[json[feature] for feature in features]])
    X = pd.DataFrame(json)[features]
    response = model.predict_proba(X)

    app.logger.info(response)
    return jsonify(response.tolist())  # response must be json serializable!

if __name__ == "__main__":
    app.run(debug=True)