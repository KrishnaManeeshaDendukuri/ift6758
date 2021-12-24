"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app
gunicorn can be installed via:
    $ pip install gunicorn
"""
import logging
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

import ift6758



app = Flask(__name__)
try:
    LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
    COMET_API_KEY = os.environ.get("COMET_API_KEY")
except Exception as e:
    print(e)
    app.logger.info("Environment variables not set")

def get_features(model_name):
    
    if model_name == 'kleitoun_lrd_1.0.0':
        features = ['distance_from_net']
        
    if model_name == 'kleitoun_lrda_1.0.0':
        features = ['distance_from_net', 'angle_from_net']
        
    if model_name == 'kleitoun_lra_1.0.0':
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
    
    if model_name == 'kleitoun_xgboost_1.0.0':
        features = ['distance_from_net', 'angle_from_net']


    
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

    if model_name == 'kleitoun_xgboost_1.0.0':
        os.rename(f'XGBoost_Baseline_model.pickle', model_name)



@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    from imp import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)
    
    # clear log file 
    with open(LOG_FILE, 'w'):
        pass

    # TODO: any other initialization before the first request (e.g. load default model)
    api = API(COMET_API_KEY)
    print(f"API key is set to {COMET_API_KEY}")
    
    # download default model if it is not already downloaded
    global model_name
    global model
    
    json = {'workspace': 'kleitoun', 'model': 'lrd', 'version': '1.0.0'}
    
    model_name = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
    model_file = Path(f"{model_name}")
    
    if not model_file.is_file():  
        api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
        # rename model file to the same name as in the api.get() call
        # this will allow us to detect if a model has already been downloaded by looking at the file name
        rename_model_file(model_name)
        model = pickle.load(open(model_name, 'rb'))
        app.logger.info(f'succesfully downloaded and loaded default model ({model_name})')
    else:
        model = pickle.load(open(model_name, 'rb'))
        app.logger.info(f'succesfully loaded default model ({model_name})')

@app.route("/health", methods=["GET"])
def health():
    return {'message': 'Healthy'} 

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
    global model
    
    current_model_name = model_name
    current_model = model
    
    new_model_name = f'{json["workspace"]}_{json["model"]}_{json["version"]}'
    model_file = Path(f"{new_model_name}")
    
    if model_file.is_file():
        
        model_name = new_model_name
        model = pickle.load(open(model_name, 'rb'))
        
        app.logger.info(f'successfully loaded model {model_name}')

    else:
        api = API(COMET_API_KEY)
        try:
            api.download_registry_model(json['workspace'], json['model'], json['version'], output_path="./", expand=True)
            rename_model_file(new_model_name)
            assert model_file.is_file()
            model_name = new_model_name
            model = pickle.load(open(model_name, 'rb'))
            app.logger.info(f'successfully downloaded and loaded model {model_name}')
            
        except:
            app.logger.warning(f'Failed to download model {new_model_name}. This is probably because this model does not exists, or because it is not recognised by the rename_model_file function. The application will keep using the currently loaded model ({current_model_name}).')
    
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
    
    app.logger.info(f'computing goal probabilities for provided events using model {model_name}')
    
    try:
        features = get_features(model_name)
        X = pd.DataFrame(json)[features]
        if model_name == 'kleitoun_xgboost-tuning_1.0.0':
            le_name_mapping = {
                'Blocked Shot': 0, 'Faceoff': 1, 'Game Official': 2, 'Giveaway': 3, 'Goal': 4, 
                'Hit': 5, 'Missed Shot': 6, 'Official Challenge': 7, 'Penalty': 8, 
                'Period End': 9, 'Period Official': 10, 'Period Ready': 11, 'Period Start': 12,
                'Shootout Complete': 13, 'Shot': 14, 'Stoppage': 15, 'Takeaway': 16
            }
            X["previous_event_type"] = X["previous_event_type"].replace(le_name_mapping)

        response = model.predict_proba(X)[:,1]
        app.logger.info(f'Finished computing goal probabilities.')
    except: 
        response=None
        app.logger.warning(f'Failed to compute predictions. This may be because the loaded model is not recognised by the function get_features.') 

    app.logger.info(response)
    return jsonify(response.tolist())  # response must be json serializable!


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)