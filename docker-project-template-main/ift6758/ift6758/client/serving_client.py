import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)

# df = pd.read_csv('../ift6758-project-template-main/notebooks/final_df.csv')

class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        logger.info(f"Initializing request to generate predictions")
        try:
            r = requests.post(
                f"{self.base_url}/predict", 
                json=json.loads(X.to_json())
            )
            logger.info(f"Successfully generated predictions")
            return r.json()
        except Exception as e:
            print(e)
            return None
        # raise NotImplementedError("TODO: implement this function")
        # logegr.info()

    def logs(self) -> dict:
        """Get server logs"""
        logger.info(f"Initializing request to server get logs")
        r = requests.get(
            f"{self.base_url}/logs"
        )
        logger.info(f"Server Logs fetched")
        return r.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        logger.info(f"Initializing request to download the model{model}-{version}")
        self.workspace = workspace
        self.model = model
        self.version = version
        self.model_filename = f"{workspace}_{model}_{version}"
        r = requests.post(
            f"{self.base_url}/download_registry_model", 
            json= {'workspace': workspace, 'model': model, 'version': version}
        )
        logger.info(f"Successfully Downloaded Model")
        # return r.json()

