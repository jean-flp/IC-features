from pathlib import Path
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv("MLFLOW_PORT")
HOST = os.getenv("ML_FLOW_HOST")
EXPERIMENT_NAME = os.getenv("ML_FLOW_EXPERIMENT_NAME")

BASE_DIR = Path(__file__).resolve().parents[1]


TRACKING_URI = f"http://{HOST}:{PORT}"

def configure_mlflow():

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)