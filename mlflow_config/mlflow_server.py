#%%
from pathlib import Path
import subprocess
import socket
import time
import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("MLFLOW_PORT"))
HOST = os.getenv("ML_FLOW_HOST")


BASE_DIR = Path(__file__).resolve().parents[1]

DATABASE = BASE_DIR / "mlflow_config" / "mlflow.db"

ARTIFACTS = BASE_DIR / "mlflow_config" / "mlartifacts"
print("*"*25+" MLFLOW CONFIGURACOES " + "*"*25)
print(PORT)
print(HOST)
print(BASE_DIR)
print(DATABASE)
print(ARTIFACTS)
print("*"*72)
#%%

def server_running():

    s = socket.socket()

    try:
        s.connect((HOST, PORT))
        return True

    except:

        return False

    finally:

        s.close()

def start_server():

    if server_running():
        return

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    subprocess.Popen(
        [
            "mlflow",
            "server",

            "--backend-store-uri",
            f"sqlite:///{DATABASE}",

            "--default-artifact-root",
            ARTIFACTS.resolve().as_uri(),

            "--host",
            HOST,

            "--port",
            str(PORT)
        ]
    )

    time.sleep(5)