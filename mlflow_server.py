import os
import subprocess
import time
import webbrowser
from pathlib import Path
import mlflow
import mlflow.sklearn

# Set Tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def start_mlflow_server():
    # Define backend and artifact paths
    backend_store_uri = "sqlite:///mlflow.db"
    artifact_root_path = Path("artifacts").absolute().as_uri()

    # Start MLflow server with specified backend and artifact root
    server_process = subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", artifact_root_path,
            "--host", "0.0.0.0",
            "--port", "5000"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to start
    time.sleep(3)

    # Open MLflow UI in browser
    webbrowser.open("http://localhost:5000")

    print(f"MLflow server started at http://localhost:5000")
    print("Press Ctrl+C to stop the server")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    start_mlflow_server()
