# data_predict.py
import time
import os
import logging
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import traceback
import threading
import requests
import json
import datetime

# Server details (adjust the host and port based on where the server is running)
SERVER_URL = 'http://127.0.0.1:5000/calculate_indicators'


# Flask App Initialization
app = Flask(__name__)

# Logger setup
logger = logging.getLogger("PredictServer")
logger.setLevel(logging.INFO)

# Paths setup
current_directory = Path(__file__).resolve().parent
sister_directory = current_directory.parent / "models"

# Paths to model, scaler, and flag file
model_path = sister_directory / "trading_model.pkl"
scaler_path = sister_directory / "scaler.pkl"
flag_file_path = (
    "C:\\Users\\Setin\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Files\\ea_flag_file.txt"
)

# File Handler for Logging
log_file =  "C:\\Users\\Setin\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Files\\predict_server.log"
try:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
except PermissionError as e:
    logger.error(f"Failed to set up file handler: {e}. Logging to console only.")

# Console Logging
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Load the model
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    logger.info(f"Model loaded successfully from {model_path}")
else:
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the scaler
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    logger.info(f"Scaler loaded successfully from {scaler_path}")
    logger.info(f"Expected feature names: {scaler.feature_names_in_}")
else:
    logger.error(f"Scaler file not found at {scaler_path}")
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

# Endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive raw request body data for inspection
        raw_data = request.data  # This will give the raw byte string
        logger.info(f"Raw request body: {raw_data}")

        # Try to decode the incoming JSON request data
        data = request.json  # This will automatically parse the JSON body
        logger.info(f"Decoded JSON data: {data}")

        # Validate input
        if "features" not in data or not isinstance(data["features"], list):
            return jsonify({"error": 'Invalid input. "features" must be a list of numeric values.'}), 400
        
        # Check if the number of features matches the model's expected input
        if len(data["features"]) != len(scaler.feature_names_in_):
            return jsonify({
                "error": f"Expected {len(scaler.feature_names_in_)} features, but received {len(data['features'])} features."
            }), 400

        # Convert to pandas DataFrame with correct column names
        column_names = scaler.feature_names_in_
        features = pd.DataFrame([data["features"]], columns=column_names)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Return the prediction
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        error_message = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"error": error_message}), 500

# Periodic check to see if the EA is still running
def check_ea_running():
    """Periodically check if the EA is still running based on the flag file."""
    try:
        # Allow the server to start even if the flag file is missing initially
        initial_run = True
        while True:
            if not os.path.exists(flag_file_path):  # Check if the flag file is deleted
                if initial_run:
                    logger.info("EA flag file not found at startup. Assuming EA will create it later.")
                    initial_run = False  # Set this to False after the first iteration
                else:
                    logger.info("EA has stopped, shutting down Python server.")
                    os._exit(0)  # Exit the process to stop the server
            else:
                initial_run = False  # Flag file exists, clear the initial condition
            time.sleep(10)  # Check every 10 seconds
    except Exception as e:
        logger.error(f"Error checking EA status: {e}")


# Endpoint for shutting down the server manually
@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shuts down the Flask server."""
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        logger.error("Shutdown not supported. Not running with the Werkzeug server.")
        raise RuntimeError("Not running with the Werkzeug Server")
    logger.info("Shutdown initiated manually via /shutdown endpoint.")
    func()
    return "Server shutting down..."


# Main entry point
if __name__ == "__main__":
    # Start the periodic EA status check in a separate thread
    threading.Thread(target=check_ea_running, daemon=True).start()

    # Start the Flask server
    logger.info("Starting the Flask server.")
    app.run(port=5000, debug=True)
