import time
import os
import logging
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import traceback

# Create or get a logger
logger = logging.getLogger()

# Set up logging to a file if needed
handler = logging.FileHandler('logs.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.setLevel(logging.INFO)  # You can change the logging level here

# Set up logging to write to a file
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Get the current directory of the Python script
current_directory = Path(__file__).resolve().parent

# Get the path of the sister directory (assuming sibling directory is one level up)
sister_directory = current_directory.parent / "models"

# Paths to the model, scaler, and flag file
model_path = sister_directory / "trading_model.pkl"
scaler_path =  sister_directory / "scaler.pkl"
flag_file_path = "C:\\Users\\Setin\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Files\\ea_flag_file.txt"  # Flag file to track EA status

# Load the model
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the scaler
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

# Endpoint for shutting down the server
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shuts down the Flask server."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()
    return "Server shutting down..."

# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data
        data = request.json
        logging.info(f"Received data: {data}")  # Logs the data to the file

        print("Received data:", data)

        # Validate input
        if 'features' not in data or not isinstance(data['features'], list):
            return jsonify({'error': 'Invalid input. "features" must be a list of numeric values.'}), 400

        # Convert to pandas DataFrame with correct column names
        column_names = scaler.feature_names_in_  # Ensure columns match those used during training
        features = pd.DataFrame([data['features']], columns=column_names)
        print("Features after reshaping:", features)

        # Scale the features
        features_scaled = scaler.transform(features)
        print("Scaled features:", features_scaled)

        # Make prediction
        prediction = model.predict(features_scaled)
        print("Prediction result:", prediction)

        # Return the prediction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        error_message = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)
        return jsonify({'error': error_message}), 500


# Periodic check to see if the EA is still running
def check_ea_running():
    try:
        while True:
            if not os.path.exists(flag_file_path):  # Check if the flag file is deleted
                logging.info("EA has stopped, shutting down Python server.")
                break  # Break the loop to stop the server
            time.sleep(10)  # Check every 10 seconds
    except Exception as e:
        logging.error(f"Error checking EA status: {e}")


# Main entry point
if __name__ == '__main__':
    # Start the periodic check in a separate thread to avoid blocking the main server
    import threading
    threading.Thread(target=check_ea_running, daemon=True).start()

    app.run(port=5000, debug=True)
