import joblib
import pandas as pd
import numpy as np
from data_calculation import process_data, log_step
from pathlib import Path
import os

# Configure paths
base_path = Path(__file__).resolve().parent.parent  # Parent folder
model_folder = base_path / 'models'
data_folder = base_path / 'data'

# Load pre-trained model and scaler
model_path = model_folder / 'trained_model.pkl'
scaler_path = model_folder / 'scaler.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    log_step("Model and scaler loaded successfully.")
except Exception as e:
    log_step(f"Error loading model or scaler: {e}")
    raise

def load_data(input_file):
    """Load data for prediction."""
    file_path = data_folder / input_file
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        log_step(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        log_step(f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        log_step(f"Error loading data from {file_path}: {e}")
        raise

def calculate_features(data):
    """Ensure all required features are calculated."""
    # Assuming 'timeframe' is already detected in the input data, or it could be passed in
    timeframes = ["M5", "H1", "H4"]
    return process_data(data, timeframes)

def prepare_input_features(data):
    """Prepare the features for prediction."""
    # Calculate the necessary features
    data = calculate_features(data)
    
    # Select the required features for prediction
    feature_columns = ['EMA_50_M5', 'RSI_M5', 'MACD_M5', 'Bollinger_upper_M5', 'Bollinger_lower_M5']
    
    # Make sure the columns are present in the processed data
    if not all(col in data.columns for col in feature_columns):
        log_step(f"Missing required columns: {feature_columns}")
        raise ValueError("Missing required features for prediction.")
    
    X = data[feature_columns]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def predict(data):
    """Generate prediction using the model."""
    X_scaled = prepare_input_features(data)
    
    # Make prediction using the model
    try:
        prediction = model.predict(X_scaled)
        log_step(f"Prediction made: {prediction}")
        return prediction
    except Exception as e:
        log_step(f"Error during prediction: {e}")
        raise

def save_prediction(prediction, output_file="prediction.txt"):
    """Save the prediction to a file."""
    output_path = data_folder / output_file
    try:
        with open(output_path, 'w') as file:
            file.write(str(prediction))
        log_step(f"Prediction saved to {output_path}.")
    except Exception as e:
        log_step(f"Error saving prediction to {output_path}: {e}")

if __name__ == "__main__":
    input_file = "data.csv"  # This could be updated based on EA input or use other input methods
    try:
        data = load_data(input_file)
        prediction = predict(data)
        save_prediction(prediction)
    except Exception as e:
        log_step(f"Error during prediction process: {e}")
