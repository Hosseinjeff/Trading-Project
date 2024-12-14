# predict.py
import joblib
import pandas as pd
import json
import os
import sys
from pathlib import Path
from utils import realtime, features_metadata, model_folder, EA_folder, setup_logger, log_step, features_metadata, processed_data_path, post_processed_data_path,load_model_related_files

from data_calculation import process_data

# Set up logger
logger = setup_logger("predict")

def generate_mock_data(file_path):
    """Generate mock data if the file doesn't exist."""
    log_step(logger, "Generating mock data for testing...")
    data = pd.DataFrame({
        'Time': pd.date_range(start="2024-12-01", periods=10, freq="H"),
        'Open': [1.2] * 10,
        'High': [1.25] * 10,
        'Low': [1.15] * 10,
        'Close': [1.22] * 10
    })
    data.set_index('Time', inplace=True)
    data.to_csv(file_path)
    log_step(logger, f"Mock data saved to {file_path}")

def load_data(file):
    """Load data for prediction."""
    file_path = EA_folder / file
    if not file_path.exists():
        generate_mock_data(file_path)

    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        log_step(logger, f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        log_step(logger, str(e))
        raise
    except pd.errors.EmptyDataError:
        log_step(logger, f"File is empty: {file_path}")
        raise
    except Exception as e:
        log_step(logger, f"Error loading data: {e}")
        raise

def prepare_features(data, model_metadata):
    """Ensure data matches model-trained features."""
    # Calculate features dynamically
    timeframes = model_metadata.get('timeframes', [])
    processed_data = process_data(data, timeframes, features_metadata, features_metadata)

    # Retain only required features
    required_features = model_metadata.get('features', [])
    missing_features = [feat for feat in required_features if feat not in processed_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    return processed_data[required_features]

def predict(file = realtime):
    """Generate predictions using the loaded model."""
    try:
        # Load latest model, scaler, and features metadata
        latest_model_path = max(model_folder.glob("best_trained_model_*.pkl"), key=os.path.getmtime)
        scaler_path, features_path = load_model_related_files(latest_model_path)

        # Load model, scaler, and features metadata
        model = joblib.load(latest_model_path)
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            model_metadata = json.load(f)

        # Log metadata
        log_step(logger, f"Model loaded from {latest_model_path}")
        log_step(logger, f"Scaler loaded from {scaler_path}")
        log_step(logger, f"Features metadata loaded from {features_path}")
        log_step(logger, f"Model info: {model_metadata['model_name']}")
        log_step(logger, f"Training timestamp: {model_metadata.get('training_timestamp')}")
        log_step(logger, f"Model RMSE: {model_metadata['metrics']['rmse']}")

        # Load and prepare input data
        data = load_data(file)
        features = prepare_features(data, model_metadata)

        # Scale features
        scaled_features = scaler.transform(features)

        # Generate predictions
        predictions = model.predict(scaled_features)

        # Save predictions
        predictions_df = pd.DataFrame({
            'Time': data.index,
            'Predicted_Close': predictions
        })
        predictions_df.to_csv(realtime, index=False)
        log_step(logger, f"Predictions saved to {realtime}")
        print("Prediction completed successfully.")

    except Exception as e:
        log_step(logger, f"Error during prediction: {e}")
        print(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    predict()
