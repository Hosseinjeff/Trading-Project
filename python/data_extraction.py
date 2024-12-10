# data_extraction.py				
import pandas as pd
from data_calculation import process_data  # Import INDICATOR_CONFIG
from utils import setup_logger, log_step
from pathlib import Path
import logging
import json
import os

logger = setup_logger('data_extraction')

# Configure paths
base_path = Path(__file__).resolve().parent.parent  # Parent folder
data_folder = base_path / 'data'
metadata_path = data_folder / 'features_metadata.json'
config_folder = Path(__file__).resolve().parent / 'configs'

with open(config_folder / 'feature_config.json', 'r') as f:
    FEATURE_CONFIG = json.load(f)

with open(config_folder / 'indicator_config.json', 'r') as f:
    INDICATOR_CONFIG = json.load(f)

def validate_features(data, expected_features):
    """Validate that all expected features are present."""
    missing_features = [feature for feature in expected_features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    log_step(logger, f"Feature validation passed for: {', '.join(expected_features)}")

def fetch_data(file_name):
    """Fetch data from a CSV file."""
    file_path = data_folder / file_name
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        log_step(logger, f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        log_step(logger, f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        log_step(logger, f"Error loading data from {file_path}: {e}")
        raise

def prepare_and_save_data(input_file, output_file):
    """Prepare data by detecting timeframe, processing indicators, and saving results."""
    try:
        # Step 1: Load the data
        log_step(logger, f"Attempting to load data from {input_file}.")
        data = fetch_data(input_file)

        # Step 2: Detect the timeframe
        timeframe = detect_timeframe(data)
        if timeframe not in FEATURE_CONFIG:
            raise ValueError(f"Unsupported timeframe detected: {timeframe}")
        log_step(logger, f"Detected timeframe: {timeframe}")

        # Step 3: Process the data for the detected timeframe
        log_step(logger, f"Processing indicators for {timeframe}.")
        processed_data = process_data(data, [timeframe], FEATURE_CONFIG, INDICATOR_CONFIG)

        # Step 4: Validate processed data
        log_step(logger, "Validating processed data.")
        validate_features(processed_data, FEATURE_CONFIG[timeframe])

        # Step 5: Save the processed data
        output_path = data_folder / output_file
        processed_data.to_csv(output_path)
        with open(metadata_path, 'w') as f:
            json.dump(FEATURE_CONFIG, f)
        log_step(logger, f"Processed data saved to {output_path}. Metadata saved to {metadata_path}.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during preparation and saving: {e}")
        raise
							   
def detect_timeframe(data):
    """Detect the timeframe of the data."""
									 
    time_diff = data.index.to_series().diff().dt.total_seconds().dropna().mode()[0]
    TIMEFRAME_MAPPING = {300: "M5", 3600: "H1", 14400: "H4"}
    return TIMEFRAME_MAPPING.get(time_diff, "Unknown")

if __name__ == "__main__":
    logger.info("Starting data extraction process.")
    input_file = "data.csv"
    output_file = "processed_data.csv"
    prepare_and_save_data(input_file, output_file)
