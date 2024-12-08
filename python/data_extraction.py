# data_extraction.py
import pandas as pd
from data_calculation import process_data
import logging
from pathlib import Path  # Import Path from pathlib
import numpy as np
import os

# Configure paths
base_path = Path(__file__).resolve().parent.parent  # Parent folder
data_folder = base_path / 'data'

# Create or get a logger
logger = logging.getLogger()

# Set up logging to a file in the parent directory
log_file = Path(__file__).resolve().parent.parent / 'log.txt'
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter(f'%(asctime)s - {Path(__file__).name} - %(message)s'))
logger.addHandler(handler)

logger.setLevel(logging.INFO)  # You can change the logging level here

def log_step(message, script_name=None):
    """Log the step with the provided script name."""
    if script_name is None:
        script_name = os.path.basename(__file__)  # Default to current script if no name is provided
    logger.info(f'{script_name} - {message}')  # Use logger instead of logging

def detect_timeframe(data):
    """Detect the timeframe of the data by analyzing time differences."""
    log_step("Detecting data timeframe.")
    time_diff = data.index.to_series().diff().dt.total_seconds().dropna().mode()[0]
    if time_diff == 60 * 5:
        return "M5"
    elif time_diff == 60 * 60:
        return "H1"
    elif time_diff == 60 * 240:
        return "H4"
    else:
        raise ValueError("Unsupported timeframe detected.")

def fetch_data(file_name):
    """Fetch data from a CSV file."""
    file_path = data_folder / file_name
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

def prepare_and_save_data(input_file, output_file):
    """Prepare data by detecting timeframe, requesting indicator calculation, and saving results."""
    log_step(f"Preparing data from {input_file}.")
    data = fetch_data(input_file)
    try:
        timeframe = detect_timeframe(data)
        symbol = "EURUSD"  # Default symbol
        log_step(f"Detected timeframe: {timeframe}. Presumed symbol: {symbol}.")
        
        # Process data for multiple timeframes (current, H1, H4)
        timeframes = [timeframe, "H1", "H4"]
        processed_data = process_data(data, timeframes)
        
        # Save processed data
        output_path = data_folder / output_file
        processed_data.to_csv(output_path)
        log_step(f"Processed data saved to {output_path}.")
    except Exception as e:
        log_step(f"Error in data preparation: {e}")

if __name__ == "__main__":
    input_file = "data.csv"
    output_file = "processed_data.csv"
    prepare_and_save_data(input_file, output_file)
