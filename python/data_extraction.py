import pandas as pd
import json
from data_calculation import process_data, validate_features  # Import process_data and validate_features
from utils import (
    setup_logger,
    log_step,
    data_folder,
    config_folder,
    features_metadata,
    indicator_config,
    log_file,
)
from pathlib import Path

logger = setup_logger('data_extraction', log_file=str(log_file))

# Load timeframes and indicators dynamically from the configuration
with open(indicator_config, 'r') as f:
    config = json.load(f)

def generate_feature_columns(timeframes, indicators):
    def generate_columns(indicator, params, tf):
        columns = []
        if indicator in ["EMA", "RSI"]:
            for period in params.get("periods", []):
                columns.append(f"{indicator}_{period}_{tf}")
        elif indicator == "MACD":
            for fast in params.get("fast_periods", []):
                for slow in params.get("slow_periods", []):
                    for signal in params.get("signal_periods", []):
                        columns.append(f"{indicator}_{fast}_{slow}_{signal}_{tf}")
        elif indicator == "Bollinger":
            for period in params.get("periods", []):
                for deviation in params.get("deviations", []):
                    columns.append(f"{indicator}_{period}_{deviation}_{tf}_Upper")
                    columns.append(f"{indicator}_{period}_{deviation}_{tf}_Middle")
                    columns.append(f"{indicator}_{period}_{deviation}_{tf}_Lower")
        return columns

    feature_columns = {}
    for tf in timeframes:
        feature_columns[tf] = [col for ind, params in indicators.items() for col in generate_columns(ind, params, tf)]
    return feature_columns

def fetch_data(file_name):
    """Fetch data from a CSV file."""
    file_path = data_folder / file_name
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        data.columns = data.columns.str.strip()  # Remove any leading/trailing spaces
        data = data.dropna(subset=['close'])
        log_step(logger, f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        log_step(logger, f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        log_step(logger, f"Error loading data from {file_path}: {e}")
        raise

def prepare_and_save_data(input_file, output_file):
    """Prepare data by processing indicators for specified or detected timeframes, and saving results."""
    try:
        # Step 1: Load the data
        log_step(logger, f"Attempting to load data from {input_file}.")
        data = fetch_data(input_file)

        # Step 2: Detect the timeframe
        detected_timeframe = detect_timeframe(data)
        log_step(logger, f"Detected timeframe: {detected_timeframe}.")

        # Step 3: Determine timeframes to process
        timeframes_to_process = config.get("timeframes", [])
        if not timeframes_to_process:
            timeframes_to_process = [detected_timeframe]  # Default to detected timeframe
        elif detected_timeframe not in timeframes_to_process:
            timeframes_to_process.append(detected_timeframe)

        log_step(logger, f"Timeframes to process: {', '.join(timeframes_to_process)}")

        # Step 4: Process the data for the selected timeframes
        log_step(logger, "Processing indicators.")
        processed_data = process_data(data, timeframes_to_process, feature_config, indicator_config)
        logger.info(f"Processed data sample:\n{processed_data.head()}")

        # Step 5: Validate processed data
        log_step(logger, "Validating processed data.")
        for timeframe in timeframes_to_process:
            validate_features(processed_data, feature_config[timeframe])

        # Step 6: Save the processed data
        output_path = data_folder / output_file
        processed_data.to_csv(output_path)
        with open(features_metadata, 'w') as f:
            json.dump({"features": feature_config, "timeframes": timeframes_to_process}, f)

        logger.info(f"Data preparation and saving completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during preparation and saving: {e}")
        raise

def detect_timeframe(data):
    """Detect the timeframe of the data."""
    time_diff = data.index.to_series().diff().dt.total_seconds().dropna().mode()[0]
    TIMEFRAME_MAPPING = {60: "M1", 300: "M5", 900: "M15", 3600: "H1", 1800: "M30", 14400: "H4", 86400: "D1"}
    detected_timeframe = TIMEFRAME_MAPPING.get(time_diff, "Unknown")
    log_step(logger, f"Detected timeframe: {detected_timeframe} based on time difference: {time_diff}")
    return detected_timeframe

def validate_indicator_config(indicator_config):
    required_keys = {"periods"}
    for indicator, params in indicator_config.items():
        if indicator == "MACD":
            required_keys = {"fast_periods", "slow_periods", "signal_periods"}
        elif indicator == "Bollinger":
            required_keys = {"periods", "deviations"}
        
        missing_keys = required_keys - params.keys()
        if missing_keys:
            raise ValueError(f"Missing keys {missing_keys} in configuration for {indicator}")
    log_step(logger, "Indicator configuration validated successfully.")

# Load timeframes and indicators
timeframes = config["timeframes"]
indicator_config = config["indicators"]
validate_indicator_config(indicator_config)

logger.info(f"Loaded timeframes: {timeframes}")
logger.info(f"Loaded indicator configurations: {indicator_config}")

feature_config = generate_feature_columns(timeframes, indicator_config)
logger.info(f"Generated feature columns: {feature_config}")

if __name__ == "__main__":
    logger.info("Starting data extraction process.")
    input_file = "data.csv"
    output_file = "processed_data.csv"
    prepare_and_save_data(input_file, output_file)
