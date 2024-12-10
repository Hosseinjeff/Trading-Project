# data_calculation.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path  # Import Path from pathlib
from utils import setup_logger, log_step
import json
import os

# Use the shared setup_logger
logger = setup_logger('data_calculation')

# Indicator calculation functions (unchanged)
def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period, column_name):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data[column_name] = 100 - (100 / (1 + rs))
    return data

def calculate_macd(data, fast_period, slow_period, signal_period):
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

import pandas as pd

def calculate_bollinger_bands(data, period=20, multiplier=2, timeframe="M5"):
    if 'close' not in data.columns:
        raise ValueError("Missing 'close' column in the dataset.")

    # Calculate the Bollinger Bands
    middle_band = data['close'].rolling(window=period).mean()
    std_dev = data['close'].rolling(window=period).std()
    upper_band = middle_band + (multiplier * std_dev)
    lower_band = middle_band - (multiplier * std_dev)
    
    # Dynamically assign column names
    data[f"Bollinger_{period}_{multiplier}_{timeframe}_Upper"] = upper_band
    data[f"Bollinger_{period}_{multiplier}_{timeframe}_Middle"] = middle_band
    data[f"Bollinger_{period}_{multiplier}_{timeframe}_Lower"] = lower_band

    return data
# Validate features based on dynamic columns
def validate_features(data, expected_features):
    missing_features = [feature for feature in expected_features if feature not in data.columns]
    if missing_features:
        for feature in missing_features:
            logger.warning(f"Feature '{feature}' is missing. Ensure indicator calculations are correct.")
        raise ValueError(f"Validation failed: Missing features - {missing_features}")
    else:
        log_step(logger, "Feature validation passed.")

def calculate_indicators(data, timeframe, feature_config, indicator_config):
    log_step(logger, f"Calculating indicators for {timeframe}.")
    logger.debug(f"Initial data snapshot: {data.head()}")

    for feature in feature_config[timeframe]:
        try:
            if feature.startswith("EMA"):
                period = int(feature.split("_")[1])
                data[feature] = calculate_ema(data, period)
            elif feature.startswith("RSI"):
                period = int(feature.split("_")[1])
                data = calculate_rsi(data, period, feature)
            elif feature.startswith("MACD"):
                params = map(int, feature.split("_")[1:4])
                macd, macd_signal, hist = calculate_macd(data, *params)
                if feature.endswith("_signal"):
                    data[feature] = macd_signal
                elif feature.endswith("_histogram"):
                    data[feature] = hist
                else:
                    data[feature] = macd
            elif feature.startswith("Bollinger"):
                params = list(map(int, feature.split("_")[1:3]))
                # Ensure column names are correct
                if 'close' in data.columns:
                    upper, middle, lower = calculate_bollinger_bands(data, *params)
                    if feature.endswith("_upper"):
                        data[feature] = upper
                    elif feature.endswith("_middle"):
                        data[feature] = middle
                    elif feature.endswith("_lower"):
                        data[feature] = lower
                else:
                    logger.error(f"Missing 'close' column for Bollinger Bands calculation.")
                    raise ValueError("Critical calculation error for Bollinger Bands due to missing 'close' column.")
        except Exception as e:
            logger.error(f"Failed to calculate {feature}: {e}")
            if feature.startswith("MACD") or feature.startswith("Bollinger"):
                raise ValueError(f"Critical calculation error for {feature}.")
            continue

    logger.debug(f"Completed indicators for {timeframe}.")
    return data

# Process data for multiple timeframes dynamically
def process_data(data, timeframes, feature_config, indicator_config):
    log_step(logger, f"Processing data for timeframes: {timeframes}.")
    processed_data = data.copy()

    for timeframe in timeframes:
        log_step(logger, f"Processing {timeframe}.")
        try:
            processed_data = calculate_indicators(processed_data, timeframe, feature_config, indicator_config)
            validate_features(processed_data, feature_config[timeframe], timeframe)
        except ValueError as ve:
            logger.error(f"Validation failed for {timeframe}: {ve}")
        except Exception as e:
            logger.error(f"Unexpected error during {timeframe} processing: {e}")

    logger.info(f"Completed processing for timeframes: {timeframes}.")
    return processed_data

def validate_config(config):
    required_keys = {"timeframes", "indicators"}
    if not all(key in config for key in required_keys):
        raise ValueError(f"Config missing required keys: {required_keys}")
    log_step(logger, "Configuration validated successfully.")
