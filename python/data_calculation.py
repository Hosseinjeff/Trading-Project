# data_calculation.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path  # Import Path from pathlib
from utils import setup_logger, log_step
import os
import json

# Use the shared setup_logger
logger = setup_logger('data_calculation')

config_folder = Path(__file__).resolve().parent / 'configs'
with open(config_folder / 'feature_config.json', 'r') as f:
    FEATURE_CONFIG = json.load(f)
with open(config_folder / 'indicator_config.json', 'r') as f:
    INDICATOR_CONFIG = json.load(f)

def calculate_ema(data, period):
    """Calculate EMA."""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period, column_name):
    """
    Calculate the RSI (Relative Strength Index) for the given data.
    Args:
        data (pd.DataFrame): The input data containing price information.
        period (int): The period to calculate RSI.
        column_name (str): The name of the resulting RSI column.
    Returns:
        pd.DataFrame: The input data with an added column for RSI.
    """
    try:
        logger.info(f"Calculating RSI for {len(data)} data points with period: {period}")
        logger.debug(f"First few rows of close data: {data[['close']].head()}")
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[column_name] = 100 - (100 / (1 + rs))
        logger.debug(f"RSI calculation for first few values: {data[column_name].head()}")
        missing_rsi_count = data[column_name].isna().sum()
        logger.debug(f"Number of missing values in {column_name}: {missing_rsi_count}")
    except Exception as e:
        logger.error(f"Error calculating RSI for {column_name}: {str(e)}")
        raise
    return data

def calculate_macd(data, fast_period, slow_period, signal_period):
    """Calculate MACD."""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line
												 

def calculate_bollinger_bands(data, period, num_std_dev):
    """Calculate Bollinger Bands."""
    sma = data['close'].rolling(window=period).mean()
    std_dev = data['close'].rolling(window=period).std()
											  
    return sma + (std_dev * num_std_dev), sma - (std_dev * num_std_dev)
								 

def validate_features(processed_data, expected_features, timeframe):
    """
    Validate that the processed data contains all expected features for a given timeframe.
    Args:
        processed_data (pd.DataFrame): The processed data containing calculated features.
        expected_features (list): A list of features expected for the timeframe.
        timeframe (str): The timeframe being validated.
    Raises:
        ValueError: If any expected feature is missing.
    """
    missing_features = [feature for feature in expected_features if feature not in processed_data.columns]
    if missing_features:
        logger.error(f"Missing features for {timeframe}: {missing_features}")
        logger.debug(f"Available columns: {processed_data.columns.tolist()}")
        raise ValueError(f"Missing features for {timeframe}: {missing_features}")
    else:
        logger.info(f"Feature validation passed for {timeframe}")

def calculate_indicators(data, timeframe, feature_config, indicator_config):
    """Calculate indicators dynamically based on feature configuration."""
    log_step(logger, f"Calculating indicators for {timeframe}.")
    logger.debug(f"Initial data snapshot: {data.head()}")

    for feature in feature_config:
        logger.debug(f"Processing feature: {feature}")
        try:
            if feature.startswith('EMA'):
                period = indicator_config['EMA']['period']
                data[feature] = calculate_ema(data, period)
            elif feature.startswith('RSI'):
                period = indicator_config['RSI']['period']
                data[feature] = calculate_rsi(data, period, feature)
            elif feature.startswith('MACD'):
                fast = indicator_config['MACD']['fast_period']
                slow = indicator_config['MACD']['slow_period']
                signal = indicator_config['MACD']['signal_period']
                macd, macd_signal, hist = calculate_macd(data, fast, slow, signal)
                if feature.endswith('_signal'):
                    data[feature] = macd_signal
                elif feature.endswith('_histogram'):
                    data[feature] = hist
                else:
                    data[feature] = macd
            elif feature.startswith('Bollinger'):
                period = indicator_config['Bollinger']['period']
                num_std_dev = indicator_config['Bollinger']['num_std_dev']
                upper, lower = calculate_bollinger_bands(data, period, num_std_dev)
                if feature.endswith('_upper'):
                    data[feature] = upper
                else:
                    data[feature] = lower
        except Exception as e:
            logger.exception(f"Failed to calculate {feature}: {e}")
            continue

    logger.debug(f"Indicators calculated for {timeframe}. Updated data snapshot: {data.head()}")
    return data

def process_data(data, timeframes, feature_config, indicator_config):
    """Process data for multiple timeframes."""
    if not isinstance(timeframes, list):
        raise ValueError("Timeframes should be provided as a list.")
    
    missing_timeframes = [tf for tf in timeframes if tf not in feature_config]
    if missing_timeframes:
        raise ValueError(f"Unsupported timeframes: {missing_timeframes}")
    
    log_step(logger, f"Processing data for timeframes: {timeframes}.")
    processed_data = data.copy()

    for timeframe in timeframes:
        expected_features = feature_config[timeframe]
        log_step(logger, f"Processing indicators for {timeframe}: {expected_features}")
        processed_data = calculate_indicators(processed_data, timeframe, expected_features, indicator_config)
        validate_features(processed_data, expected_features, timeframe)

    logger.info(f"Completed processing for timeframes: {timeframes}.")
    return processed_data
