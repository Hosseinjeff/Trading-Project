# data_calculation.py
import pandas as pd
import numpy as np
import logging
# Create or get a logger
logger = logging.getLogger()

# Set up logging to a file in the parent directory
log_file = Path(__file__).resolve().parent.parent / 'log.txt'
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter(f'%(asctime)s - {Path(__file__).name} - %(message)s'))
logger.addHandler(handler)

logger.setLevel(logging.INFO)  # You can change the logging level here

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_step(message):
    script_name = os.path.basename(__file__)
    logging.info(f'{script_name} - {message}')

def calculate_ema(data, period):
    """Calculate Exponential Moving Average (EMA)."""
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(data, period=20, num_std_dev=2):
    """Calculate Bollinger Bands."""
    sma = data['close'].rolling(window=period).mean()
    std_dev = data['close'].rolling(window=period).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

def calculate_indicators(data, timeframe):
    """Calculate all indicators for a given timeframe."""
    log_step(f"Calculating indicators for {timeframe}.")
    data[f'EMA_50_{timeframe}'] = calculate_ema(data, 50)
    data[f'RSI_{timeframe}'] = calculate_rsi(data)
    macd, signal, hist = calculate_macd(data)
    data[f'MACD_{timeframe}'] = macd
    data[f'MACD_signal_{timeframe}'] = signal
    data[f'MACD_histogram_{timeframe}'] = hist
    upper, lower = calculate_bollinger_bands(data)
    data[f'Bollinger_upper_{timeframe}'] = upper
    data[f'Bollinger_lower_{timeframe}'] = lower
    log_step(f"Indicators calculated for {timeframe}.")
    return data

def process_data(data, timeframes):
    """Calculate indicators for multiple timeframes."""
    processed_data = data.copy()
    for tf in timeframes:
        processed_data = calculate_indicators(processed_data, tf)
    return processed_data
