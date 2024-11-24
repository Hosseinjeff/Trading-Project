import pandas as pd
import MetaTrader5 as mt5
import logging
import datetime
import numpy as np

timeframe_map = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
    'MN1': mt5.TIMEFRAME_MN1
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band



# Updated fetch_data function to convert string timeframe to correct integer
def fetch_data(symbol, timeframe, start_datetime, end_datetime):
    mt5_timeframe = timeframe_map.get(timeframe)
    if mt5_timeframe is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_datetime, end_datetime)
    data = pd.DataFrame(rates)
    data['Time'] = pd.to_datetime(data['time'], unit='s')  # Convert timestamp to datetime
    data.drop(columns=['time'], inplace=True)  # Drop the original 'time' column
    return data



# Function to get dynamic higher timeframes
def get_higher_timeframes(current_timeframe):
    timeframe_minutes = {
        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
    }
    
    current_minutes = timeframe_minutes.get(current_timeframe, 1)
    
    higher_timeframes = []
    
    if current_minutes == 1:
        higher_timeframes = ['M15', 'H4']
    elif current_minutes == 5:
        higher_timeframes = ['H1', 'H4']
    elif current_minutes == 15:
        higher_timeframes = ['H4', 'D1']
    elif current_minutes == 30:
        higher_timeframes = ['H1', 'D1']
    elif current_minutes == 60:
        higher_timeframes = ['H4', 'D1']
    elif current_minutes == 240:
        higher_timeframes = ['D1', 'W1']
    
    logging.info(f"Based on {current_timeframe} timeframe, using higher timeframes: {higher_timeframes}")
    
    return higher_timeframes


# Connect to MetaTrader 5
if not mt5.initialize():
    logging.error("MetaTrader5 initialization failed")
    quit()

symbol = 'EURUSD'
current_timeframe = 'M5'

# Get dynamic higher timeframes
higher_timeframes = get_higher_timeframes(current_timeframe)

# Set up start and end times for data retrieval
end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(days=30)

# Fetch current data
current_data = fetch_data(symbol, current_timeframe, start_time, end_time)

# Check the columns of current_data
logging.info(f"Fetched data columns: {current_data.columns.tolist()}")

# Ensure 'Time' is in the data and convert it
if 'Time' in current_data.columns:
    current_data['Time'] = pd.to_datetime(current_data['Time'], unit='s')
else:
    logging.error("'Time' column is missing from the data.")

higher_data = {}
for tf in higher_timeframes:
    higher_data[tf] = fetch_data(symbol, tf, start_time, end_time)

# Calculate technical indicators
current_data['EMA_50'] = current_data['close'].ewm(span=50, adjust=False).mean()
current_data['RSI'] = calculate_rsi(current_data)
current_data['MACD'], current_data['MACD_signal'], current_data['MACD_histogram'] = calculate_macd(current_data)
current_data['Bollinger_upper'], current_data['Bollinger_lower'] = calculate_bollinger_bands(current_data)

logging.info(f"Calculated indicators for {current_timeframe}:")
logging.info(current_data[['Time', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'Bollinger_upper', 'Bollinger_lower']].head())

for tf, data in higher_data.items():
    logging.info(f"Fetched data for higher timeframe ({tf}):")
    logging.info(data.head())

# Further analysis or machine learning model integration can go here
