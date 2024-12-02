import pandas as pd
import MetaTrader5 as mt5
import logging
import datetime
import numpy as np
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
from pathlib import Path

# Create or get a logger
logger = logging.getLogger()

# Set up logging to a file if needed
handler = logging.FileHandler('data.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.setLevel(logging.INFO)  # You can change the logging level here

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# Fetch data from MetaTrader 5
def fetch_data(symbol, timeframe, start_datetime, end_datetime):
    mt5_timeframe = timeframe_map.get(timeframe)
    if mt5_timeframe is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    logging.info(f"Fetching data for {symbol} from {start_datetime} to {end_datetime} on timeframe {timeframe}...")
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_datetime, end_datetime)
    
    if len(rates) == 0:
        logging.error(f"No data fetched for {symbol} from {start_datetime} to {end_datetime} on timeframe {timeframe}")
        return pd.DataFrame()  # Return an empty DataFrame if no data is fetched
    
    data = pd.DataFrame(rates)
    data['Time'] = pd.to_datetime(data['time'], unit='s')  # Convert timestamp to datetime
    data.drop(columns=['time'], inplace=True)  # Drop the original 'time' column
    logging.info(f"Fetched data columns: {data.columns.tolist()}")
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

# Fetch data for current and higher timeframes
logging.info(f"Fetching current data for {symbol} on timeframe {current_timeframe}...")
current_data = fetch_data(symbol, current_timeframe, start_time, end_time)

higher_data = {}
for tf in tqdm(higher_timeframes, desc="Fetching higher timeframes", unit="timeframe"):
    logging.info(f"Fetching higher timeframe data for {symbol} on {tf}...")
    higher_data[tf] = fetch_data(symbol, tf, start_time, end_time)

# Calculate technical indicators for each timeframe
def calculate_indicators(data):
    logging.info("Calculating indicators...")
    data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_signal'], data['MACD_histogram'] = calculate_macd(data)
    data['Bollinger_upper'], data['Bollinger_lower'] = calculate_bollinger_bands(data)
    logging.info("Indicators calculated.")
    return data

# Calculate indicators for current timeframe
logging.info(f"Calculating indicators for current timeframe {current_timeframe}...")
current_data = calculate_indicators(current_data)

# Log the first few rows after calculating indicators for the current timeframe
logging.info(f"Calculated indicators for {current_timeframe}:")
logging.info(current_data[['Time', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'Bollinger_upper', 'Bollinger_lower']].head())

# Calculate indicators for higher timeframes and merge with current data
for tf, data in tqdm(higher_data.items(), desc="Calculating and merging higher timeframes", unit="timeframe"):
    logging.info(f"Calculating indicators for higher timeframe {tf}...")
    data = calculate_indicators(data)
    logging.info(f"Merging higher timeframe {tf} data with current data...")
    # Merge the higher timeframe indicators with the current data (align on 'Time' column)
    current_data = pd.merge(current_data, data[['Time', f'EMA_50', f'RSI', f'MACD', f'MACD_signal', f'MACD_histogram', f'Bollinger_upper', f'Bollinger_lower']],
                            on='Time', suffixes=('', f'_{tf}'))

    # Log the fetched higher timeframe data
    logging.info(f"Fetched data for higher timeframe ({tf}):")
    logging.info(data.head())

# Save processed data to CSV
logging.info("Saving processed data to 'processed_data.csv'...")
# Get the current directory of the Python script
current_directory = Path(__file__).resolve().parent

# Get the path of the sister directory (assuming sibling directory is one level up)
sister_directory = current_directory.parent / "data"

# Ensure the "data" directory exists
if not sister_directory.exists():
    logging.info(f"Creating directory: {sister_directory}")
    sister_directory.mkdir(parents=True, exist_ok=True)

# Save processed data to CSV
logging.info("Saving processed data to 'processed_data.csv'...")
current_data.to_csv(sister_directory / 'processed_data.csv', index=False)
logging.info(f"Processed data saved to '{sister_directory / 'processed_data.csv'}'.")
