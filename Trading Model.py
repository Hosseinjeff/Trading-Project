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

# Fetch data from MetaTrader 5
def fetch_data(symbol, timeframe, start_datetime, end_datetime):
    mt5_timeframe = timeframe_map.get(timeframe)
    if mt5_timeframe is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    logging.info(f"Fetching data for {symbol} from {start_datetime} to {end_datetime} on timeframe {timeframe}...")
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_datetime, end_datetime)
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

def validate_data(data, required_columns):
    """
    Validates the dataset for missing columns and handles missing values.
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.warning(f"Missing columns in the dataset: {missing_columns}")
        return False
    
    # Check for missing values in required columns
    if data[required_columns].isnull().any().any():
        logging.warning("Missing values detected in the required columns.")
        data.fillna(method='ffill', inplace=True)  # Forward-fill missing values
        data.fillna(method='bfill', inplace=True)  # Backward-fill if needed
        logging.info("Filled missing values using forward and backward fill.")
    
    return True

# Validate current data
required_columns = ['Time', 'open', 'high', 'low', 'close']
logging.info("Validating current timeframe data...")
if not validate_data(current_data, required_columns):
    logging.error("Validation failed for current timeframe data. Exiting program.")
    quit()

higher_data = {}  # Initialize higher_data as a dictionary

for tf in higher_timeframes:
    logging.info(f"Fetching higher timeframe data for {symbol} on {tf}...")
    
    # Fetch and store the higher timeframe data
    higher_data[tf] = fetch_data(symbol, tf, start_time, end_time)
    
    # Validate the data
    logging.info(f"Validating higher timeframe data for {tf}...")
    if not validate_data(higher_data[tf], required_columns):
        logging.error(f"Validation failed for higher timeframe {tf}. Exiting program.")
        quit()

    logging.info(f"Timeframe: {tf}, DataFrame columns: {list(higher_data[tf].columns)}")

# Ensure 'higher_data' contains DataFrames before trying to create one
if isinstance(higher_data, dict):
    higher_data = {key: value for key, value in higher_data.items() if isinstance(value, pd.DataFrame)}

# Convert to DataFrame if needed (this line is optional, depending on your goal)
higher_data_df = pd.concat(higher_data.values(), axis=0, ignore_index=True)

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
import pandas as pd

# Sample loop assuming 'current_data' and 'higher_data' are loaded DataFrames

for timeframe in ['H1']:  # Example timeframe loop
    print(f"Calculating indicators for higher timeframe {timeframe}...")
    
    # Example: Calculate and rename indicators for higher timeframe (dummy calculation)
    # For the actual code, replace with your indicator calculation logic.
    
    print("Indicators calculated.")
    # Ensure higher_data is a DataFrame
    if isinstance(higher_data, dict):
        higher_data = pd.DataFrame(higher_data)
            
    print(type(higher_data))  # Should print <class 'pandas.core.frame.DataFrame'>

    # Rename columns (as seen in logs)
    higher_data.rename(columns={'Time': f'Time_{timeframe}'}, inplace=True)

    # Strip any extra spaces from column names
    current_data.columns = current_data.columns.str.strip()
    higher_data.columns = higher_data.columns.str.strip()

    # Ensure both 'Time' columns are datetime types
    current_data['Time'] = pd.to_datetime(current_data['Time'])
    higher_data['Time_H1'] = pd.to_datetime(higher_data['Time_H1'])

    print(f"Columns in data after renaming: {current_data.columns.tolist()}")

    # Check if 'Time_H1' exists
    if 'Time_H1' in higher_data.columns:
        print("Column 'Time_H1' found. Proceeding to merge...")
        
        try:
            current_data = pd.merge(current_data, higher_data, left_on='Time', right_on='Time_H1', how='left')
            print("Merge successful.")
        except KeyError as e:
            print(f"Merge failed. Error: {e}")
    else:
        print("Column 'Time_H1' not found in higher data.")


# Updated Backtesting Function with Higher Timeframes
def backtest_with_higher_timeframes(data, initial_balance=1000):
    balance = initial_balance
    positions = []
    logging.info("Starting backtest with higher timeframe confirmation...")
    
    for i in range(1, len(data)):
        row = data.iloc[i]
        prev_row = data.iloc[i - 1]

        # Example strategy using higher timeframes
        if not positions:  # Open position logic
            if row['RSI'] < 30 and row['RSI_H1'] > 40 and row['MACD_histogram'] > 0:
                positions.append({'type': 'long', 'entry_price': row['close'], 'entry_time': row['Time']})
                logging.info(f"Opening long position at {row['close']} on {row['Time']}.")
            elif row['RSI'] > 70 and row['RSI_H1'] < 60 and row['MACD_histogram'] < 0:
                positions.append({'type': 'short', 'entry_price': row['close'], 'entry_time': row['Time']})
                logging.info(f"Opening short position at {row['close']} on {row['Time']}.")

        # Close position logic
        for position in positions[:]:
            if position['type'] == 'long' and (row['RSI'] > 70 or row['MACD_histogram_H1'] < 0):
                profit = row['close'] - position['entry_price']
                balance += profit
                positions.remove(position)
                logging.info(f"Closing long position at {row['close']} on {row['Time']}. Profit: {profit}")
            elif position['type'] == 'short' and (row['RSI'] < 30 or row['MACD_histogram_H1'] > 0):
                profit = position['entry_price'] - row['close']
                balance += profit
                positions.remove(position)
                logging.info(f"Closing short position at {row['close']} on {row['Time']}. Profit: {profit}")

    logging.info(f"Backtest completed. Final balance: {balance}")
    return balance

# Run the backtest with the updated strategy
final_balance = backtest_with_higher_timeframes(current_data)
logging.info(f"Final account balance after backtesting: {final_balance}")

# Save processed data to CSV
logging.info("Saving processed data to 'processed_data.csv'...")
current_data.to_csv('processed_data.csv', index=False)
logging.info(f"Processed data saved to 'processed_data.csv'.")

# Further analysis or machine learning model integration can go here
