import pandas as pd
import MetaTrader5 as mt5
import logging
import datetime
from flask import Flask, request, jsonify
import json

# Flask app initialization
app = Flask(__name__)

# Logger setup
logger = logging.getLogger()
handler = logging.FileHandler('server.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Timeframe mapping for MetaTrader 5
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
    rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_datetime, end_datetime)
    
    if len(rates) == 0:
        raise ValueError(f"No data fetched for {symbol} on {timeframe} timeframe.")
    
    data = pd.DataFrame(rates)
    data['Time'] = pd.to_datetime(data['time'], unit='s')  # Convert timestamp to datetime
    data.drop(columns=['time'], inplace=True)
    return data

# Function to calculate all technical indicators
def calculate_indicators(symbol, timeframe, start_datetime, end_datetime):
    # Fetch data from MT5
    data = fetch_data(symbol, timeframe, start_datetime, end_datetime)
    
    # Calculate indicators
    data['EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_signal'], data['MACD_histogram'] = calculate_macd(data)
    data['Bollinger_upper'], data['Bollinger_lower'] = calculate_bollinger_bands(data)
    
    return data

# Flask endpoint for real-time indicator calculations
@app.route('/calculate_indicators', methods=['POST'])
def calculate_indicators_api():
    try:
        # Get the request data
        data = request.json
        
        symbol = data['symbol']
        timeframe = data['timeframe']
        start_datetime = datetime.datetime.strptime(data['start_time'], '%Y-%m-%d %H:%M:%S')
        end_datetime = datetime.datetime.strptime(data['end_time'], '%Y-%m-%d %H:%M:%S')
        
        # Calculate indicators
        calculated_data = calculate_indicators(symbol, timeframe, start_datetime, end_datetime)
        
        # Convert the DataFrame to a dictionary and send back as JSON
        result = calculated_data.to_dict(orient='records')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Initialize MetaTrader 5 connection
if not mt5.initialize():
    logger.error("MetaTrader5 initialization failed")
    quit()

# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
