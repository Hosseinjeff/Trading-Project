import logging
import json
import datetime
import socket
import pandas as pd
import MetaTrader5 as mt5
import win32pipe, win32file
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PIPE_UNLIMITED_INSTANCES = 255

# Directories and logging configuration
current_directory = Path(__file__).resolve().parent
log_file = current_directory.parent / "logs.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - data_server - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

def serialize_timestamps(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# Fetch historical data from MetaTrader 5
def fetch_data(symbol, timeframe, start_time, end_time):
    """Fetch historical data from MetaTrader 5 with error handling."""
    try:
        logging.info(f"Fetching data for {symbol} from {start_time} to {end_time}...")
        
        # Convert timeframe string to MetaTrader5 constant
        mt5_timeframe = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'D1': mt5.TIMEFRAME_D1
        }.get(timeframe, None)

        if not mt5_timeframe:
            logging.error(f"Invalid timeframe: {timeframe}")
            return pd.DataFrame()

        # Fetch the rates
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_time, end_time)
        if rates is None or len(rates) == 0:
            logging.error(f"No data available for {symbol} in the given range.")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        logging.info(f"Fetched {df.shape[0]} rows of data for {symbol}.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Calculate RSI
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Process client requests
def process_request(pipe):
    try:
        # Read request data
        hr, data = win32file.ReadFile(pipe, 65536)
        request = json.loads(data.decode('utf-8'))
        logging.info(f"Received request: {request}")

        # Process the request
        symbol = request['symbol']
        timeframe = request['timeframe']
        start_time = pd.Timestamp(request['start_time'])
        end_time = pd.Timestamp(request['end_time'])
        indicator = request['indicator']

        # Fetch data and calculate indicator
        data = fetch_data(symbol, timeframe, start_time, end_time)
        if data.empty:
            response_data = {"error": "No data available."}
        elif indicator == "RSI":
            data['RSI'] = calculate_rsi(data)
            response_data = data.to_dict(orient='records')
        else:
            response_data = {"error": f"Unsupported indicator: {indicator}"}

        # Send response
        json_response = json.dumps(response_data, default=serialize_timestamps)
        win32file.WriteFile(pipe, json_response.encode('utf-8'))
        logging.info("Response sent successfully.")
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        error_response = json.dumps({"error": str(e)})
        win32file.WriteFile(pipe, error_response.encode('utf-8'))
    finally:
        if 'pipe' in locals():
            win32file.CloseHandle(pipe)

# Named pipe server logic
def start_named_pipe_server(pipe_name):
    logging.info(f"Starting named pipe server on pipe: {pipe_name}")
    with ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            try:
                pipe = win32pipe.CreateNamedPipe(
                    rf'\\.\pipe\{pipe_name}',
                    win32pipe.PIPE_ACCESS_DUPLEX,
                    win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                    PIPE_UNLIMITED_INSTANCES,
                    65536,  # Output buffer size
                    65536,  # Input buffer size
                    0,  # Timeout in milliseconds
                    None  # Security
                )
                win32pipe.ConnectNamedPipe(pipe, None)
                executor.submit(process_request, pipe)
            except Exception as e:
                logging.error(f"Error during pipe connection: {e}")
                time.sleep(1)  # Optional small delay to prevent CPU overuse

# Threaded function to handle each client
def handle_client(client_pipe):
    try:
        # Simulate data fetching and processing logic
        logging.info("Client connected to the pipe.")
        # Your logic to fetch and process data goes here
        time.sleep(1)  # Simulate processing time
        logging.info("Data sent successfully.")
    except Exception as e:
        logging.error(f"Error processing client: {e}")
    finally:
        # Ensure the pipe is properly closed after processing
        client_pipe.close()

def send_data(client_pipe, data):
    try:
        if not os.path.exists(client_pipe):  # Check if the pipe exists and is writable
            raise Exception(f"Pipe {client_pipe} is not available.")
        # Proceed with sending the data to the client
        send_data(client_pipe, data)
    except Exception as e:
        logging.error(f"Error while sending data: {e}")

# This is the entry point for the server script
if __name__ == "__main__":
    pipe_name = "DataServerPipe"  # Match this with the client
    start_named_pipe_server(pipe_name)
