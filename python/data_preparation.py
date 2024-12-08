# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm  # Import tqdm for the progress bar
from pathlib import Path

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

# Get the current directory of the Python script
current_directory = Path(__file__).resolve().parent

# Get the path of the sister directory (assuming sibling directory is one level up)
sister_directory = current_directory.parent / "data"

# Function to load and prepare data
def prepare_data(file_path=sister_directory / 'processed_data.csv', save_path=sister_directory / 'post_processed_data.csv'):
    logging.info(f"Starting data preparation process...")

    # Load the dataset
    logging.info(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape of the data: {data.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return
    
    # Check for missing values and handle them (if any)
    logging.info("Checking for missing values...")
    if data.isnull().sum().any():
        logging.info("Missing values found, filling missing values...")
        data.fillna(method='ffill', inplace=True)  # Forward fill missing values
        logging.info("Missing values filled.")
    else:
        logging.info("No missing values found.")
    
    # Drop any unnecessary columns (if applicable)
    logging.info("Dropping irrelevant columns...")
    columns_to_drop = ['Time']  # Drop time or any other irrelevant columns for training
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    logging.info(f"Dropped columns: {columns_to_drop}")

    # Feature Scaling (Standardization)
    logging.info("Scaling the features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    logging.info("Feature scaling complete.")

    # Convert the scaled data back to a DataFrame
    scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
    
    # Split the data into features (X) and target (y)
    logging.info("Splitting data into features and target...")
    X = scaled_data_df.drop(columns=['close'])  # Drop target variable
    y = scaled_data_df['close']  # Target variable
    logging.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    
    # Split the data into training and testing sets (80% train, 20% test)
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    # Add a new column to mark whether the row is for training or testing
    X_train['Data_Split'] = 'train'
    X_test['Data_Split'] = 'test'
    
    # Combine X_train, X_test, y_train, and y_test into one DataFrame
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    logging.info("Training and testing data combined.")
    
    # Combine the train and test datasets into one final dataset
    final_data = pd.concat([train_data, test_data], axis=0)
    logging.info(f"Final data shape after combining: {final_data.shape}")

    # Save the processed data to CSV with a progress bar
    logging.info(f"Saving processed data to {save_path}...")
    try:
        with tqdm(total=len(final_data), desc="Saving Data", unit="row") as pbar:
            final_data.to_csv(save_path, index=False)
            pbar.update(len(final_data))  # Update progress bar after saving
        logging.info(f"Data saved successfully to {save_path}.")
    except Exception as e:
        logging.error(f"Error saving data to {save_path}: {e}")

# Call the function to prepare the data
prepare_data()
