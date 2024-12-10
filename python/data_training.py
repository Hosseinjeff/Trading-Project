# data_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
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

# Get the current directory of the Python script
current_directory = Path(__file__).resolve().parent

# Get the path of the sister directory (assuming sibling directory is one level up)
sister_directory = current_directory.parent / "data"

# Function to load data and train model
def train_model(file_path= sister_directory / 'post_processed_data.csv'):
    # Load the post-processed data
    logging.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Dynamically prepare the features (X) and target (y)
    logging.info("Preparing features and target variables...")
    feature_columns = [col for col in data.columns if col not in ['close', 'Data_Split']]  # Dynamically exclude target and Data_Split
    X = data[feature_columns]  # Features
    y = data['close']  # Target variable
    
    # Split the data into training and testing sets
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the scaler and scale the features
    logging.info("Scaling the features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
    X_test_scaled = scaler.transform(X_test)        # Only transform the test data (using the trained scaler)
    
    # Initialize and train the model (Random Forest Regressor)
    logging.info("Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Log the performance metrics
    logging.info(f"RMSE: {rmse}")
    logging.info(f"MAE: {mae}")
    
    # Get the current directory of the Python script
    current_directory = Path(__file__).resolve().parent

    # Get the path of the sister directory (assuming sibling directory is one level up)
    sister_directory = current_directory.parent / "models"

    # Save the trained model, scaler, and feature columns
    model_filename = sister_directory / 'trained_model.pkl'
    scaler_filename = sister_directory / 'scaler.pkl'
    features_filename = sister_directory / 'features.pkl'
    
    # Save the model, scaler, and features list to disk
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(feature_columns, features_filename)  # Save the features list
    
    logging.info(f"Model training completed. RMSE: {rmse}, MAE: {mae}")
    logging.info(f"Model, scaler, and feature list saved successfully.")

# Call the train_model function
if __name__ == "__main__":
    train_model()
