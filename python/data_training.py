# data_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from utils import setup_logger, log_step, model_folder, post_processed_data_path
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import logging
from datetime import datetime
import json
import time

# Logger setup
logger = setup_logger('data_preparation')

# Function to check for package availability and use the best option
def check_and_use_gpu_package():
    log_step(logger,"Checking for available training packages...")
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            log_step(logger,"Using TensorFlow with GPU.")
            return 'tensorflow'
    except ImportError:
        logging.warning("TensorFlow unavailable.")

    try:
        import torch
        if torch.cuda.is_available():
            log_step(logger,"Using PyTorch with GPU.")
            return 'pytorch'
    except ImportError:
        logging.warning("PyTorch unavailable.")

    log_step(logger,"Defaulting to Scikit-learn (CPU).")
    return 'sklearn'

# Function to load data and train models
def train_model(file_path= post_processed_data_path):
    # Check for the available package (TensorFlow, PyTorch, or scikit-learn)
    package = check_and_use_gpu_package()

    # Load the post-processed data
    log_step(logger,f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)

    # Dynamically prepare the features (X) and target (y)
    log_step(logger,"Preparing features and target variables...")
    feature_columns = [col for col in data.columns if col not in ['close', 'Data_Split']]  # Dynamically exclude target and Data_Split
    X = data[feature_columns]  # Features
    y = data['close']  # Target variable

    # Ensure that we are only using numeric columns for scaling
    X_numeric = X.select_dtypes(include=['number']).copy()
    if X_numeric.empty:
        raise ValueError("No numeric features found for training.")

    # Split the data into training and testing sets
    log_step(logger,"Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

    # Initialize the scaler and scale the features
    log_step(logger,"Scaling the features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
    X_test_scaled = scaler.transform(X_test)        # Only transform the test data (using the trained scaler)

    # Models to evaluate (based on selected package)
    models = {}

    model_info = []  # Store model information for metadata

    if package == 'tensorflow' or package == 'tensorflow_cpu':
        import tensorflow as tf
        log_step(logger,f"TensorFlow version: {tf.__version__}")
        log_step(logger,f"Available devices: {tf.config.list_physical_devices()}")
        models['TensorFlow'] = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        models['TensorFlow'].compile(optimizer='adam', loss='mean_squared_error')
        model_info.append({
            'model': 'TensorFlow',
            'package': package,
            'architecture': 'Dense layers',
            'epochs': 10,
            'batch_size': 32
        })
    elif package == 'pytorch' or package == 'pytorch_cpu':
        import torch
        import torch.nn as nn
        models['PyTorch'] = nn.Sequential(
            nn.Linear(len(feature_columns), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        models['PyTorch'] = models['PyTorch'].to('cuda' if package == 'pytorch' else 'cpu')
        model_info.append({
            'model': 'PyTorch',
            'package': package,
            'architecture': 'Fully connected layers',
            'epochs': 10,
            'batch_size': 32
        })
    else:  # Fallback to scikit-learn if no GPU-enabled library is found
        models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        model_info.append({
            'model': 'Random Forest',
            'package': package,
            'n_estimators': 100,
            'random_state': 42
        })

    # Dictionary to store model performance
    performance_metrics = {}

    # Train, evaluate, and save each model
    best_model = None
    best_rmse = float('inf')

    for model_name, model in models.items():
        log_step(logger,f"Training the {model_name} model...")

        if package == 'tensorflow' or package == 'tensorflow_cpu':
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)  # Adjust epochs and batch size as needed
            y_pred = model.predict(X_test_scaled)
        elif package == 'pytorch' or package == 'pytorch_cpu':
            # PyTorch training loop
            model.train()
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(model.device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(model.device)
            optimizer.zero_grad()
            output = model(X_train_tensor)
            loss = criterion(output, y_train_tensor)
            loss.backward()
            optimizer.step()
            y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)).cpu().detach().numpy()
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        # Calculate performance metrics
        rmse = root_mean_squared_error(y_test, y_pred)  # Directly use root_mean_squared_error from the start
        mae = mean_absolute_error(y_test, y_pred)

        # Log the performance metrics
        log_step(logger,f"RMSE: {rmse}")
        log_step(logger,f"MAE: {mae}")

        # Store performance metrics
        performance_metrics[model_name] = {'RMSE': rmse, 'MAE': mae}

        # Track the best model (lowest RMSE)
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # Log the performance of all models
    log_step(logger,f"Performance metrics for all models: {performance_metrics}")

       
    # Get the current date for versioning
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate model filenames with date-based versioning
    best_model_filename = model_folder / f'best_trained_model_{current_date}.pkl'
    scaler_filename = model_folder / f'scaler_{current_date}.pkl'
    features_filename = model_folder / f'features_{current_date}.json'  # Save features as JSON file
    
    # Prepare metadata
    metadata = {
        "model_name": type(best_model).__name__,
        "model_parameters": best_model.get_params() if hasattr(best_model, 'get_params') else "N/A",
        "features": feature_columns,
        "target": "close",
        "metrics": {
            "rmse": best_rmse,
            "performance_metrics": performance_metrics
        },
        "training_data_info": {
            "train_size": len(X_train),
            "test_size": len(X_test)
        },
        "training_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save the best model, scaler, and features list to disk
    joblib.dump(best_model, best_model_filename)
    joblib.dump(scaler, scaler_filename)
    with open(features_filename, 'w') as f:
        json.dump(metadata, f, indent=4)  # Save the features list as JSON
            
    log_step(logger,f"Best model training completed. RMSE: {best_rmse}")
    log_step(logger,f"Best model, scaler, and feature list saved successfully.")

# Call the train_model function
if __name__ == "__main__":
    train_model()
