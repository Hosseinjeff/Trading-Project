# data_preparation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
from utils import setup_logger, log_step

# Logger setup
logger = setup_logger('data_preparation')

# Paths and configurations
base_path = Path(__file__).resolve().parent.parent
data_folder = base_path / 'data'
processed_data_path = data_folder / 'processed_data.csv'
post_processed_data_path = data_folder / 'post_processed_data.csv'
config_folder = Path(__file__).resolve().parent / 'configs'

# Load feature configurations
with open(config_folder / 'feature_config.json', 'r') as f:
    FEATURE_CONFIG = json.load(f)
log_step(logger, f"Loaded feature configuration: {FEATURE_CONFIG}")

# Parameters
TARGET_TYPE = "trend"  # Options: "price_change", "trend", "classification_label"
TARGET_HORIZON = 5  # Predict `n` periods ahead

def load_processed_data(file_path):
    """Load processed data from the file."""
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        log_step(logger, f"Data loaded successfully from {file_path}.")
        
        # Log the shape before dropping invalid rows
        log_step(logger, f"Initial data shape: {data.shape}")

        # Optionally: Fill missing values before dropping rows (if you prefer that over dropping)
        data.fillna(0, inplace=True)  # Or you could fill with the column mean/median

        # Drop rows with missing or invalid values
        data = data.dropna()  
        log_step(logger, f"Shape after dropping rows with missing values: {data.shape}")

        return data
    except FileNotFoundError:
        log_step(logger, f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        log_step(logger, f"Error loading data from {file_path}: {e}")
        raise

def prepare_data_for_training(data, target_column):
    """Prepare data for training by splitting and scaling."""
    try:
        log_step(logger, "Preparing data for training.")
        
        # Ensure target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Split data into features (X) and target (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        log_step(logger, f"Data split into training and testing sets. Training size: {len(X_train)}, Testing size: {len(X_test)}.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_step(logger, f"Error during data preparation: {e}")
        raise

def save_post_processed_data(data, file_path):
    """Save the prepared data to a CSV file."""
    try:
        data.to_csv(file_path)
        log_step(logger, f"Post-processed data saved to {file_path}.")
    except Exception as e:
        log_step(logger, f"Error saving post-processed data: {e}")
        raise

def validate_features(data, required_features):
    """Validate that required features exist in the data."""
    available_features = data.columns.tolist()
    missing_features = [feature for feature in required_features if feature not in available_features]

    if missing_features:
        log_step(logger, f"Missing features: {', '.join(missing_features)}. They will be skipped.")
        
    # Ensure that the list is filtered to only include valid features
    valid_features = [feature for feature in required_features if feature in available_features]
    
    # Debugging: Log available and missing features
    log_step(logger, f"Available features in data: {', '.join(available_features)}")
    log_step(logger, f"Valid features to be used for scaling: {', '.join(valid_features)}")
    
    return valid_features

def drop_improper_rows(data):
    """
    Drop rows with missing or invalid data.
    """
    try:
        # Count missing values per row
        missing_row_count = data.isna().sum(axis=1).sum()
        if missing_row_count > 0:
            log_step(logger, f"Dropping {missing_row_count} rows with missing or invalid values.")
            data.dropna(inplace=True)
            
            # Log rows that were dropped and the overall data description
            invalid_rows = data[data.isnull().any(axis=1)]
            logger.debug(f"Dropping rows due to NaNs: {invalid_rows}")
            logger.debug(f"Invalid values in columns: {data.describe(include='all')}")
        else:
            log_step(logger, "No missing or invalid rows detected.")
        return data
    except Exception as e:
        log_step(logger, f"Error dropping improper rows: {e}")
        raise

def calculate_target(data, target_type, target_horizon):
    """Calculate the target column based on type and horizon."""
    try:
        if target_type == "price_change":
            data["target"] = (data["close"].shift(-target_horizon) - data["close"]) / data["close"] * 100
        elif target_type == "trend":
            data["target"] = (data["close"].shift(-target_horizon) > data["close"]).astype(int)
        elif target_type == "classification_label":
            change = (data["close"].shift(-target_horizon) - data["close"]) / data["close"] * 100
            bins = [-float("inf"), -2, -0.5, 0.5, 2, float("inf")]
            labels = [0, 1, 2, 3, 4]  # Example: 0=large drop, 4=large rise
            data["target"] = pd.cut(change, bins=bins, labels=labels)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        data.dropna(inplace=True)  # Remove rows with NaN target values
        log_step(logger, f"Target '{target_type}' calculated with a horizon of {target_horizon}.")
        return data
    except Exception as e:
        log_step(logger, f"Error calculating target: {e}")
        raise

def scale_features(data, features):
    """Scale features using MinMaxScaler."""
    try:
        if data[features].empty:
            log_step(logger, "No data available for scaling after filtering. Skipping scaling step.")
            return data  # Return the original data
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])
        log_step(logger, f"Features scaled: {', '.join(features)}")
        return data
    except Exception as e:
        log_step(logger, f"Error during feature scaling: {e}")
        raise

# Inside the main script logic
if __name__ == "__main__":
    try:
        # Load processed data
        log_step(logger, f"Loading processed data from {processed_data_path}.")
        processed_data = load_processed_data(processed_data_path)

        # Drop rows with missing or improper data
        log_step(logger, f"Shape after dropping improper rows: {processed_data.shape}")
        processed_data = drop_improper_rows(processed_data)

        # Check if data is empty after dropping rows
        if processed_data.empty:
            raise ValueError("No valid data available after dropping rows. Please check data integrity.")

        # Calculate the target column dynamically
        log_step(logger, f"Shape before calculating target: {processed_data.shape}")
        processed_data = calculate_target(processed_data, TARGET_TYPE, TARGET_HORIZON)

        # Detect features dynamically based on configuration
        timeframes = list(FEATURE_CONFIG.keys())  # Example: ['M5', 'H1']
        dynamic_features = []
        for timeframe in timeframes:
            dynamic_features += FEATURE_CONFIG[timeframe]

        # Validate and filter features
        validated_features = validate_features(processed_data, dynamic_features)

        # Check if any features are left after validation
        if not validated_features:
            raise ValueError("No valid features available for scaling. Please check the data or feature configuration.")

        # Scale the features
        if processed_data.empty:
            raise ValueError("No valid data available after preprocessing. Check earlier steps for issues.")
        scaled_data = scale_features(processed_data, validated_features)

        # Prepare the data for training
        X_train, X_test, y_train, y_test = prepare_data_for_training(scaled_data, "target")

        # Save the prepared data (optional: save X_train, X_test, y_train, y_test separately if needed)
        combined_data = pd.concat([X_train, y_train], axis=1)
        save_post_processed_data(combined_data, post_processed_data_path)

    except Exception as e:
        logger.error(f"Unexpected error in data preparation: {e}")
        raise
