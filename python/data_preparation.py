import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import json
from utils import setup_logger, log_step, features_metadata, indicator_config, processed_data_path, post_processed_data_path
from data_calculation import calculate_indicators  # For advanced indicator calculations

# Logger setup
logger = setup_logger('data_preparation')

# Load configuration files
try:
    with open(features_metadata, 'r') as f:
        FEATURES_METADATA = json.load(f)
    log_step(logger, f"Loaded features metadata: {FEATURES_METADATA}")

    with open(indicator_config, 'r') as f:
        INDICATOR_CONFIG = json.load(f)
    log_step(logger, f"Loaded indicator configuration: {INDICATOR_CONFIG}")
except Exception as e:
    logger.error(f"Error loading configuration files: {e}")
    raise

# Parameters
TARGET_TYPE = "trend"  # Options: "price_change", "trend", "classification_label"
TARGET_HORIZON = 5  # Predict `n` periods ahead


def load_processed_data(file_path):
    """Load processed data for validation and preparation."""
    try:
        data = pd.read_csv(file_path, parse_dates=['Time'], index_col='Time')
        log_step(logger, f"Data loaded successfully from {file_path}. Shape: {data.shape}")
        return data.dropna()  # Drop missing values for clean processing
    except FileNotFoundError:
        log_step(logger, f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        log_step(logger, f"Error loading data: {e}")
        raise

def validate_features(data, required_features):
    """Validate and filter features from the dataset."""
    available_features = data.columns.tolist()
    missing_features = [f for f in required_features if f not in available_features]

    if missing_features:
        log_step(logger, f"Missing features: {missing_features}. They will be skipped.")

    valid_features = [f for f in required_features if f in available_features]
    log_step(logger, f"Valid features identified: {valid_features}")
    return valid_features

def calculate_target(data, target_type, target_horizon):
    """Calculate the target column dynamically."""
    try:
        if target_type == "price_change":
            data["target"] = (data["close"].shift(-target_horizon) - data["close"]) / data["close"] * 100
        elif target_type == "trend":
            data["target"] = (data["close"].shift(-target_horizon) > data["close"]).astype(int)
        elif target_type == "classification_label":
            change = (data["close"].shift(-target_horizon) - data["close"]) / data["close"] * 100
            bins = [-float("inf"), -2, -0.5, 0.5, 2, float("inf")]
            labels = [0, 1, 2, 3, 4]
            data["target"] = pd.cut(change, bins=bins, labels=labels)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

        data.dropna(inplace=True)
        log_step(logger, f"Target '{target_type}' calculated with horizon {target_horizon}.")
        return data
    except Exception as e:
        log_step(logger, f"Error calculating target: {e}")
        raise


def scale_features(data, features):
    """Scale specified features using MinMaxScaler."""
    try:
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])
        log_step(logger, f"Features scaled: {features}")
        return data
    except Exception as e:
        log_step(logger, f"Error scaling features: {e}")
        raise


def prepare_training_data(data, target_column):
    """Split data into training and testing sets."""
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log_step(logger, f"Data split: Training size {len(X_train)}, Testing size {len(X_test)}.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_step(logger, f"Error preparing training data: {e}")
        raise


def main():
    try:
        # Load processed data
        processed_data = load_processed_data(processed_data_path)

        # Get required features from FEATURES_METADATA
        required_features = FEATURES_METADATA.get("features", {}).get("M5", [])  # Example for M5 timeframe
        validated_features = validate_features(processed_data, required_features)

        # Ensure features are present
        if not validated_features:
            raise ValueError("No valid features available for processing.")

        # Calculate the target
        processed_data = calculate_target(processed_data, TARGET_TYPE, TARGET_HORIZON)

        # Scale the features
        scaled_data = scale_features(processed_data, validated_features)

        # Prepare the training data
        X_train, X_test, y_train, y_test = prepare_training_data(scaled_data, "target")

        # Save the prepared training data
        training_data = pd.concat([X_train, y_train], axis=1)
        training_data.to_csv(post_processed_data_path, index=True)
        log_step(logger, f"Post-processed training data saved to {post_processed_data_path}.")
    except Exception as e:
        logger.error(f"Error in data preparation: {e}")
        raise


if __name__ == "__main__":
    main()
