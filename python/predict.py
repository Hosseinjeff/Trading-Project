# predict.py
import pandas as pd
import numpy as np
import json
import os
import logging
import sqlite3
from pathlib import Path
from utils import setup_logger, log_step, load_model_related_files, model_folder, db_path
from data_calculation import calculate_indicators, validate_features
from joblib import load as joblib_load

# Use the shared logger from utils
logger = setup_logger('predict')

def create_connection(db_path = db_path):
    """Create a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Error creating database connection: {e}")
        raise

def initialize_database(features_metadata):
    """Initialize the database with required tables dynamically."""
    try:
        conn = create_connection()
        
        # Dynamically build the table schema from features_metadata
        feature_columns = [f"{feature} REAL" for feature in features_metadata['features']]
        schema_columns = ",\n".join(["timestamp DATETIME PRIMARY KEY", "Time TEXT"] + feature_columns + ["target REAL"])

        processed_data_schema = f"""
            CREATE TABLE IF NOT EXISTS processed_data (
                {schema_columns}
            )
        """

        ensure_table_exists(conn, "processed_data", schema=processed_data_schema)
        conn.close()
        logger.info("Database initialized with dynamic table schema based on model features.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def ensure_table_exists(conn, table_name, schema=None):
    """Ensure that a table exists in the database."""
    try:
        query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        result = pd.read_sql(query, conn)
        if result.empty:
            if schema:
                conn.execute(schema)
                logger.info(f"Table '{table_name}' created with specified schema.")
            else:
                logger.warning(f"Table '{table_name}' does not exist and no schema provided.")
    except Exception as e:
        logger.error(f"Error ensuring table {table_name}: {e}")

def load_feature_config():
    """Load the feature configuration from the database."""
    try:
        conn = create_connection()
        query = "SELECT * FROM feature_config"
        feature_config = pd.read_sql(query, conn).to_dict(orient='records')[0]  # Assuming 1 record
        conn.close()
        logger.info("Feature configuration loaded.")
        return feature_config
    except Exception as e:
        logger.error(f"Error loading feature configuration: {e}")
        raise

def load_indicator_config():
    """Load indicator configuration from the database."""
    try:
        conn = create_connection()
        query = "SELECT * FROM indicator_config"
        indicator_config = pd.read_sql(query, conn).to_dict(orient='records')[0]  # Assuming 1 record
        conn.close()
        logger.info("Indicator configuration loaded.")
        return indicator_config
    except Exception as e:
        logger.error(f"Error loading indicator configuration: {e}")
        raise

def load_data():
    """Load the most recent data from the database."""
    try:
        conn = create_connection()
        query = """
            SELECT * FROM processed_data
            ORDER BY timestamp DESC
            LIMIT 100
        """
        data = pd.read_sql(query, conn)
        conn.close()
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def check_inference_readiness(data, features, required_points):
    """Check if the database has enough data points for inference."""
    missing_features = [feat for feat in features if feat not in data.columns]
    if missing_features:
        logger.info(f"Missing features: {missing_features}. Calculating...")
        return False
    
    if len(data) < required_points:
        logger.info(f"Not enough data points. Required: {required_points}, Available: {len(data)}")
        return False
    
    logger.info("Sufficient data points and features available for inference.")
    return True

from data_calculation import process_data  # Ensure process_data is imported

def process_data_for_inference(required_features, required_points=100):
    """
    Process primary data to calculate necessary features dynamically,
    ensuring readiness for inference.
    """
    try:
        conn = create_connection()

        # Load primary data (raw, unprocessed data inserted by EA)
        query = "SELECT * FROM processed_data ORDER BY timestamp ASC"
        primary_data = pd.read_sql(query, conn)

        # Load timeframes and indicator/feature configurations dynamically
        feature_config = {"features": required_features}
        indicator_config = load_indicator_config()
        timeframes = indicator_config.get("timeframes", ["M1"])  # Default to M1 if not specified

        # Process data using process_data (which internally calls calculate_indicators)
        calculated_data = process_data(
            primary_data, 
            timeframes=timeframes, 
            feature_config=feature_config, 
            indicator_config=indicator_config
        )

        # Update database with calculated features
        calculated_data.to_sql('processed_data', conn, if_exists='replace', index=False)

        # Validate readiness for inference
        if not check_inference_readiness(calculated_data, required_features, required_points):
            logger.warning("Data is not yet ready for inference. Waiting for more data.")
            conn.close()
            return None

        conn.close()
        logger.info("Data processed and ready for inference.")
        return calculated_data
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

def load_latest_model():
    """Automatically load the most recent model, scaler, and feature configuration."""
    try:
        model_files = list(model_folder.glob("*.pkl")) + list(model_folder.glob("*.json"))
        model_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)

        model_file = next((f for f in model_files if 'best_trained_model' in f.stem), None)
        scaler_file = next((f for f in model_files if 'scaler' in f.stem), None)
        features_file = next((f for f in model_files if 'features' in f.stem), None)

        if not model_file or not scaler_file or not features_file:
            raise FileNotFoundError("Missing required model files.")

        logger.info(f"Loaded latest model: {model_file.stem}")
        return model_file, scaler_file, features_file
    except Exception as e:
        logger.error(f"Error loading the latest model files: {e}")
        raise

def make_prediction(model, scaler, features, data):
    """Make predictions using the model."""
    try:
        scaled_data = scaler.transform(data[features])
        predictions = model.predict(scaled_data)
        logger.info("Predictions made successfully.")
        return predictions
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise

def main():
    """Main function to manage data flow, feature calculation, and inference."""
    try:
        # Load model and associated metadata
        model_file, scaler_file, features_file = load_latest_model()
        model = joblib_load(model_file)
        scaler = joblib_load(scaler_file)
        with open(features_file, 'r') as f:
            features_metadata = json.load(f)
        required_features = features_metadata['features']
        required_points = features_metadata.get('required_points', 100)  # Default to 100 if unspecified

        initialize_database(features_metadata)
        
        # Process data for inference
        processed_data = None
        while not processed_data:
            processed_data = process_data_for_inference(required_features, required_points)

        # Perform inference
        predictions = make_prediction(model, scaler, required_features, processed_data)
        logger.info(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
