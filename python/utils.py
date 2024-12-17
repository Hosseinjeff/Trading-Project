# utils.py
import logging
import os
import sys
from pathlib import Path

# Paths
current_folder = Path(__file__).resolve().parent  # 'python' folder
base_path = current_folder.parent

# Folders
EA_folder = Path("C:/Users/Setin/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files") #EA's accessible folder
model_folder = base_path / "models"
data_folder = base_path / "data"
config_folder = current_folder / "configs"

# Files
log_file = base_path / "project_log.log"
features_metadata = config_folder / "features_metadata.json"
indicator_config = config_folder / "indicator_config.json"
processed_data_path = data_folder / 'processed_data.csv'
post_processed_data_path = data_folder / 'post_processed_data.csv'
db_path = EA_folder/ 'prediction_data.db'  # Path to your SQLite database

# Append 'python' folder to system path for imports
sys.path.append(str(current_folder))

# Configure a shared logger
def setup_logger(module_name, log_file='project_log.log'):
    """Set up a logger for a module."""
    logger = logging.getLogger(module_name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

def log_step(logger, message, script_name=None):
    """Log a step with an optional script name."""
    if script_name is None:
        script_name = os.path.basename(__file__)
    logger.info(f"{script_name} - {message}")

def load_model_related_files(model_path):
    """
    Determines the paths of the related scaler and features metadata based on the model file.
    """
    try:
        # Extract model name and timestamp
        model_name = os.path.basename(model_path)
        timestamp = "_".join(model_name.split('_')[-2:]).split('.')[0]  # Extract the timestamp

        # Determine the folder and corresponding file paths
        folder = os.path.dirname(model_path)
        scaler_path = os.path.join(folder, f"scaler_{timestamp}.pkl")
        features_path = os.path.join(folder, f"features_{timestamp}.json")

        # Check if files exist
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features metadata file not found: {features_path}")

        logging.info(f"Loaded model: {model_path}")
        logging.info(f"Loaded scaler: {scaler_path}")
        logging.info(f"Loaded features metadata: {features_path}")

        return scaler_path, features_path

    except Exception as e:
        logging.error(f"Error loading model, scaler, or features metadata: {e}")
        raise
