import pandas as pd
import pickle
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load and preprocess new data
def preprocess_new_data(file_path):
    # Load new data (can be a new CSV file with similar structure to post_processed_data.csv)
    logging.info(f"Loading new data from {file_path}...")
    data = pd.read_csv(file_path)

    # Check for missing values and handle them (if any)
    logging.info("Checking for missing values...")
    if data.isnull().sum().any():
        logging.info("Missing values found, filling missing values...")
        data.fillna(method='ffill', inplace=True)  # Forward fill missing values
    else:
        logging.info("No missing values found.")
    
    # Drop any unnecessary columns (if applicable)
    logging.info("Dropping irrelevant columns...")
    columns_to_drop = ['Time', 'Data_Split']  # Drop time and data_split or any irrelevant columns
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Feature Scaling (Standardization)
    logging.info("Scaling the features...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Convert the scaled data back to a DataFrame
    scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns)
    
    return scaled_data_df

# Function to make predictions
def make_predictions(model_path, new_data_path):
    # Load the trained model
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess the new data
    new_data = preprocess_new_data(new_data_path)
    
    # Extract features (X) and make predictions
    logging.info("Making predictions...")
    X_new = new_data.drop(columns=['close'], errors='ignore')  # Drop target column if exists
    predictions = model.predict(X_new)
    
    # Add predictions to the new data (for comparison)
    new_data['predicted_close'] = predictions
    
    # Save the predictions to a new CSV
    output_file = 'predictions.csv'
    logging.info(f"Saving predictions to {output_file}...")
    new_data.to_csv(output_file, index=False)
    logging.info(f"Predictions saved successfully to {output_file}.")

# Call the function to make predictions
make_predictions('trading_model.pkl', 'new_data.csv')  # Provide path to your new data CSV
