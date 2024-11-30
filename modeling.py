import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data and train model
def train_model(file_path='post_processed_data.csv'):
    # Load the post-processed data
    logging.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Split the data into features (X) and target (y)
    logging.info("Preparing features and target variables...")
    X = data.drop(columns=['close', 'Data_Split'])  # Dropping 'close' and 'Data_Split' as they are not features
    y = data['close']  # Target variable
    
    # Split the data into training and testing sets
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model (Random Forest Regressor)
    logging.info("Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    logging.info("Evaluating the model...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Log the performance metrics
    logging.info(f"RMSE: {rmse}")
    logging.info(f"MAE: {mae}")
    
    # Save the trained model
    model_filename = 'C:\\Users\\Setin\\Documents\\GitHub\\Trading-Project\\trading_model.pkl'
    joblib.dump(model, model_filename)
    
    logging.info(f"Model training completed. RMSE: {rmse}, MAE: {mae}")
    logging.info(f"Model saved successfully as {model_filename}.")

# Call the train_model function
if __name__ == "__main__":
    train_model()
