import pickle
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Absolute path to the model file
model_path = "C:\\users\\Setin\\Documents\\GitHub\\Trading-Project\\trading_model.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data

    # Convert data into a DataFrame for preprocessing
    data_df = pd.DataFrame([data])

    # Preprocess the data (same as during training)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_df)  # Apply the same scaling technique

    # Make the prediction
    prediction = model.predict(scaled_data)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Expose the server on port 5000
