# app.py
# This script runs a Flask web server to provide model predictions.

from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Load Trained Model and Features ---
try:
    model = joblib.load('maintenance_model.pkl')
    model_features = joblib.load('model_features.pkl')
    print("Model and features loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run train.py first.")
    model = None
    model_features = None

def get_maintenance_message(probability):
    """Generates a user-friendly message based on failure probability."""
    if probability > 0.8:
        return "Critical Alert: Immediate maintenance required."
    elif probability > 0.6:
        return "Warning: Maintenance recommended within the next 500 miles."
    elif probability > 0.4:
        return "Advisory: Plan for service in the next 1000-2000 miles."
    else:
        return "OK: Component is operating within normal parameters."

@app.route('/')
def home():
    """A simple home route to confirm the API is running."""
    return "<h1>AI Maintenance Prediction API</h1><p>Use the /predict endpoint to get predictions.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Takes a JSON payload with sensor data and returns a prediction.
    """
    if not model:
        return jsonify({'error': 'Model is not loaded. Cannot make predictions.'}), 500

    try:
        # Get data from the POST request
        json_data = request.get_json(force=True)
        
        # Convert JSON to pandas DataFrame
        # The input data should be a dictionary like {'Mileage': 85000, ...}
        input_df = pd.DataFrame(json_data, index=[0])
        
        # Ensure the order of columns matches the model's training features
        input_df = input_df[model_features]

        # --- Make Prediction ---
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1] # Probability of class 1 (Failure)

        # --- Prepare Response ---
        output = {
            'prediction': int(prediction[0]),
            'probability': round(float(probability[0]), 4),
            'message': get_maintenance_message(probability[0])
        }

        return jsonify(output)

    except Exception as e:
        # Handle potential errors, like missing keys or invalid data
        return jsonify({
            'error': 'An error occurred during prediction.',
            'details': str(e)
        }), 400

if __name__ == '__main__':
    # Run the app. 
    # Use host='0.0.0.0' to make it accessible from the network.
    # debug=True will auto-reload the server on code changes.
    app.run(host='0.0.0.0', port=5000, debug=True)
