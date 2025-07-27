# train.py
# This script handles data simulation and model training.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def simulate_vehicle_data(num_records=5000):
    """
    Generates a synthetic dataset for predictive maintenance.

    Args:
        num_records (int): The number of data records to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the synthetic vehicle data.
    """
    print("Generating synthetic vehicle data...")
    
    # Seed for reproducibility
    np.random.seed(42)

    # Feature generation
    data = {
        'VehicleID': range(1, num_records + 1),
        'Mileage': np.random.randint(1000, 150000, size=num_records),
        'Temperature': np.random.normal(90, 15, size=num_records), # Avg temp 90F, std 15
        'Vibration': np.random.normal(5, 2, size=num_records),   # Avg vibration 5, std 2
        'HoursOperated': np.random.randint(50, 5000, size=num_records)
    }
    df = pd.DataFrame(data)

    # --- Create a target variable: 'Failure' ---
    # Failure is more likely with high mileage, high temp, and high vibration
    failure_probability = (df['Mileage'] / 150000) + \
                          (df['Temperature'] / 150) + \
                          (df['Vibration'] / 10)

    # Normalize probability to be between 0 and 1
    failure_probability = (failure_probability - failure_probability.min()) / \
                          (failure_probability.max() - failure_probability.min())

    # Assign failure label based on a threshold
    # We introduce some randomness to make it more realistic
    failure_threshold = 0.75
    df['Failure'] = (failure_probability > failure_threshold) | (np.random.rand(num_records) < 0.05)
    df['Failure'] = df['Failure'].astype(int)
    
    print(f"Data generation complete. Number of failures: {df['Failure'].sum()}")
    print("Sample data:")
    print(df.head())
    
    return df

def train_model(df):
    """
    Trains a Gradient Boosting Classifier model and saves it.

    Args:
        df (pandas.DataFrame): The DataFrame containing the training data.
    """
    print("\nStarting model training...")
    
    # Define features (X) and target (y)
    features = ['Mileage', 'Temperature', 'Vibration', 'HoursOperated']
    target = 'Failure'
    
    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train the Gradient Boosting Classifier
    # These parameters are a good starting point
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Evaluate the model
    print("\nModel evaluation:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    model_filename = 'maintenance_model.pkl'
    print(f"\nSaving trained model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("Model saved successfully.")

    # Save features to a file for the API to use
    features_filename = 'model_features.pkl'
    joblib.dump(features, features_filename)
    print(f"Model features saved to {features_filename}")


if __name__ == "__main__":
    # --- Main execution block ---
    
    # 1. Generate Data
    data_filename = 'vehicle_data.csv'
    if not os.path.exists(data_filename):
        vehicle_df = simulate_vehicle_data()
        vehicle_df.to_csv(data_filename, index=False)
        print(f"\nData saved to {data_filename}")
    else:
        print(f"Loading existing data from {data_filename}...")
        vehicle_df = pd.read_csv(data_filename)

    # 2. Train Model
    train_model(vehicle_df)
