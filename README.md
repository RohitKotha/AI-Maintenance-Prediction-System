AI Predictive Maintenance SystemThis project implements an end-to-end system for predicting vehicle component failures using machine learning. It includes a data simulator, a model training pipeline, a REST API to serve predictions, and a responsive web dashboard for user interaction.(Feel free to replace the link above with a direct upload of your screenshot to the repository)FeaturesSynthetic Data Generation: A Python script (train.py) simulates realistic sensor data (mileage, temperature, vibration, etc.) for thousands of vehicles.Machine Learning Model: A GradientBoostingClassifier is trained to predict the probability of component failure based on sensor readings.REST API: A lightweight Flask API (app.py) exposes a /predict endpoint to serve real-time predictions from the trained model.Interactive Dashboard: A user-friendly, responsive front-end (index.html) built with HTML, Tailwind CSS, and Chart.js allows users to input data and visualize the failure probability on a gauge.System ArchitectureThe project follows a simple, yet effective, machine learning system design pattern:[1. Data Simulation & Training] -> [2. Saved Model (.pkl)] -> [3. Flask API] <-> [4. Web Dashboard]
Data Simulation & Training (train.py):Generates a vehicle_data.csv file with synthetic sensor data.Trains a scikit-learn model on this data.Saves the trained model as maintenance_model.pkl.Flask API (app.py):Loads the maintenance_model.pkl file on startup.Listens for POST requests with JSON data at the /predict endpoint.Returns a JSON response containing the failure prediction and probability.Web Dashboard (index.html):Provides a form for users to enter vehicle sensor data.On submission, sends a fetch request to the Flask API.Parses the API response and dynamically updates the UI with the prediction message and a visual gauge chart.Getting StartedFollow these instructions to get a copy of the project up and running on your local machine.PrerequisitesPython 3.8+GitInstallation & SetupClone the repository:git clone https://github.com/your-username/AI-Maintenance-Prediction.git
cd AI-Maintenance-Prediction
Create a virtual environment (recommended):# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required Python packages:pip install -r requirements.txt
Running the SystemThe system requires two separate terminal processes to run concurrently: one for the back-end API and one for the front-end.Terminal 1: Generate Data & Train Model:If you haven't already, run the training script. This only needs to be done once.python train.py
This will generate vehicle_data.csv and maintenance_model.pkl.Terminal 1: Start the API Server:Launch the Flask API.python app.py
The API will now be running at http://127.0.0.1:5000.Terminal 2: Launch the Dashboard:The simplest way to serve the index.html file is by using Python's built-in HTTP server.python -m http.server
Now, open your web browser and navigate to http://localhost:8000.You should now see the dashboard and be able to make predictions.API EndpointURL: /predictMethod: POSTData Format: JSONExample Payload:{
  "Mileage": 85000,
  "Temperature": 95.5,
  "Vibration": 4.2,
  "HoursOperated": 2150
}
Success Response:{
  "prediction": 1,
  "probability": 0.8872,
  "message": "Warning: Maintenance recommended within the next 500 miles."
}
