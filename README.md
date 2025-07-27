# 🔧 AI Predictive Maintenance System

This project uses **machine learning** to predict vehicle component failures, helping to anticipate and prevent costly breakdowns. It includes a **Python Flask API** backend for prediction and an **HTML/JavaScript** frontend for displaying results.

---

## 🚀 How to Run

### 1. Prerequisites

Ensure you have the following installed:

- Python **3.8+**
- Git

---

### 2. Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/AI-Maintenance-Prediction.git
cd AI-Maintenance-Prediction

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Running the Application

You’ll need **two terminals**:

#### Terminal 1 – Start the Backend (API)

```bash
# Train and save the ML model (run once)
python train.py

# Start the Flask API
python app.py
```

#### Terminal 2 – Start the Frontend

```bash
# Serve the frontend via HTTP server
python -m http.server
```

Then, open your browser and visit:  
👉 **http://localhost:8000**

---

## 📦 Required Libraries

The following Python packages are required and are listed in `requirements.txt`:

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `joblib`  
- `Flask`  
- `Flask-Cors`

---

## 📝 Notes

- This project is ideal for demonstrating predictive analytics in automotive diagnostics.
- It can be extended to real-time sensor data integration using tools like MQTT or WebSockets.
