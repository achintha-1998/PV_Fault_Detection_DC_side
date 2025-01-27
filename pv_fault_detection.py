import streamlit as st
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import tempfile
import requests
import matplotlib.pyplot as plt
import os
import subprocess
import sys

# Try to install scikit-learn if not present
try:
    import sklearn
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

# Initialize global variables
scaler = StandardScaler()
local_model = None
SERVER_URL = "http://your-server-url.com/update_global_model"  # Replace with actual server URL


# Simulate real-time sensor data
def get_sensor_data():
    voltage = random.uniform(300, 600)
    current = random.uniform(0, 10)
    irradiance = random.uniform(0, 1000)
    temperature = random.uniform(10, 40)
    power = voltage * current
    return voltage, current, irradiance, temperature, power


# Load the global model
def load_global_model():
    global_model_path = "global_model.pkl"
    if os.path.exists(global_model_path):
        return joblib.load(global_model_path)
    else:
        st.error("Global model not found. Please ensure 'global_model.pkl' is available.")
        return None


# Train the local model
def train_local_model():
    global local_model, scaler
    dataset_path = "PV_simulated_data.csv"
    if not os.path.exists('PV_simulated_data.csv'):
        st.error("Dataset not found. Please ensure 'PV_simulated_data.csv' is available.")
        return None

    # Load and preprocess the dataset
    df = pd.read_csv(dataset_path)
    X = df[["Voltage (V)", "Current (A)", "Irradiance (W/m²)", "Temperature (°C)", "Power (W)"]]
    y = df["Fault Condition"]

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Train the local model
    local_model = RandomForestClassifier(n_estimators=100, random_state=42)
    local_model.fit(X_scaled, y)

    # Save the trained local model
    joblib.dump(local_model, "local_model.pkl")
    st.success("Local model trained and saved successfully!")

    # Send local model weights to the server for global aggregation
    send_model_to_server(local_model)

    return local_model


# Send local model updates to the server
def send_model_to_server(local_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_model_file:
        joblib.dump(local_model, temp_model_file.name)
        with open(temp_model_file.name, "rb") as model_file:
            files = {"model": model_file}
            response = requests.post(SERVER_URL, files=files)
            if response.status_code == 200:
                st.success("Local model successfully sent to the server.")
            else:
                st.error(f"Failed to send the model to the server. Error: {response.text}")


# Generate SHAP explanations
def explain_shap(local_model, input_data):
    explainer = shap.TreeExplainer(local_model)
    shap_values = explainer.shap_values(input_data)
    return shap_values


# Update sensor data and make predictions
def update_sensor_data():
    # Simulate sensor data
    voltage, current, irradiance, temperature, power = get_sensor_data()
    sensor_data = np.array([[voltage, current, irradiance, temperature, power]])

    # Scale the input data
    input_data = scaler.transform(sensor_data)

    # Load the global model
    global_model = load_global_model()
    if not global_model:
        return "Failed to load global model.", "N/A", None

    # Make predictions using the global model
    prediction = global_model.predict(input_data)[0]
    fault_status = "FAULT" if prediction == 1 else "NORMAL"

    # Explain using SHAP
    shap_values = explain_shap(local_model, input_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    shap.summary_plot(shap_values[1], input_data,
                      feature_names=["Voltage", "Current", "Irradiance", "Temperature", "Power"], show=False)
    plt.savefig(temp_file.name)
    plt.close()

    return f"Voltage: {voltage}V\nCurrent: {current}A\nIrradiance: {irradiance}W/m²\nTemperature: {temperature}°C\nPower: {power}W", \
        fault_status, temp_file.name


# Streamlit app UI
def app():
    st.title("PV System Fault Detection - Local & Global Models with FL")
    st.sidebar.title("Controls")

    # Retrain the local model
    if st.sidebar.button("Retrain Local Model"):
        train_local_model()

    # Display sensor data and predictions
    sensor_data, fault_prediction, shap_plot = update_sensor_data()

    st.subheader("Real-Time Sensor Data")
    st.text(sensor_data)

    st.subheader("Fault Prediction")
    st.text(f"Prediction: {fault_prediction}")

    st.subheader("SHAP Summary Plot")
    if shap_plot:
        st.image(shap_plot)


# Run the app
if __name__ == "__main__":
    if not os.path.exists("local_model.pkl"):
        train_local_model()
    else:
        local_model = joblib.load("local_model.pkl")

    app()
