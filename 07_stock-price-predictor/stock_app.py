"""
Monitoring with Prometheus and Grafana - Integrate Prometheus with Streamlit

Utilize the streamlit-extras Prometheus integration
This exposes metrics at the /metrics endpoint, which Prometheus can scrape.
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
from model import LSTMModel  # Ensure this matches your model's class name and location
from utils import preprocess_input  # Function to preprocess user input
import joblib
from streamlit_extras import prometheus
from prometheus_client import Counter, start_http_server

scaler = joblib.load("scaler.pkl")

# Load the trained model
device = device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

st.title("Stock Price Predictor")

# User input
user_input = st.text_input("Enter stock prices (comma-separated):", "100,101,102,103,104")


# PREDICTION_COUNT = Counter(
#     "prediction_count",
#     "Number of predictions made",
#     registry=prometheus.streamlit_registry()
# )

if st.button("Predict"):
    try:
        # Step 1: Parse user input
        input_sequence = [float(i) for i in user_input.split(",")]

        # Step 2: Scale the input using the same scaler used during training
        input_scaled = scaler.transform(np.array(input_sequence).reshape(-1, 1))

        # Step 3: Reshape for LSTM model: (batch_size, seq_length, input_size)
        input_tensor = torch.tensor(input_scaled.reshape(1, -1, 1), dtype=torch.float32).to(device)

        # Step 4: Predict
        model.eval()
        with torch.no_grad():
            pred_scaled = model(input_tensor)

        # Step 5: Inverse transform the prediction
        pred_price = scaler.inverse_transform(pred_scaled.cpu().numpy())

        # Step 6: Show result
        st.write(f"Predicted next price: {pred_price[0][0]:.2f}")
    
    except Exception as e:
        st.error(f"Error: {e}")
    # PREDICTION_COUNT.inc()

