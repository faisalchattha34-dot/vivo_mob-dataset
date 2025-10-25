# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib

st.title("Mobile Price Prediction App")

# Function to extract RAM and Storage from 'Memory' string
def extract_memory_features(memory_str):
    ram = None
    storage = None
    if isinstance(memory_str, str):
        ram_match = re.search(r'(\d+)\s*GB\s*RAM', memory_str, re.IGNORECASE)
        if ram_match:
            ram = int(ram_match.group(1))

        storage_match = re.search(r'(\d+)\s*GB', memory_str, re.IGNORECASE)
        if storage_match:
            potential_storage = int(storage_match.group(1))
            if potential_storage != ram:
                storage = potential_storage
            elif ram is None:
                storage = potential_storage

        storage_range_match = re.search(r'(\d+)\s*/\s*(\d+)\s*GB', memory_str, re.IGNORECASE)
        if storage_range_match:
            storage = max(int(storage_range_match.group(1)), int(storage_range_match.group(2)))

    return ram, storage

# --- Load saved model and features ---
# Make sure you have saved these in your notebook/environment
model = joblib.load("model.pkl")  # Your trained RandomForestRegressor
selected_features = joblib.load("selected_features.pkl")  # List of features used for training

# Upload CSV or single prediction
uploaded_file = st.file_uploader("Upload CSV file (optional)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

# Manual input for single prediction
st.subheader("Predict Single Mobile Price")
input_data = {}
for feature in selected_features:
    value = st.text_input(f"Enter {feature}", "")
    input_data[feature] = value

# Handle 'Memory' column specially if user inputs it
if 'Memory' in input_data and 'RAM' not in input_data and 'Storage' not in input_data:
    ram, storage = extract_memory_features(input_data.get('Memory'))
    input_data['RAM'] = ram
    input_data['Storage'] = storage
    del input_data['Memory']

if st.button("Predict Price"):
    # Prepare DataFrame for prediction
    prediction_input = {feature: input_data.get(feature) for feature in selected_features}
    input_df = pd.DataFrame([prediction_input])
    
    # Make prediction
    predicted_price = model.predict(input_df)
    st.success(f"Predicted Price: {predicted_price[0]}")
