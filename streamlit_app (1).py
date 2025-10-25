import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib

# -----------------------------
# Load trained model pipeline
# -----------------------------
# The pipeline should include all preprocessing + model
# Example: OneHotEncoding + RandomForestRegressor
pipeline = joblib.load("pipeline.pkl")  # Make sure pipeline.pkl is in the same folder

# Load selected features list
selected_features = joblib.load("selected_features.pkl")  # Same as used during training

# -----------------------------
# Function to extract RAM and Storage
# -----------------------------
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

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Mobile Price Prediction")

# User Inputs
brand = st.selectbox("Select Brand", ["Vivo", "Samsung", "Xiaomi", "Oppo", "Realme", "Apple", "OnePlus", "Motorola", "Infinix"])
ram = st.number_input("RAM (GB)", min_value=1, max_value=32, value=4)
storage = st.number_input("Storage (GB)", min_value=8, max_value=1024, value=64)
camera = st.text_input("Camera (MP)", value="50")
battery = st.number_input("Battery (mAh)", min_value=1000, max_value=10000, value=5000)

# Optional Memory input (like "4GB RAM + 64GB")
memory_input = st.text_input("Memory (optional, e.g., '4GB RAM + 64GB')", "")

if memory_input:
    ram_extracted, storage_extracted = extract_memory_features(memory_input)
    if ram_extracted:
        ram = ram_extracted
    if storage_extracted:
        storage = storage_extracted

# Button to predict
if st.button("Predict Price"):
    # Prepare input dataframe
    input_dict = {
        "Brand": brand,
        "RAM": ram,
        "Storage": storage,
        "Camera": camera,
        "Battery": battery
    }

    # Ensure all selected_features exist
    prediction_input = {feature: input_dict.get(feature, 0) for feature in selected_features}
    input_df = pd.DataFrame([prediction_input])

    # Make prediction
    predicted_price = pipeline.predict(input_df)[0]

    st.success(f"Predicted Mobile Price: PKR {predicted_price:,.0f}")
