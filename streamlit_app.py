import streamlit as st
import pandas as pd
import joblib
import re

# Rest of the app code will go here

# Function to extract numerical features from Memory (copied from the notebook)
def extract_memory_features(memory_str):
    ram = None
    storage = None
    if isinstance(memory_str, str):
        # Look for RAM (e.g., 4GB RAM, 8GB RAM)
        ram_match = re.search(r'(\d+)\s*GB\s*RAM', memory_str, re.IGNORECASE)
        if ram_match:
            ram = int(ram_match.group(1))

        # Look for Storage (e.g., 128GB Built-in, 256GB)
        storage_match = re.search(r'(\d+)\s*GB', memory_str, re.IGNORECASE)
        if storage_match:
            # Ensure we don't pick up the RAM value again if it's the first number
            potential_storage = int(storage_match.group(1))
            if potential_storage != ram: # Simple check to avoid using RAM as storage if it appears first
                 storage = potential_storage
            elif ram is None and potential_storage is not None: # If no RAM found, the first number might be storage
                 storage = potential_storage


        # More specific pattern for storage like '128/256GB'
        storage_range_match = re.search(r'(\d+)\s*/\s*(\d+)\s*GB', memory_str, re.IGNORECASE)
        if storage_range_match:
            # For simplicity, take the larger storage option
            storage = max(int(storage_range_match.group(1)), int(storage_range_match.group(2)))

    return ram, storage

# Load the trained model and data
try:
    model = joblib.load('model.pkl')
    selected_features = joblib.load('selected_features.pkl')
    categorical_features = joblib.load('categorical_features.pkl')
    df = pd.read_csv('vivo_all_features.csv')

    # Convert 'Price' to numeric after loading
    if 'Price' in df.columns and df['Price'].dtype == 'object':
        df['Price'] = df['Price'].str.replace('Rs.', '', regex=False).str.replace('Expected Price', '', regex=False).str.replace(',', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Price'] + [col for col in selected_features if col in df.columns], inplace=True) # Drop rows with NaN in price or selected features

except FileNotFoundError:
    st.error("Error: Model or data files not found. Please ensure 'model.pkl', 'selected_features.pkl', 'categorical_features.pkl', and 'vivo_all_features.csv' are in the same directory.")
    st.stop() # Stop the app if files are missing
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()

# Step 1: Add a title to the Streamlit app
st.title("Vivo Mobile Price Predictor")

# Step 2: Add a section header for user input
st.header("Enter Phone Specifications")

# Step 3: Include input fields for each of the selected_features
# We need to handle numerical and text features differently

# Identify numerical and text features from selected_features
numerical_input_features = [feature for feature in selected_features if feature in ['RAM', 'Storage']]
text_input_features = [feature for feature in selected_features if feature in ['Camera', 'Battery']]


input_data = {}

# Input fields for numerical features
for feature in numerical_input_features:
    # Use st.number_input for numerical features
    min_val = df[feature].min() if feature in df.columns else 0
    max_val = df[feature].max() if feature in df.columns else 1024 # Set a reasonable max if column not in df
    default_val = df[feature].median() if feature in df.columns else 128 # Use median as a default
    input_data[feature] = st.number_input(f"Enter {feature} (in GB):", min_value=float(min_val), max_value=float(max_val), value=float(default_val), step=1.0)

# Input fields for text features
for feature in text_input_features:
    # Use st.text_input for text features
     default_val = df[feature].mode()[0] if feature in df.columns and not df[feature].mode().empty else ""
     input_data[feature] = st.text_input(f"Enter {feature} specifications:", value=default_val)


# Step 4: Add a button to trigger the price prediction
predict_button = st.button("Predict Price")

# The prediction logic will be added in the next step, triggered by this button

# Step 5: Implement prediction logic
if predict_button:
    try:
        # Create a pandas DataFrame from the input data
        # Ensure the order of columns matches the selected_features used during training
        # We need to handle the 'Name' column separately as it was not used for training the model
        # and is not in selected_features
        prediction_input = {feature: input_data.get(feature) for feature in selected_features}
        input_df_features = pd.DataFrame([prediction_input])

        # Make the prediction
        predicted_price = model.predict(input_df_features)

        # Display the predicted price
        st.subheader("Predicted Price:")
        st.success(f"Rs. {predicted_price[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
