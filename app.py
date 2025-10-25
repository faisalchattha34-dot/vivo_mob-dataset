import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # To save and load the model

# Assume the dataframe 'df' and the trained 'model' are available from the notebook execution
# In a real app.py, you would load the data and the trained model


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

# Assume 'selected_features', 'categorical_features', and 'model' are defined globally or loaded
# In a real app, these would be loaded or defined based on how the model was trained

# For demonstration purposes, let's assume the model and preprocessor are saved as joblib files
# We need to save them from the notebook environment first

# --- Placeholder for loading model and preprocessor ---
# model = joblib.load('model.pkl')
# selected_features = joblib.load('selected_features.pkl')
# categorical_features = joblib.load('categorical_features.pkl')
# --- End Placeholder ---

# For now, let's use the model and features from the current notebook session
# In a production app, you would not rely on the notebook session

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        input_data = json_data['features'] # Assuming input is in a 'features' key

        # Create a pandas DataFrame from the input data
        # Ensure the order of columns matches the selected_features used during training
        # We need to handle the 'Name' column separately as it was not used for training the model
        # and is not in selected_features
        # Assuming input_data is a dictionary like {'Name': '...', 'RAM': ..., 'Storage': ..., 'Camera': '...', 'Battery': '...'}

        # Apply feature engineering to extract RAM and Storage if 'Memory' is provided instead
        if 'Memory' in input_data and 'RAM' not in input_data and 'Storage' not in input_data:
             ram, storage = extract_memory_features(input_data.get('Memory'))
             input_data['RAM'] = ram
             input_data['Storage'] = storage
             del input_data['Memory'] # Remove original Memory feature if new ones are created


        # Create DataFrame for prediction
        # We need to ensure all selected_features are present in input_data, even if None
        prediction_input = {feature: input_data.get(feature) for feature in selected_features}
        input_df_features = pd.DataFrame([prediction_input])


        # Make the prediction
        predicted_price = model.predict(input_df_features)

        return jsonify({'predicted_price': predicted_price[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # This is for running the Flask app directly. In production, use a production-ready server.
    # To run this in Colab, you might need to use ngrok or a similar service to expose the port.
    # Alternatively, you can call the predict function directly for testing within Colab.
    print("Flask app started. Access the prediction endpoint at /predict via POST requests.")
    # app.run(host='0.0.0.0', port=5000) # Uncomment to run the Flask app
