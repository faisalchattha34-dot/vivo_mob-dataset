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




       
