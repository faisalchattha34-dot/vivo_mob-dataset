import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# -----------------------------
# Load existing model and features
# -----------------------------
model = joblib.load("model.pkl")
selected_features = joblib.load("selected_features.pkl")
categorical_features = joblib.load("categorical_features.pkl")

numeric_features = [f for f in selected_features if f not in categorical_features]

# -----------------------------
# Load your training data
# -----------------------------
# Make sure your training CSV has the same columns as selected_features + target Price
df = pd.read_csv("train_data.csv")  # replace with your CSV path
X_train = df[selected_features]
y_train = df["Price"]

# -----------------------------
# Create preprocessor and pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# Fit the pipeline with training data
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Save the pipeline
# -----------------------------
joblib.dump(pipeline, "pipeline.pkl")

print("pipeline.pkl created successfully! You can now use it in Streamlit.")
