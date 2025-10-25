import streamlit as st
import joblib
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Mobile Price Prediction", layout="centered")

st.title("📱 Mobile Price Prediction")
st.write("Enter phone specs and click **Predict** to estimate the price.")

# ------- Config / defaults -------
MODEL_PATH = "model.pkl"         # your confirmed model filename
PREPROCESSOR_PATHS = ["preprocessor.pkl", "preprocessor.joblib", "preprocessor.sav"]
LABEL_ENCODER_PATHS = ["label_encoder.pkl", "label_encoder.joblib"]
# Default brand list used if no preprocessor/encoder is provided
DEFAULT_BRANDS = [
    "Samsung","Vivo","Xiaomi","Infinix","Oppo","Realme",
    "OnePlus","Huawei","Motorola","Nokia","Tecno","Other"
]

# ------- Load model -------
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None, f"Model file not found at: {path}"
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {e}"

model, model_err = load_model(MODEL_PATH)
if model_err:
    st.error(model_err)
    st.stop()

# ------- Try loading a saved preprocessor (optional) -------
def try_load_preprocessor(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                prep = joblib.load(p)
                return prep, p
            except Exception:
                # continue to next
                pass
    return None, None

preprocessor, preproc_path = try_load_preprocessor(PREPROCESSOR_PATHS)

# If no preprocessor saved, build a fallback ColumnTransformer
def build_fallback_preprocessor(brands):
    # OneHotEncoder with fixed categories so we can transform without fitting
    ohe = OneHotEncoder(categories=[brands], handle_unknown="ignore", sparse=False)
    # ColumnTransformer expects column indices or names at transform time; we'll call ohe.transform manually
    return {"type": "fallback", "ohe": ohe, "brands": brands}

if preprocessor is None:
    preprocessor = build_fallback_preprocessor(DEFAULT_BRANDS)
    preproc_path = "fallback (in-memory)"

# ------- User inputs -------
with st.form("input_form"):
    brand = st.selectbox("Brand", preprocessor["brands"] if preprocessor.get("type") == "fallback" else DEFAULT_BRANDS)
    ram = st.number_input("RAM (GB)", min_value=1, max_value=64, value=4, step=1)
    storage = st.number_input("Storage (GB)", min_value=8, max_value=2048, value=64, step=8)
    camera = st.number_input("Rear Camera (MP)", min_value=0, max_value=200, value=12, step=1)
    submitted = st.form_submit_button("Predict")

# ------- Prediction helper -------
def preprocess_input(brand, ram, storage, camera, preprocessor):
    """
    Returns a 2D numpy array ready to feed model.predict.
    Handles two scenarios:
      - preprocessor is a fitted sklearn ColumnTransformer / Pipeline -> use .transform
      - preprocessor is our fallback dict -> use its OneHotEncoder with fixed categories
    """
    X_numeric = np.array([[ram, storage, camera]])  # shape (1,3)

    # If user supplied a fitted preprocessor (ColumnTransformer/Pipeline)
    if hasattr(preprocessor, "transform"):
        # Build a single-row dict/array matching the order used during training.
        # We try to be flexible: many preprocessors accept a 2D array with columns in order
        # [Brand, RAM, Storage, Camera]. If your trained preprocessor expects different order,
        # this may fail and we'll surface an error to you.
        try:
            row = np.array([[brand, ram, storage, camera]])
            X_trans = preprocessor.transform(row)
            return X_trans
        except Exception as e:
            raise RuntimeError(f"Saved preprocessor exists (loaded from disk) but failed to transform input: {e}")

    # Otherwise use fallback OHE
    if isinstance(preprocessor, dict) and preprocessor.get("type") == "fallback":
        ohe = preprocessor["ohe"]
        brands = preprocessor["brands"]
        # OneHotEncoder with explicit categories can transform without fitting.
        # It requires a 2D array for brand input.
        brand_arr = np.array([[brand]])
        try:
            brand_ohe = ohe.transform(brand_arr)  # shape (1, n_brands)
        except Exception as e:
            # If the OHE complains because it wasn't fitted, force-create using categories argument
            # The sklearn OneHotEncoder with categories param does not need fit; transform will work.
            raise RuntimeError(f"Fallback OneHotEncoder transform failed: {e}")

        # Concatenate brand_ohe + numeric features in the order expected by our fallback
        X_final = np.hstack([brand_ohe, X_numeric])
        return X_final

    raise RuntimeError("Unknown preprocessor configuration.")

# ------- Run prediction -------
if submitted:
    try:
        X = preprocess_input(brand, ram, storage, camera, preprocessor)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.info("If you used a custom preprocessor during training, save it (joblib.dump) as 'preprocessor.pkl' alongside model.pkl.")
        st.stop()

    # Predict
    try:
        pred = model.predict(X)
        # If model outputs array-like
        pred_value = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        st.success(f"Predicted price (model output): {pred_value:.2f}")
        st.write("Note: model output units depend on what you trained the model to predict (e.g. PKR, USD, or a label).")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

# ------- Footer / info -------
st.markdown("---")
st.write(f"Loaded model: `{MODEL_PATH}`")
st.write(f"Preprocessor: `{preproc_path}` — using fallback encoder if not present.")
st.caption("If predictions look wrong, ensure your `model.pkl` was trained with features in exactly this order: [Brand, RAM, Storage, Camera].")

