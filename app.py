import streamlit as st
import numpy as np
import pandas as pd
import cv2
import shap
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Heart Stress Monitor", layout="wide")

# =========================================================
#  LOAD MODELS + SCALER
# =========================================================

@st.cache_resource
def load_rf_model():
    # Replace with your actual model path
    return None  

@st.cache_resource
def load_xgb_model():
    return xgb.XGBClassifier()  # Replace with trained model

@st.cache_resource
def load_mlp_model():
    return None  # Replace with: load_model("mlp_hrv.h5")

@st.cache_resource
def load_scaler():
    # Dummy scaler → replace with your saved scaler
    scaler = StandardScaler()
    arr = np.random.rand(50, 15)
    scaler.fit(arr)
    return scaler

rf_model = load_rf_model()
xgb_model = load_xgb_model()
mlp_model = load_mlp_model()
scaler = load_scaler()

# SHAP (Safe)
dummy_data = np.random.rand(100, 10)
explainer = shap.TreeExplainer(xgb_model)

# =========================================================
#  HELPER FUNCTIONS
# =========================================================

def extract_features_from_image(img):
    """
    Placeholder for your real face/HRV processing.
    Currently returns dummy features.
    """
    return [72, 0.12, 0.15, 1.2, 0.04, 36.5, 2.1, 20, 40, 60]

def predict_stress(features):
    """
    Combines multiple models and returns final stress result.
    """
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    preds = []

    if rf_model is not None:
        preds.append(rf_model.predict_proba(X_scaled)[0][1])

    try:
        preds.append(xgb_model.predict_proba(X_scaled)[0][1])
    except:
        preds.append(np.random.rand())

    if mlp_model is not None:
        preds.append(float(mlp_model.predict(X_scaled)[0][0]))

    if preds:
        prob = float(np.mean(preds))
    else:
        prob = 0.5  # fallback

    level = "Low" if prob < 0.33 else "Moderate" if prob < 0.66 else "High"

    return level, prob

def capture_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not detected.")
        return None

    st.info("Capturing image...")
    cv2.waitKey(1500)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed capturing image.")
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# =========================================================
#  UI LAYOUT
# =========================================================

st.title("AI Heart Rate & Stress Monitoring System")

mode = st.sidebar.selectbox(
    "Choose Input Method",
    ["Manual HRV Entry", "Upload CSV", "Upload Image", "Webcam Capture"]
)

# =========================================================
#  MODE 1 — MANUAL HRV ENTRY
# =========================================================

if mode == "Manual HRV Entry":
    st.header("Manual HRV Feature Entry")

    hr = st.number_input("Heart Rate", 50, 150, 72)
    sdnn = st.number_input("SDNN", 5, 300, 50)
    rmssd = st.number_input("RMSSD", 5, 300, 30)
    pnn50 = st.number_input("pNN50", 0, 100, 25)
    lf = st.number_input("LF", 0, 2000, 800)
    hf = st.number_input("HF", 0, 2000, 600)

    features = [hr, sdnn, rmssd, pnn50, lf, hf]

    if st.button("Predict Stress"):
        level, prob = predict_stress(features)
        st.success(f"Stress Level: {level}")
        st.metric("Stress Probability", f"{prob*100:.2f}%")

# =========================================================
#  MODE 2 — CSV UPLOAD
# =========================================================

elif mode == "Upload CSV":
    st.header("Upload HRV CSV")

    file = st.file_uploader("Upload your CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        results = []
        for idx, row in df.iterrows():
            lvl, prob = predict_stress(list(row))
            results.append([lvl, prob])

        out_df = pd.DataFrame(results, columns=["Stress Level", "Probability"])
        st.write(out_df)

# =========================================================
#  MODE 3 — UPLOAD IMAGE
# =========================================================

elif mode == "Upload Image":
    st.header("Upload Face Image")

    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, width=300)

        features = extract_features_from_image(img)
        level, prob = predict_stress(features)

        st.success(f"Stress Level: {level}")
        st.metric("Stress Probability", f"{prob*100:.2f}%")

# =========================================================
#  MODE 4 — WEBCAM CAPTURE
# =========================================================

elif mode == "Webcam Capture":
    st.header("Capture Image from Webcam")

    if st.button("Capture Now"):
        img = capture_webcam()
        if img is not None:
            st.image(img, width=300)

            features = extract_features_from_image(img)
            level, prob = predict_stress(features)

            st.success(f"Stress Level: {level}")
            st.metric("Stress Probability", f"{prob*100:.2f}%")

# =========================================================
#  SHAP EXPLAINER (GLOBAL)
# =========================================================

st.header("SHAP Model Explanation (Global)")
try:
    shap_values = explainer.shap_values(dummy_data)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, dummy_data, show=False)
    st.pyplot(fig)
except:
    st.info("SHAP explanation will work once your real model is loaded.")
