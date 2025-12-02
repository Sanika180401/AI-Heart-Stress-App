import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # Disable all sklearn warnings

# ML Models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# ---------------------------
# Train dummy models
# ---------------------------
X_demo = np.random.rand(100, 4)
y_demo = np.random.randint(0, 2, 100)

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300).fit(X_demo, y_demo)
rf = RandomForestClassifier().fit(X_demo, y_demo)
xgb = XGBClassifier().fit(X_demo, y_demo)

# SHAP Explainer (TreeExplainer works best with RF)
explainer = shap.TreeExplainer(rf)


# ---------------------------
# HRV Extraction (Simulated)
# ---------------------------
def extract_hrv_features(image):
    """
    Generates realistic mock HRV values.
    Replace with real rPPG pipeline later.
    """
    hr = np.random.randint(65, 100)
    rmssd = np.random.uniform(20, 50)
    sd1 = np.random.uniform(15, 40)
    sd2 = np.random.uniform(30, 70)

    features = np.array([[hr, rmssd, sd1, sd2]])
    return features, hr, rmssd, sd1, sd2


# ---------------------------
# Ensemble Prediction
# ---------------------------
def ensemble_predict(features):
    p1 = mlp.predict_proba(features)[0][1]
    p2 = rf.predict_proba(features)[0][1]
    p3 = xgb.predict_proba(features)[0][1]

    prob = (p1 + p2 + p3) / 3
    stress = "High" if prob >= 0.5 else "Low"
    return prob, stress


# ---------------------------
# Adaptive Thresholding
# ---------------------------
def adaptive_threshold(prob):
    threshold = 0.55
    level = "High" if prob >= threshold else "Low"
    return level, threshold


# ---------------------------
# SHAP Table Output
# ---------------------------
def shap_table(features):
    shap_values = explainer.shap_values(features)[0]
    df = pd.DataFrame({
        "Feature": ["HR", "RMSSD", "SD1", "SD2"],
        "Contribution": shap_values
    })
    return df


# ---------------------------
# AI Advice
# ---------------------------
def ai_advice(level):
    if level == "High":
        return "Your stress appears elevated. Try slow breathing, hydration, and relaxing activities."
    else:
        return "Stress level is low. Maintain your routine and stay mindful."


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="AI Stress Detection", layout="centered")
st.title("AI-Based Heart Rate & Stress Detection System")
st.write("Select an input method:")

mode = st.selectbox("Choose Mode:", ["Manual Entry", "Upload Image", "Webcam Capture Image"])

st.markdown("---")

# ======================================================================
# 1. Manual Entry
# ======================================================================
if mode == "Manual Entry":
    hr = st.number_input("Enter Heart Rate (BPM):", 40, 200)

    if st.button("Predict Stress"):
        stress = "High" if hr >= 90 else "Low"
        st.success(f"Stress Level: {stress}")
        st.info(ai_advice(stress))


# ======================================================================
# 2. Upload Image
# ======================================================================
elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        features, hr, rmssd, sd1, sd2 = extract_hrv_features(image)
        prob, stress = ensemble_predict(features)

        st.success(f"Heart Rate: {hr} BPM")
        st.success(f"Stress Level: {stress}")

        st.subheader("Feature Impact (SHAP)")
        df = shap_table(features)
        st.table(df)

        st.info(ai_advice(stress))


# ======================================================================
# 3. Webcam Capture
# ======================================================================
elif mode == "Webcam Capture Image":
    snap = st.camera_input("Capture Image")

    if snap:
        image = Image.open(snap)
        st.image(image, caption="Captured Image", use_column_width=True)

        features, hr, rmssd, sd1, sd2 = extract_hrv_features(image)
        prob, stress = ensemble_predict(features)

        st.success(f"Heart Rate: {hr} BPM")
        st.success(f"Stress Level: {stress}")

        st.subheader("Feature Impact (SHAP)")
        df = shap_table(features)
        st.table(df)

        st.info(ai_advice(stress))
