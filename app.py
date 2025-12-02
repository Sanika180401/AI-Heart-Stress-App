import streamlit as st
import numpy as np
from PIL import Image
import mediapipe as mp
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import shap

# -------------------------------------------------------------------------
#  PRE-BUILT ML MODELS (FAKE TRAINING FOR DEMO)
# -------------------------------------------------------------------------

# MLP (scikit-learn version)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)

# Random Forest
rf = RandomForestClassifier()

# XGBoost
xgb = XGBClassifier()

# Fit all models on dummy data (so SHAP + ensemble can run today)
X_demo = np.random.rand(50, 4)
y_demo = np.random.randint(0, 2, 50)

mlp.fit(X_demo, y_demo)
rf.fit(X_demo, y_demo)
xgb.fit(X_demo, y_demo)

# SHAP Explainer for RF (works best)
explainer = shap.TreeExplainer(rf)


# -------------------------------------------------------------------------
# HRV Extraction (Simulated Today)
# -------------------------------------------------------------------------
def extract_hrv_from_face(image):
    """
    Simulated HRV values.
    Replace with real HRV later.
    """
    hr = np.random.randint(70, 100)
    rmssd = np.random.uniform(20, 50)
    sd1 = np.random.uniform(15, 40)
    sd2 = np.random.uniform(30, 80)

    features = np.array([[hr, rmssd, sd1, sd2]])
    return features, hr, rmssd, sd1, sd2


# -------------------------------------------------------------------------
# Ensemble Prediction (MLP + RF + XGBoost)
# -------------------------------------------------------------------------
def ensemble_predict(features):
    p1 = mlp.predict_proba(features)[0][1]
    p2 = rf.predict_proba(features)[0][1]
    p3 = xgb.predict_proba(features)[0][1]

    avg_prob = (p1 + p2 + p3) / 3

    stress = "High" if avg_prob > 0.5 else "Low"
    return avg_prob, stress


# -------------------------------------------------------------------------
# Adaptive Thresholding
# -------------------------------------------------------------------------
def adaptive_decision(prob):
    threshold = 0.55  # simple threshold for demo
    level = "High" if prob >= threshold else "Low"
    return level, threshold


# -------------------------------------------------------------------------
# GPT-like Advice (Offline)
# -------------------------------------------------------------------------
def get_ai_advice(level):
    if level == "High":
        return "Stress seems high. Try deep breathing, hydration, and short breaks."
    else:
        return "Stress is low. Maintain your routine and stay relaxed."


# -------------------------------------------------------------------------
# SHAP Plot Table
# -------------------------------------------------------------------------
def get_shap_table(features):
    shap_values = explainer.shap_values(features)[0]
    df = pd.DataFrame({
        "Feature": ["HR", "RMSSD", "SD1", "SD2"],
        "Contribution": shap_values[0]
    })
    return df


# -------------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------------
st.title("AI Heart Rate & Stress Detection System")

mode = st.selectbox("Choose Input Mode", ["Manual Entry", "Upload Image", "Webcam Capture Image"])
st.markdown("---")

# ---------------------- MANUAL ENTRY ----------------------
if mode == "Manual Entry":
    hr = st.number_input("Enter Heart Rate (BPM)", 40, 200)

    if st.button("Predict"):
        stress = "High" if hr > 90 else "Low"
        st.success(f"Stress: {stress}")
        st.info(get_ai_advice(stress))

# ---------------------- UPLOAD IMAGE ----------------------
elif mode == "Upload Image":
    upl = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if upl:
        img = Image.open(upl)
        st.image(img)

        features, hr, rmssd, sd1, sd2 = extract_hrv_from_face(img)
        prob, stress = ensemble_predict(features)

        st.success(f"Heart Rate: {hr} BPM")
        st.success(f"Stress Prediction: {stress}")

        df = get_shap_table(features)
        st.subheader("Feature Contributions (SHAP)")
        st.table(df)

        st.info(get_ai_advice(stress))


# ---------------------- WEBCAM CAPTURE ----------------------
elif mode == "Webcam Capture Image":
    cam = st.camera_input("Capture Image")
    if cam:
        img = Image.open(cam)
        st.image(img)

        features, hr, rmssd, sd1, sd2 = extract_hrv_from_face(img)
        prob, stress = ensemble_predict(features)

        st.success(f"Heart Rate: {hr} BPM")
        st.success(f"Stress Prediction: {stress}")

        df = get_shap_table(features)
        st.subheader("Feature Contributions (SHAP)")
        st.table(df)

        st.info(get_ai_advice(stress))
