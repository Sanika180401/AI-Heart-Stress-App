import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import openai

# ----------------------------
# PATH SETUP (Avoid FileNotFound Errors)
# ----------------------------
BASE_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_PATH, "models")

def safe_load(filename):
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
    return path

# ----------------------------
# Load All Models & Assets
# ----------------------------
@st.cache_resource
def load_all():
    pre = joblib.load(safe_load("preprocess.joblib"))
    rf = joblib.load(safe_load("rf_hrv.joblib"))
    xgb = joblib.load(safe_load("xgb_hrv.joblib"))
    mlp = load_model(safe_load("mlp_hrv_v2.h5"))
    thr = np.load(safe_load("adaptive_thresholds.npy"), allow_pickle=True).item()
    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all()

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("### Early Heart Attack Risk Prediction using Non-Wearable Sensors")

option = st.radio("Select Input Method:", ("Manual Entry", "Upload CSV"))

# ----------------------------
# LLM Explanation Function
# ----------------------------
openai.api_key = st.secrets.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

def gpt_explain(stress_level, prob):
    """Generate explanation via GPT."""
    prompt = f"""
    The model predicted a stress probability of {prob*100:.2f}%.
    Stress level category: {stress_level}.
    Provide a short, human-understandable explanation and 3 personalized tips to reduce stress.
    """
    try:
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT explanation unavailable: {e})"

# ----------------------------
# Prediction Function
# ----------------------------
def predict_stress(input_data):
    input_scaled = pre['scaler'].transform(input_data)
    p_rf = rf.predict_proba(input_scaled)[:, 1]
    p_xgb = xgb.predict_proba(input_scaled)[:, 1]
    p_mlp = mlp.predict(input_scaled).flatten()
    avg_pred = (p_rf + p_xgb + p_mlp) / 3

    low, high = thr['low'], thr['high']
    if avg_pred < low:
        level = "Low Stress"
    elif avg_pred < high:
        level = "Moderate Stress"
    else:
        level = "High Stress"
    return avg_pred[0], level

# ----------------------------
# Manual Input Section
# ----------------------------
if option == "Manual Entry":
    st.subheader("Enter HRV Features")
    col1, col2 = st.columns(2)

    with col1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 40, 200, 75)
        sdnn = st.number_input("SDNN (ms)", 10, 200, 50)
        rmssd = st.number_input("RMSSD (ms)", 10, 200, 30)
        sd1 = st.number_input("SD1 (ms)", 5.0, 100.0, 20.0)

    with col2:
        pnn50 = st.number_input("pNN50 (%)", 0, 100, 20)
        lf = st.number_input("LF Power (ms²)", 100, 5000, 800)
        hf = st.number_input("HF Power (ms²)", 100, 5000, 600)
        sd2 = st.number_input("SD2 (ms)", 5.0, 200.0, 40.0)

    if st.button("Predict Stress Level"):
        input_data = np.array([[mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2]])
        prob, level = predict_stress(input_data)

        st.subheader("Prediction Result")
        st.metric("Stress Probability", f"{prob*100:.2f}%")
        if level == "Low Stress":
            st.success(level)
        elif level == "Moderate Stress":
            st.warning(level)
        else:
            st.error(level)

        st.write("GPT Explanation:")
        st.info(gpt_explain(level, prob))

# ----------------------------
# CSV Upload Section
# ----------------------------
elif option == "Upload CSV":
    st.write("Upload CSV with features: mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        try:
            input_scaled = pre['scaler'].transform(df)
            p_rf = rf.predict_proba(input_scaled)[:, 1]
            p_xgb = xgb.predict_proba(input_scaled)[:, 1]
            p_mlp = mlp.predict(input_scaled).flatten()
            avg_pred = (p_rf + p_xgb + p_mlp) / 3

            low, high = thr['low'], thr['high']
            df['Stress_Probability'] = avg_pred
            df['Stress_Level'] = pd.cut(avg_pred, bins=[0, low, high, 1], labels=["Low", "Moderate", "High"])
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv_out, "stress_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ----------------------------
# Optional SHAP Visualization
# ----------------------------
st.markdown("---")
if st.button("Show Feature Importance (SHAP)"):
    explainer = shap.TreeExplainer(rf)
    sample = np.random.choice(len(pre['scaler'].mean_), size=min(50, len(pre['scaler'].mean_)), replace=False)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values[1], sample, feature_names=pre['scaler'].feature_names_in_)
    st.pyplot(plt)
