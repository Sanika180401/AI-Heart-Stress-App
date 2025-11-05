import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ------------------------------
# LOAD ALL MODELS & PREPROCESSORS
# ------------------------------
@st.cache_resource
def load_all():
    pre = joblib.load("models/preprocess.joblib")
    rf = joblib.load("models/rf_hrv.joblib")
    xgb = joblib.load("models/xgb_hrv.joblib")
    mlp = load_model("models/mlp_hrv.h5")
    thr = np.load("models/adaptive_thresholds.npy", allow_pickle=True).item()
    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all()
cols = pre['feature_cols']
sc = pre['scaler']
imp = pre['imputer']

# ------------------------------
# STREAMLIT PAGE SETUP
# ------------------------------
st.set_page_config(page_title="AI Heart & Stress Monitor", layout="wide")
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring System")
st.markdown("*Early Heart Attack Risk Prediction using Non-Wearable Sensors*")

mode = st.radio("Select Input Mode", ["Manual Input", "Upload CSV File"])

# ------------------------------
# MANUAL INPUT MODE
# ------------------------------
if mode == "Manual Input":
    st.subheader("Enter HRV Feature Values")
    cols_layout = st.columns(len(cols))

    inputs = {}
    for i, c in enumerate(cols):
        with cols_layout[i % len(cols_layout)]:
            inputs[c] = st.number_input(c, value=50.0, step=0.1)

    df_in = pd.DataFrame([inputs])

# ------------------------------
# CSV UPLOAD MODE
# ------------------------------
else:
    uploaded_file = st.file_uploader("Upload your HRV CSV file", type=["csv"])
    if uploaded_file:
        df_in = pd.read_csv(uploaded_file)
        st.dataframe(df_in.head())
    else:
        df_in = None

# ------------------------------
# PREDICTION LOGIC
# ------------------------------
if st.button("Predict Stress Level") and df_in is not None:
    X = imp.transform(df_in[cols])
    X = sc.transform(X)

    p_rf = rf.predict_proba(X)[:, 1]
    p_xgb = xgb.predict_proba(X)[:, 1]
    p_mlp = mlp.predict(X).flatten()
    p_avg = (p_rf + p_xgb + p_mlp) / 3

    low, high = thr['low'], thr['high']

    def classify(p):
        if p < low: return "Low Stress"
        elif p < high: return "Moderate Stress"
        else: return "High Stress"

    df_in['Stress_Probability'] = p_avg
    df_in['Stress_Level'] = [classify(p) for p in p_avg]

    st.success("Prediction Completed")
    st.dataframe(df_in)

    fig, ax = plt.subplots()
    ax.bar(['RF', 'XGB', 'MLP', 'Ensemble'], [p_rf[0], p_xgb[0], p_mlp[0], p_avg[0]])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted Stress Probability")
    st.pyplot(fig)

    # ------------------------------
    # OPTIONAL: GPT EXPLANATION
    # ------------------------------
    st.markdown("AI Explanation (LLM Integration)")
    use_llm = st.checkbox("Generate Explanation using GPT")
    if use_llm:
        import openai
        key = st.text_input("Enter your OpenAI API Key:", type="password")
        if key:
            openai.api_key = key
            prompt = f"The stress prediction probability is {p_avg[0]:.2f}. Based on HRV features {inputs}, explain the possible physiological reason and suggest 3 personalized stress management tips."
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7
                )
                st.info(response['choices'][0]['message']['content'])
            except Exception as e:
                st.error(f"OpenAI API error: {e}")
