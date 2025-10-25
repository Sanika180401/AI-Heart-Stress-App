# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_models():
    preprocess = joblib.load('preprocess.joblib')
    rf_model = joblib.load('random_forest_hrv.joblib')
    xgb_model = joblib.load('xgboost_hrv.joblib')
    return preprocess, rf_model, xgb_model

preprocess, rf_model, xgb_model = load_models()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI-Based Heart & Stress Monitoring", layout="centered")

st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("### Early Heart Attack Risk Prediction using Non-Wearable Sensors")
st.write("Enter HRV (Heart Rate Variability) features manually or upload a CSV file to predict stress level.")

option = st.radio("Select Input Method:", ("Manual Entry", "Upload CSV"))

# ------------------------------
# Manual Input
# ------------------------------
if option == "Manual Entry":
    col1, col2 = st.columns(2)

    with col1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 40, 200, 75)
        sdnn = st.number_input("SDNN (ms)", 10, 200, 50)
        rmssd = st.number_input("RMSSD (ms)", 10, 200, 30)

    with col2:
        pnn50 = st.number_input("pNN50 (%)", 0, 100, 20)
        lf = st.number_input("LF Power (ms²)", 100, 5000, 800)
        hf = st.number_input("HF Power (ms²)", 100, 5000, 600)
        lf_hf = st.number_input("LF/HF Ratio", 0.1, 10.0, 1.5)

    if st.button("Predict Stress Level"):
        input_data = np.array([[mean_hr, sdnn, rmssd, pnn50, lf, hf, lf_hf]])
        input_scaled = preprocess['scaler'].transform(input_data)

        rf_pred = rf_model.predict_proba(input_scaled)[0, 1]
        xgb_pred = xgb_model.predict_proba(input_scaled)[0, 1]
        avg_pred = (rf_pred + xgb_pred) / 2

        st.subheader("Predicted Stress Probability:")
        st.metric(label="Stress Probability", value=f"{avg_pred*100:.2f}%")

        if avg_pred < 0.4:
            st.success("Low Stress - Normal Condition")
        elif avg_pred < 0.7:
            st.warning("Moderate Stress - Monitor Regularly")
        else:
            st.error("High Stress / Potential Risk! Seek Medical Attention.")

# ------------------------------
# CSV Upload
# ------------------------------
elif option == "Upload CSV":
    st.write("Upload a CSV file containing HRV features (mean_hr, sdnn, rmssd, pnn50, lf, hf, lf_hf):")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        try:
            X = preprocess.transform(df)
            rf_pred = rf_model.predict_proba(X)[:, 1]
            xgb_pred = xgb_model.predict_proba(X)[:, 1]
            avg_pred = (rf_pred + xgb_pred) / 2

            df['Stress_Probability'] = avg_pred
            df['Stress_Level'] = pd.cut(avg_pred,
                                        bins=[0, 0.4, 0.7, 1],
                                        labels=["Low", "Moderate", "High"])

            st.subheader("Prediction Results:")
            st.dataframe(df)

            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv_out, "stress_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
