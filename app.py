import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================================================
# STREAMLIT CONFIG — MUST BE FIRST
# ===============================================================
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("### Early Heart Attack Risk Prediction using Non-Wearable Sensors")

# ===============================================================
# PATH SETUP
# ===============================================================
BASE_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_PATH, "models")

def safe_load(filename):
    """Safely locate model files."""
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        raise FileNotFoundError(f"Missing file: {path}")
    return path

# ===============================================================
# LOAD MODELS
# ===============================================================
@st.cache_resource
def load_all():
    """Load all models and preprocessors safely."""
    pre = joblib.load(safe_load("preprocess.joblib"))
    rf = joblib.load(safe_load("rf_hrv.joblib"))

    # Load XGBoost if available
    try:
        xgb = joblib.load(safe_load("xgb_hrv.joblib"))
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")
        xgb = None

    # Load MLP — new clean version compatible with TF 2.19
    try:
        mlp = load_model(safe_load("mlp_hrv_clean_fixed.keras"), compile=False)
    except Exception as e:
        st.warning(f"Could not load MLP model: {e}")
        mlp = None

    thr = np.load(safe_load("adaptive_thresholds.npy"), allow_pickle=True).item()
    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all()

# ===============================================================
# GPT EXPLANATION (OPTIONAL)
# ===============================================================
try:
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    openai = None
    st.warning("OpenAI not configured — GPT explanations disabled.")

def gpt_explain(stress_level, prob):
    """Optional GPT-based stress explanation."""
    if not openai or not openai.api_key:
        return "(GPT explanation unavailable — missing API key.)"
    prompt = f"""
    The model predicted a stress probability of {prob*100:.2f}%.
    Detected stress level: {stress_level}.
    Provide a short human explanation and 3 simple stress-relief tips.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a calm, medical AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT explanation unavailable: {e})"

# ===============================================================
# PREDICTION FUNCTION
# ===============================================================
def predict_stress(input_data):
    """Predict stress using RF + XGB + MLP ensemble."""
    input_scaled = pre["scaler"].transform(input_data)
    preds = [rf.predict_proba(input_scaled)[:, 1]]

    if xgb is not None:
        preds.append(xgb.predict_proba(input_scaled)[:, 1])
    if mlp is not None:
        preds.append(mlp.predict(input_scaled).flatten())

    avg_pred = np.mean(preds, axis=0)
    low, high = thr["low"], thr["high"]

    if avg_pred < low:
        level = "Low Stress"
    elif avg_pred < high:
        level = "Moderate Stress"
    else:
        level = "High Stress"

    return float(avg_pred[0]), level

# ===============================================================
# INPUT OPTIONS
# ===============================================================
option = st.radio("Select Input Method:", ("Manual Entry", "Upload CSV"))

# ===============================================================
# MANUAL ENTRY
# ===============================================================
if option == "Manual Entry":
    st.subheader("Enter HRV & Physiological Features")

    col1, col2 = st.columns(2)
    with col1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 40, 200, 75)
        sdnn = st.number_input("SDNN (ms)", 10, 200, 50)
        rmssd = st.number_input("RMSSD (ms)", 10, 200, 30)
        sd1 = st.number_input("SD1 (ms)", 5.0, 100.0, 20.0)
        temp = st.number_input("Skin Temperature (°C)", 30.0, 40.0, 36.5)

    with col2:
        pnn50 = st.number_input("pNN50 (%)", 0, 100, 20)
        lf = st.number_input("LF Power (ms²)", 100, 5000, 800)
        hf = st.number_input("HF Power (ms²)", 100, 5000, 600)
        sd2 = st.number_input("SD2 (ms)", 5.0, 200.0, 40.0)
        eda = st.number_input("Electrodermal Activity (µS)", 0.1, 10.0, 2.0)

    input_data = np.array([[mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda]])

    if st.button("Predict Stress Level"):
        prob, level = predict_stress(input_data)
        st.subheader("Prediction Result")
        st.metric("Stress Probability", f"{prob*100:.2f}%")

        if level == "Low Stress":
            st.success(level)
        elif level == "Moderate Stress":
            st.warning(level)
        else:
            st.error(level)

        st.markdown("#### GPT Explanation:")
        st.info(gpt_explain(level, prob))

# ===============================================================
# CSV UPLOAD
# ===============================================================
elif option == "Upload CSV":
    st.write("Upload CSV with 10 columns: mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda")
    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        try:
            input_scaled = pre["scaler"].transform(df)
            preds = [
                rf.predict_proba(input_scaled)[:, 1],
                xgb.predict_proba(input_scaled)[:, 1] if xgb else 0,
                mlp.predict(input_scaled).flatten() if mlp else 0
            ]
            avg_pred = np.mean(preds, axis=0)
            low, high = thr["low"], thr["high"]

            df["Stress_Probability"] = avg_pred
            df["Stress_Level"] = pd.cut(avg_pred, bins=[0, low, high, 1],
                                        labels=["Low", "Moderate", "High"])

            st.dataframe(df)
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv_out,
                               "stress_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ===============================================================
# SHAP FEATURE IMPORTANCE
# ===============================================================
st.markdown("---")
if st.button("Show Feature Importance (SHAP)"):
    try:
        feature_names = getattr(pre["scaler"], "feature_names_in_", [f"Feature {i+1}" for i in range(len(pre["scaler"].mean_))])
        X_sample = pd.DataFrame(np.random.rand(50, len(feature_names)), columns=feature_names)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        plt.title("Feature Importance (SHAP Summary)")
        shap.summary_plot(
            shap_values[1] if isinstance(shap_values, list) else shap_values,
            X_sample,
            show=False
        )
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error displaying SHAP plot: {e}")
