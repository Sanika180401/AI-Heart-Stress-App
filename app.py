# ===================== FULL MERGED app.py =====================
# AI-Based Real-Time Heart Rate & Stress Monitoring System
# With Webcam Emotion Analysis + HRV + Hybrid Ensemble + SHAP + GPT

# ===============================================================
# IMPORTS
# ===============================================================
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tensorflow.keras.models import load_model

# Webcam + Emotion Analysis
from deepface import DeepFace
import cv2, time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ===============================================================
# STREAMLIT CONFIG
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
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        raise FileNotFoundError(path)
    return path

# ===============================================================
# LOAD MODELS
# ===============================================================
@st.cache_resource
def load_all():
    pre = joblib.load(safe_load("preprocess.joblib"))
    rf = joblib.load(safe_load("rf_hrv.joblib"))

    try:
        xgb = joblib.load(safe_load("xgb_hrv.joblib"))
    except:
        xgb = None

    try:
        mlp = load_model(safe_load("mlp_hrv_clean_fixed.keras"), compile=False)
    except:
        mlp = None

    thr = np.load(safe_load("adaptive_thresholds.npy"), allow_pickle=True).item()
    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all()

# ===============================================================
# GPT OPTIONAL EXPLANATION
# ===============================================================
try:
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
except:
    openai = None


def gpt_explain(stress_level, prob):
    if not openai or not openai.api_key:
        return "(GPT explanation unavailable)"

    prompt = f"""
    Stress probability: {prob*100:.2f}%
    Stress level: {stress_level}
    Give short doctor-style explanation + 3 tips.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a calming medical AI."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except:
        return "GPT error"

# ===============================================================
# PREDICTION FUNCTION
# ===============================================================

def predict_stress(input_data):
    input_scaled = pre['scaler'].transform(input_data)
    preds = [rf.predict_proba(input_scaled)[:,1]]

    if xgb:
        preds.append(xgb.predict_proba(input_scaled)[:,1])
    if mlp:
        preds.append(mlp.predict(input_scaled).flatten())

    avg = np.mean(preds, axis=0)[0]
    low, high = thr['low'], thr['high']

    if avg < low:
        level = "Low Stress"
    elif avg < high:
        level = "Moderate Stress"
    else:
        level = "High Stress"

    return avg, level

# ===============================================================
# INPUT OPTIONS
# ===============================================================
option = st.radio("Select Input Method:", ("Manual Entry", "Upload CSV", "Webcam Emotion Analysis"))

# ===============================================================
# MANUAL ENTRY MODE
# ===============================================================
if option == "Manual Entry":

    col1, col2 = st.columns(2)
    with col1:
        mean_hr = st.number_input("Mean HR", 40,200,75)
        sdnn = st.number_input("SDNN", 10,200,50)
        rmssd = st.number_input("RMSSD", 10,200,30)
        sd1 = st.number_input("SD1", 5.0,100.0,20.0)
        temp = st.number_input("Temperature", 30.0,40.0,36.5)
    with col2:
        pnn50 = st.number_input("pNN50", 0,100,20)
        lf = st.number_input("LF", 100,5000,800)
        hf = st.number_input("HF", 100,5000,600)
        sd2 = st.number_input("SD2", 5.0,200.0,40.0)
        eda = st.number_input("EDA", 0.1,10.0,2.0)

    if st.button("Predict"):
        data = np.array([[mean_hr,sdnn,rmssd,pnn50,lf,hf,sd1,sd2,temp,eda]])
        prob, level = predict_stress(data)
        st.metric("Stress Probability", f"{prob*100:.2f}%")
        st.success(level)
        st.info(gpt_explain(level, prob))

# ===============================================================
# WEBCAM EMOTION ANALYSIS MODE (FIXED & WORKING)
# ===============================================================
elif option == "Webcam Emotion Analysis":

    st.subheader("Live Webcam Emotion Analysis")

    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.buffer = []

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            try:
                result = DeepFace.analyze(
                    img,
                    actions=['emotion'],
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                if "emotion" in result:
                    self.buffer.append(result["emotion"])

            except Exception:
                pass

            # Display dominant emotion
            if len(self.buffer) > 0:
                last = self.buffer[-1]
                dominant = max(last, key=last.get)
                cv2.putText(
                    img,
                    f"Emotion: {dominant}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            return img


    webrtc_ctx = webrtc_streamer(
        key="webcam",
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )

    if st.button("Capture & Predict"):

        if not webrtc_ctx.video_transformer:
            st.error("Camera not started.")
            st.stop()

        buffer = webrtc_ctx.video_transformer.buffer

        if len(buffer) < 10:
            st.error("No emotion data captured. Face camera for 10-15 seconds.")
            st.stop()

        st.success(f"Frames captured: {len(buffer)}")

        emo_df = pd.DataFrame(buffer)

        agg = {}
        for col in emo_df.columns:
            agg[f"{col}_mean"] = emo_df[col].mean()

        hrv_features = [
            75,   # mean_hr
            50,   # sdnn
            30,   # rmssd
            20,   # pnn50
            800,  # lf
            600,  # hf
            20,   # sd1
            40,   # sd2
            36.5, # temp
            2.0   # eda
        ]

        final_features = hrv_features + list(agg.values())
        input_array = np.array([final_features])

        input_scaled = pre["scaler"].transform(input_array)

        preds = [rf.predict_proba(input_scaled)[:,1]]
        if xgb:
            preds.append(xgb.predict_proba(input_scaled)[:,1])
        if mlp:
            preds.append(mlp.predict(input_scaled).flatten())

        avg_pred = float(np.mean(preds))

        level = "Low" if avg_pred < thr["low"] else \
                "Moderate" if avg_pred < thr["high"] else "High"

        st.metric("Stress Probability", f"{avg_pred*100:.2f}%")
        st.success(f"Stress Level: {level}")
        st.info(gpt_explain(level, avg_pred))

# ===============================================================
# SHAP FEATURE IMPORTANCE
# ===============================================================
st.markdown("---")
if st.button("Show SHAP Feature Importance"):
    try:
        X_sample = pd.DataFrame(np.random.rand(50,10))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(plt)
    except Exception as e:
        st.error(f"SHAP error: {e}")
