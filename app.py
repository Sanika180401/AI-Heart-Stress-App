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
import cv2
import streamlit as st
import numpy as np
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# -----------------------------
#  Webcam Handler
# -----------------------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img


def capture_emotion():
    st.info("Click START above, look at the camera for 10 seconds, then click CAPTURE.")

    ctx = webrtc_streamer(
        key="emotion",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_processor:
        if st.button("CAPTURE & ANALYZE"):
            st.info("Capturing frames... please wait 3 seconds")

            frames = []
            import time
            t_end = time.time() + 3

            # Collect frames for 3 seconds
            while time.time() < t_end:
                if ctx.video_processor.frame is not None:
                    frames.append(ctx.video_processor.frame.copy())

            st.success(f"Captured {len(frames)} frames")

            # Perform emotion analysis on the middle frame
            if len(frames) > 5:
                frame = frames[len(frames) // 2]
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                emotion = result['dominant_emotion']
                st.success(f"Detected Emotion: **{emotion}**")

                # convert emotion to stress score
                stress_map = {
                    "happy": 10,
                    "neutral": 20,
                    "surprise": 30,
                    "sad": 70,
                    "fear": 80,
                    "angry": 90,
                    "disgust": 85
                }

                stress = stress_map.get(emotion, 50)

                st.info(f"Estimated Stress Level: **{stress}/100**")

                return emotion, stress

    return None, None

emotion, stress = capture_emotion()
if emotion:
    st.write("Emotion:", emotion)
    st.write("Stress Score:", stress)

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
