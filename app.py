# app.py — Clean version (Manual + Upload Image + Camera Capture)

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- Streamlit Config ---------------------------
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")

# --------------------------- ML/DL Imports ------------------------------
from tensorflow.keras.models import load_model
from sklearn.exceptions import NotFittedError
import shap

# --------------------------- MediaPipe ---------------------------------
import cv2
import mediapipe as mp

# --------------------------- Paths -------------------------------------
BASE_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_PATH, "models")

def safe_path(fname):
    p = os.path.join(MODELS_PATH, fname)
    if not os.path.exists(p):
        st.warning(f"Missing model file: {fname}")
        return None
    return p

# ---------------------- Load all models safely --------------------------
@st.cache_resource
def load_all_models():
    pre = None
    rf = None
    xgb = None
    mlp = None
    thr = {"low": 0.4, "high": 0.7}

    # Preprocessor
    p = safe_path("preprocess.joblib")
    if p:
        try: pre = joblib.load(p)
        except: st.warning("Could not load preprocess.joblib")

    p = safe_path("rf_hrv.joblib")
    if p:
        try: rf = joblib.load(p)
        except: st.warning("Could not load rf_hrv.joblib")

    p = safe_path("xgb_hrv.joblib")
    if p:
        try: xgb = joblib.load(p)
        except: st.warning("Could not load xgb_hrv.joblib")

    # MLP
    for name in ["mlp_hrv.h5", "mlp_hrv_clean.h5", "mlp_hrv.weights.h5"]:
        p = safe_path(name)
        if p:
            try:
                mlp = load_model(p, compile=False)
                break
            except:
                continue

    p = safe_path("adaptive_thresholds.npy")
    if p:
        try: thr = np.load(p, allow_pickle=True).item()
        except: pass

    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all_models()

# ---------------------- Face Feature Extraction -------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def _euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def extract_face_features_from_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w = img_bgr.shape[:2]

    def xy(i):
        return (int(lm[i].x * w), int(lm[i].y * h))

    try:
        mouth_ar = _euclid(xy(13), xy(14)) / (_euclid(xy(61), xy(291)) + 1e-6)
    except: mouth_ar = 0.0

    try:
        r_vert = _euclid(xy(159), xy(145))
        r_horiz = _euclid(xy(33), xy(133))
        l_vert = _euclid(xy(386), xy(374))
        l_horiz = _euclid(xy(362), xy(263))
        eye_ar = (r_vert/r_horiz + l_vert/l_horiz) / 2
    except: eye_ar = 0.0

    try:
        brow_gap = _euclid(xy(10), xy(159)) / (h + 1e-6)
    except: brow_gap = 0.0

    return float(mouth_ar), float(eye_ar), float(brow_gap)

# ---------------------- Input Builder -----------------------------------
def build_input_row(hrv, face, scaler):
    base_feats = ['mean_hr','sdnn','rmssd','pnn50','lf','hf','sd1','sd2','temp','eda']
    face_feats = ['mouth_ar_mean', 'eye_ar_mean', 'brow_gap_mean']

    row = [float(hrv[k]) for k in base_feats]

    if face:
        for k in face_feats: row.append(float(face[k]))
    else:
        row.extend([0,0,0])

    row = np.array([row])

    if scaler:
        expected = scaler.n_features_in_
        current = row.shape[1]
        if current < expected:
            row = np.concatenate([row, np.zeros((1, expected-current))], axis=1)
        elif current > expected:
            row = row[:, :expected]

    return row

def ensemble_predict(row):
    if pre is None or 'scaler' not in pre:
        raise RuntimeError("Scaler missing")

    row_s = pre['scaler'].transform(row)
    probs = []
    detail = {}

    if rf:
        p = rf.predict_proba(row_s)[:,1][0]
        probs.append(p)
        detail["rf"] = float(p)

    if xgb:
        try: 
            p = xgb.predict_proba(row_s)[:,1][0]
            probs.append(p)
            detail["xgb"] = float(p)
        except:
            detail["xgb"] = None

    if mlp:
        try:
            p = float(mlp.predict(row_s)[0])
            probs.append(p)
            detail["mlp"] = float(p)
        except:
            detail["mlp"] = None

    if len(probs)==0:
        raise RuntimeError("No working models found.")

    avg = float(np.mean(probs))

    if avg < thr["low"]:
        level = "Low Stress"
    elif avg < thr["high"]:
        level = "Moderate Stress"
    else:
        level = "High Stress"

    return avg, level, detail

# ---------------------- GPT Support -------------------------------------
def gpt_explain_stub(level, prob):
    return f"Predicted {level} with probability {prob*100:.2f}%. (GPT Disabled)"

openai_available = False
try:
    import openai
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        openai_available = True
except:
    openai_available = False

def gpt_explain(level, prob):
    if not openai_available:
        return gpt_explain_stub(level, prob)

    prompt = f"""
    Stress probability: {prob*100:.2f}%.
    Level: {level}.
    Provide a medically-safe explanation and 3 lifestyle tips.
    """

    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=200
        )
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT Error: {e})"

# ---------------------- UI ----------------------------------------------
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")

mode = st.selectbox("Choose Mode", ["Manual Entry", "Upload Image (face)", "Camera Capture (face)"])

# ---------------------- HRV Inputs --------------------------------------
st.subheader("HRV / Physiological Inputs")

c1, c2 = st.columns(2)

with c1:
    mean_hr = st.number_input("Mean Heart Rate", 30, 200, 75)
    sdnn = st.number_input("SDNN", 1, 500, 50)
    rmssd = st.number_input("RMSSD", 1, 300, 30)
    pnn50 = st.number_input("pNN50 (%)", 0.0, 100.0, 20.0)
    lf = st.number_input("LF Power", 0.0, 5000.0, 800.0)

with c2:
    hf = st.number_input("HF Power", 0.0, 5000.0, 600.0)
    sd1 = st.number_input("SD1", 1.0, 200.0, 20.0)
    sd2 = st.number_input("SD2", 1.0, 300.0, 40.0)
    temp = st.number_input("Skin Temp (°C)", 30.0, 40.0, 36.5)
    eda = st.number_input("EDA (µS)", 0.0, 20.0, 2.0)

hrv = {
    "mean_hr": mean_hr, "sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50,
    "lf": lf, "hf": hf, "sd1": sd1, "sd2": sd2, "temp": temp, "eda": eda
}

# ---------------------- MODE 1 — Manual ---------------------------------
if mode == "Manual Entry":
    if st.button("Predict (Manual)"):
        row = build_input_row(hrv, None, pre['scaler'] if pre else None)
        try:
            prob, level, detail = ensemble_predict(row)
            st.metric("Stress Probability", f"{prob*100:.2f}%")
            st.write("Category:", level)
            st.write("Model probabilities:", detail)
            st.info(gpt_explain(level, prob))
        except Exception as e:
            st.error(str(e))

# ---------------------- MODE 2 — Upload Image ----------------------------
elif mode == "Upload Image (face)":
    img = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

    if img:
        data = img.read()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        feats = extract_face_features_from_bgr(bgr)
        if feats:
            face = {
                "mouth_ar_mean": feats[0],
                "eye_ar_mean": feats[1],
                "brow_gap_mean": feats[2]
            }
            st.write("Face features:", face)

            row = build_input_row(hrv, face, pre['scaler'] if pre else None)

            try:
                prob, level, detail = ensemble_predict(row)
                st.metric("Stress Probability", f"{prob*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probabilities:", detail)
                st.info(gpt_explain(level, prob))
            except Exception as e:
                st.error(str(e))
        else:
            st.error("No face detected!")

# ---------------------- MODE 3 — Camera Capture -------------------------
elif mode == "Camera Capture (face)":
    cam = st.camera_input("Capture a face photo")

    if cam:
        data = cam.getvalue()
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        feats = extract_face_features_from_bgr(bgr)
        if feats:
            face = {
                "mouth_ar_mean": feats[0],
                "eye_ar_mean": feats[1],
                "brow_gap_mean": feats[2]
            }

            row = build_input_row(hrv, face, pre['scaler'] if pre else None)

            try:
                prob, level, detail = ensemble_predict(row)
                st.metric("Stress Probability", f"{prob*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probabilities:", detail)
                st.info(gpt_explain(level, prob))
            except Exception as e:
                st.error(str(e))
        else:
            st.error("No face detected!")

st.markdown("---")
st.caption("Place model files inside the models/ folder. Add OpenAI key in .streamlit/secrets.toml to enable GPT explanations.")
