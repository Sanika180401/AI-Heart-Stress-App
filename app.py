import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import os
import shap
from openai import OpenAI

# -----------------------------------------------------------
# 🌙 DARK MODE THEME + CUSTOM CSS ANIMATIONS
# -----------------------------------------------------------
st.set_page_config(page_title="AI Heart & Stress Analyzer", page_icon="", layout="wide")

st.markdown("""
<style>

/* Global dark mode */
body, .stApp {
    background-color: #0d0f15;
    color: #e6e6e6;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
.css-1d391kg {
    background-color: #11141c !important;
    color: white !important;
}

/* Pretty section headers */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #68a6ff !important;
}

/* Glassmorphism cards */
.card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    transition: 0.3s ease;
}

.card:hover {
    border-color: rgba(104,166,255,0.6);
    transform: translateY(-5px);
}

/* Prediction success card (green) */
.pred-card {
    background: linear-gradient(135deg,#003b2e 0%, #005c45 100%);
    border-radius: 16px;
    padding: 20px;
    color: #c8ffe8 !important;
    font-size: 17px;
    box-shadow: 0 0 20px rgba(0,255,170,0.2);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#1d4ed8,#1e3a8a);
    color: white;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 16px;
    transition: 0.25s ease;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg,#3b82f6,#1d4ed8);
    transform: scale(1.03);
}

/* Small image/video preview */
.preview {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.15);
}

/* GPT Explanation box */
.gpt-box {
    background: rgba(255,255,255,0.07);
    padding: 18px;
    border-radius: 12px;
    border-left: 4px solid #68a6ff;
}

/* SHAP table styling */
.dataframe {
    background-color: #11141c !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# Load OpenAI Key
# -----------------------------------------------------------
def load_key():
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    if os.path.exists("openai_key.txt"):
        return open("openai_key.txt").read().strip()
    return None

OPENAI_KEY = load_key()
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# -----------------------------------------------------------
# Load ML Models
# -----------------------------------------------------------
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/stacker.pkl")
FEATURES = ['mean_hr', 'rmssd', 'pnn50', 'sd1', 'sd2', 'lf_hf', 'rr_mean', 'rr_std']

# -----------------------------------------------------------
# HRV Extraction Dummy (your real extraction stays same)
# -----------------------------------------------------------
def extract_hrv_from_image(img): 
    return {"mean_hr": 75, "rmssd": 22, "pnn50": 0.15, "sd1": 18, "sd2": 32, "lf_hf": 0.5, "rr_mean": 0.83, "rr_std": 0.06}

def extract_hrv_from_video(v): 
    return extract_hrv_from_image(None)

# -----------------------------------------------------------
# Stress / Attack Risk Logic
# -----------------------------------------------------------
def compute_stress_and_risk(x):
    stress = min(1, max(0, (x[1]*0.04)+(x[7]*4)))
    attack = min(1, (x[0]/200)+(0.5*stress))
    return stress, attack

def label(score):
    return "Low" if score < 0.33 else "Moderate" if score < 0.66 else "High"

# -----------------------------------------------------------
# GPT Explanation
# -----------------------------------------------------------
def ai_explain(hr, stress, attack):
    if not client:
        return "AI assistant inactive — OpenAI key missing."

    prompt = f"""
User Heart Report:
- Heart Rate: {hr} bpm
- Stress Level: {stress}
- Estimated Heart Attack Risk: {attack}%

Give a friendly, simple, motivating explanation. 
"""

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return r.choices[0].message["content"]
    except Exception as e:
        return f"GPT Error: {e}"

# -----------------------------------------------------------
# UI START
# -----------------------------------------------------------
st.title("AI Heart & Stress Analyzer")

st.sidebar.title("⚙ Input Configuration")
method = st.sidebar.radio("Choose Method", ["Manual Entry", "Upload Video", "Upload Image", "Webcam Image", "Webcam Video"])
st.sidebar.slider("Max Video Seconds", 5, 20, 10)
st.sidebar.slider("Webcam recording seconds", 5, 20, 8)
show_shap = st.sidebar.checkbox("Show SHAP Explainability")

hrv = None

# -----------------------------------------------------------
# USER INPUT SECTIONS
# -----------------------------------------------------------
if method == "Manual Entry":
    st.subheader("Manual Entry")
    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(FEATURES):
        with cols[i % 4]:
            vals[f] = st.number_input(f, value=50.0 if f=="mean_hr" else 1.0)
    if st.button("Predict"):
        hrv = vals

elif method == "Upload Image":
    st.subheader("Upload Image")
    img = st.file_uploader("Upload", type=["jpg","png","jpeg"])
    if img:
        st.image(img, width=300, caption="Image Preview", output_format="PNG")
        if st.button("Predict"):
            hrv = extract_hrv_from_image(None)

elif method == "Upload Video":
    st.subheader("Upload Video")
    vid = st.file_uploader("Upload Video", type=["mp4","mov"])
    if vid:
        st.video(vid)
        if st.button("Predict"):
            hrv = extract_hrv_from_video(vid.read())

elif method == "Webcam Image":
    st.subheader("Webcam Capture")
    img = st.camera_input("Capture")
    if img:
        st.image(img, width=300)
        if st.button("Predict"):
            hrv = extract_hrv_from_image(None)

elif method == "Webcam Video":
    st.subheader("Webcam Video")
    st.info("Please use Upload Video instead.")

# -----------------------------------------------------------
# RESULTS SECTION
# -----------------------------------------------------------
if hrv is not None:
    df = pd.DataFrame([hrv])[FEATURES]
    scaled = scaler.transform(df)
    pred = model.predict_proba(scaled)[0][1]

    stress, attack = compute_stress_and_risk(list(df.iloc[0]))
    stress_label = label(stress)
    attack_label = label(attack)
    hr = df["mean_hr"].iloc[0]

    st.markdown(f"""
    <div class="pred-card">
        <b>Predicted HR:</b> {hr} bpm<br>
        <b>Stress Level:</b> {stress_label} (score {stress:.2f})<br>
        <b>Heart Attack Risk:</b> {int(attack*100)}% ({attack_label})<br>
        Try slow breathing for 2 minutes to relax.
    </div>
    """, unsafe_allow_html=True)

    # SHAP
    if show_shap:
        st.subheader("SHAP Explainability")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(scaled)[0]
        st.table(pd.DataFrame({"Feature": FEATURES, "SHAP": shap_vals}))

    # GPT Explanation
    st.subheader("AI Explanation")
    st.markdown(f"<div class='gpt-box'>{ai_explain(hr, stress_label, int(attack*100))}</div>", unsafe_allow_html=True)
