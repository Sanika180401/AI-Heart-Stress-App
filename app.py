# final app.py — AI Heart Rate & Stress Analyzer (robust, premium UI + panels)
# Replace your existing app.py with this file.
# Requirements (some optional): streamlit, opencv-python, numpy, pandas, joblib, scipy, plotly, shap (optional), fpdf (optional), matplotlib (optional)

import os
import tempfile
import time
import math
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
from joblib import load
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend

# Optional libs
try:
    import shap
except Exception:
    shap = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

# ---------------------------
# Page config + CSS (light premium)
# ---------------------------
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer", layout="wide")

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
LOTTIE_FILE = ASSETS_DIR / "breathing.json"

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

st.markdown("""
<style>
/* App background */
.stApp { background: #f7f9fb; color: #0b1721; }

/* Header/logo */
.logo-box {
    width:64px; height:64px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#ff4d6d,#ff7a59);
    color:white; font-weight:700; font-size:28px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}

/* Card */
.card { background:white; padding:18px; border-radius:12px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.04);
        border:1px solid rgba(15,23,42,0.02); margin-bottom:14px; }

/* Colorful primary buttons */
.stButton > button {
    background: linear-gradient(90deg,#ff4d6d,#ff7a59);
    color: white; font-weight:700; padding:10px 18px; border-radius:10px;
    border: none;
}
.stButton > button:hover { transform: translateY(-3px); }

/* secondary */
.stButton.secondary > button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white;
}

/* compact preview */
.preview-img { width:360px; height:200px; object-fit:cover; border-radius:10px; border:1px solid rgba(2,6,23,0.04); }

/* stress bar */
.progress-wrap { background:#eef2f7; border-radius:12px; padding:8px; }
.progress-inner { height:26px; border-radius:8px; width:0%; background:linear-gradient(90deg,#ff7a59,#ff4d6d); color:white; display:flex; align-items:center; justify-content:center; font-weight:700; transition: width 1s ease; }

/* butterfly */
.butterfly { width:120px; height:120px; margin:auto; display:block; }
@keyframes flutter { 0% { transform: translateY(0) rotate(-2deg);} 50% { transform: translateY(-6px) rotate(2deg);} 100% { transform: translateY(0) rotate(-2deg);} }
.butterfly svg { animation: flutter 2s infinite ease-in-out; transform-origin:center; }

/* muted */
.muted { color:#64748b; font-size:13px; }

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Safe model loader
# ---------------------------
def safe_load_joblib(path):
    try:
        if not Path(path).exists():
            return None
        return load(path)
    except Exception:
        return None

scaler = safe_load_joblib(MODELS_DIR / "scaler.pkl")
mlp = safe_load_joblib(MODELS_DIR / "mlp.pkl")
rf = safe_load_joblib(MODELS_DIR / "rf.pkl")
xgb = safe_load_joblib(MODELS_DIR / "xgb.pkl")
stacker = safe_load_joblib(MODELS_DIR / "stacker.pkl")

models_loaded = {
    "scaler": scaler is not None,
    "mlp": mlp is not None,
    "rf": rf is not None,
    "xgb": xgb is not None,
    "stacker": stacker is not None
}

# ---------------------------
# rPPG helpers
# ---------------------------
def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_mean_green_signal_from_video_file(video_path, max_seconds=12, resize_width=360):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds)
    max_frames = min(total, int(fps*max_seconds))
    sig = []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h,w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        cx, cy = w//2, h//2
        wbox, hbox = int(w*0.35), int(h*0.45)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(w, cx+wbox//2), min(h, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1
            continue
        sig.append(float(np.mean(roi[:,:,1])))
        frames += 1
    cap.release()
    if len(sig) < 8:
        return None, fps
    return np.array(sig), fps

def get_hr_from_signal(sig, fs):
    try:
        filt = bandpass(sig - np.mean(sig), fs)
    except Exception:
        filt = sig - np.mean(sig)
    distance = max(1, int(0.4 * fs))
    peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.18)
    if len(peaks) < 2:
        peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.08)
    if len(peaks) < 2:
        return None, None
    times = peaks / float(fs)
    rr = np.diff(times) * 1000.0
    hr_series = 60000.0 / rr
    return float(np.mean(hr_series)), hr_series

def compute_hrv(rr_ms, hr_ser=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff)>0 else 0.0
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0) if len(diff)>0 else 0.0
    sd1 = float(np.sqrt(np.var(diff) / 2.0)) if len(diff)>0 else 0.0
    sd2 = float(np.sqrt(max(0.0, 2*np.var(rr_ms) - np.var(diff)/2.0))) if len(diff)>0 else 0.0
    rr_mean = float(np.mean(rr_ms))
    rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_ser)) if hr_ser is not None else float(60000.0/np.mean(rr_ms))
    lf_hf = 0.0
    try:
        fs_interp = 4.0
        times = np.cumsum(rr_ms)/1000.0
        if len(times) >= 4:
            t_interp = np.arange(0, times[-1], 1.0/fs_interp)
            inst_hr = 60000.0/rr_ms
            beat_times = times[:-1]
            interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
            f, p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
            lf_mask = (f>=0.04) & (f<=0.15)
            hf_mask = (f>0.15) & (f<=0.4)
            lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
            hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
            lf_hf = float(lf/hf) if hf>0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {"mean_hr":mean_hr,"rmssd":rmssd,"pnn50":pnn50,"sd1":sd1,"sd2":sd2,"lf_hf":lf_hf,"rr_mean":rr_mean,"rr_std":rr_std}

# ---------------------------
# Features vectorize & ensemble prediction
# ---------------------------
def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

def safe_prob(model, Xs):
    try:
        return float(model.predict_proba(Xs)[:,1][0])
    except Exception:
        try:
            return float(model.predict(Xs)[0])
        except Exception:
            return 0.5

def predict_ensemble(feats):
    X = vectorize_features(feats)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    probs = []
    if mlp is not None:
        probs.append(safe_prob(mlp, Xs))
    if rf is not None:
        probs.append(safe_prob(rf, Xs))
    if xgb is not None:
        probs.append(safe_prob(xgb, Xs))
    if len(probs) == 0:
        # heuristic fallback
        hr = feats.get("mean_hr", 70)
        rmssd = feats.get("rmssd", 30) or 30
        prob = min(0.99, max(0.01, 0.4*(hr/100.0) + 0.6*(30.0/(rmssd+1.0))))
        return float(prob), []
    meta = np.array(probs).reshape(1, -1)
    try:
        final = float(stacker.predict_proba(meta)[:,1][0]) if stacker is not None else float(np.mean(probs))
    except Exception:
        final = float(np.mean(probs))
    return final, probs

def categorize_stress(p):
    if p < 0.33: return "Low"
    if p < 0.66: return "Moderate"
    return "High"

def categorize_hr(hr):
    if hr < 60: return "Low"
    if hr <= 100: return "Normal"
    return "High"

def heart_attack_risk_heuristic(prob, mean_hr):
    hr_score = max(0.0, (mean_hr - 60.0) / 60.0)
    risk_raw = 0.6 * prob + 0.4 * min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20: cat="Low"
    elif risk_pct < 50: cat="Moderate"
    else: cat="High"
    return risk_pct, cat

# ---------------------------
# SHAP robust helper (for MLP and tree models)
# ---------------------------
def compute_shap_table(feats):
    if shap is None:
        return None, "SHAP not installed."
    # prepare vector
    X = vectorize_features(feats)
    try:
        # prefer tree models (rf/xgb) for faster explainers
        if rf is not None:
            model_for_shap = rf
            explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(np.zeros((1, len(FEATURE_ORDER)))))
            svals = explainer(X)
            vals = np.array(svals.values).reshape(-1)[:len(FEATURE_ORDER)]
        elif xgb is not None:
            model_for_shap = xgb
            explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(np.zeros((1, len(FEATURE_ORDER)))))
            svals = explainer(X)
            vals = np.array(svals.values).reshape(-1)[:len(FEATURE_ORDER)]
        elif mlp is not None:
            # KernelExplainer fallback for MLP (single-output)
            try:
                # background of zeros (fast, not ideal but works)
                background = np.zeros((10, len(FEATURE_ORDER)))
                explainer = shap.KernelExplainer(lambda z: np.array(mlp.predict_proba(z)[:,1]) , background)
                svals = explainer.shap_values(X, nsamples=64)
                vals = np.array(svals).reshape(-1)[:len(FEATURE_ORDER)]
            except Exception as e:
                return None, f"SHAP failure: {e}"
        else:
            return None, "No model available for SHAP."
        if len(vals) < len(FEATURE_ORDER):
            vals = np.pad(vals, (0, len(FEATURE_ORDER)-len(vals)), 'constant', constant_values=0.0)
        df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": [float(x) for x in vals]})
        return df, None
    except Exception as e:
        return None, f"SHAP not available: {e}"

# ---------------------------
# AI explanation templates (EN/HI/MR)
# ---------------------------
TEMPLATES = {
    "English": {
        "Low": {
            "one": "Your HRV and heart rate are within a healthy range — low stress.",
            "recommend": ["Immediate: Take 1–2 calm breaths.", "Today: Short breaks and hydration.", "Long-term: regular exercise & sleep."],
            "seek": "If you have chest pain or fainting, seek medical care."
        },
        "Moderate": {
            "one": "Moderate stress indicators and slightly elevated heart rate.",
            "recommend": ["Immediate: 2 min paced breathing (4s in/4s out).", "Today: Reduce caffeine, take a short walk.", "Long-term: mindfulness, exercise."],
            "seek": "If symptoms worsen or persist, consult a clinician."
        },
        "High": {
            "one": "High stress markers and elevated heart rate.",
            "recommend": ["Immediate: 3 min guided breathing & sit down.", "Today: Avoid stimulants and rest.", "Long-term: seek stress-management support."],
            "seek": "If severe chest pain or breathlessness occur, get emergency care."
        }
    },
    "Hindi": {
        "Low": {
            "one":"आपका हृदय-क्रम और HRV सामान्य हैं — तनाव कम है।",
            "recommend":["तुरंत: 1–2 मिनट गहरी साँसें लें।","आज: ब्रेक और पानी पिएँ।","दीर्घकालिक: नियमित व्यायाम और नींद।"],
            "seek":"सीने में दर्द या चक्कर आने पर डॉक्टर से संपर्क करें।"
        },
        "Moderate": {
            "one":"तनाव के संकेत मध्यम हैं और हृदय-दर थोड़ी बढ़ी है।",
            "recommend":["तुरंत: 2 मिनट शांत साँस लें।","आज: कैफीन कम करें।","दीर्घकालिक: ध्यान और व्यायाम।"],
            "seek":"लक्षण बिगड़ें तो चिकित्सक से मिलें।"
        },
        "High": {
            "one":"उच्च तनाव चिन्ह और बढ़ा हुआ हृदय-दर।",
            "recommend":["तुरंत: 3 मिनट गहरी श्वास।","आज: आराम करें, स्टिमुलेंट टाळें।","दीर्घकालिक: विशेषज्ञ से सलाह लें।"],
            "seek":"सीने में दर्द या सांस लेने में कठिनाई हो तो तुरंत अस्पताल जाएँ।"
        }
    },
    "Marathi": {
        "Low": {
            "one":"तुमचा हृदयाचा वेग व HRV सामान्य आहेत — ताण कमी आहे.",
            "recommend":["तुरंत: 1–2 मिनिटे श्वास घ्या.","आज: ब्रेक आणि पाणी प्या.","दीर्घकालीन: नियमित व्यायाम व झोप."],
            "seek":"छातीत वेदना किंवा चक्कर आल्यास डॉक्टरांकडे जा."
        },
        "Moderate": {
            "one":"तणावाच्या चिन्हे मध्यम आहेत आणि हृदयाचा वेग थोडा वाढलेला आहे.",
            "recommend":["तुरंत: 2 मिनिटे शांत श्वास.","आज: कैफीन कमी करा.","दीर्घकालीन: ध्यान व व्यायाम करा."],
            "seek":"लक्षणे वाढल्यास डॉक्टरांचा सल्ला घ्या."
        },
        "High": {
            "one":"उच्च ताण आणि वाढलेला हृदयाचा वेग आहेत.",
            "recommend":["तुरंत: 3 मिनिटे श्वास व्यायाम.","आज: विश्रांती घ्या.","दीर्घकालीन: तज्ञांचा सल्ला घ्या."],
            "seek":"छातीत वेदना किंवा श्वास घेण्यात त्रास झाला तर ताबडतोब वैद्यकीय मदत घ्या."
        }
    }
}

def generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang="English"):
    label = "Low" if prob<0.33 else "Moderate" if prob<0.66 else "High"
    tpl = TEMPLATES.get(lang, TEMPLATES["English"]).get(label, TEMPLATES["English"]["Moderate"])
    lines = []
    lines.append(tpl["one"])
    lines.append("")
    lines.append("Recommendations:")
    for r in tpl["recommend"]:
        lines.append("- " + r)
    lines.append("")
    lines.append("When to seek care: " + tpl["seek"])
    return "\n".join(lines)

# ---------------------------
# UI layout
# ---------------------------

# Ensure waveform and features exist globally (avoid reference errors)
waveform = None
features = None

col_logo, col_title = st.columns([0.8, 6])
with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=72)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("<h1>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>', unsafe_allow_html=True)

st.write("")

# Sidebar settings
with st.sidebar:
    st.header("Input & Settings")
    method = st.radio("Choose method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max video seconds", 6, 20, 10)
    rec_seconds = st.slider("Webcam recording sec", 4, 12, 8)
    show_shap = st.checkbox("Show SHAP explainability", True)
    lang = st.selectbox("Language", ["English","Hindi","Marathi"])
    st.markdown("---")

# main input + preview
left, right = st.columns([1, 1.4])
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input")
    # local features/waveform will (re)assign global 'features' and 'waveform'
    if method == "Manual Entry":
        vals = {}
        cols = st.columns(2)
        for i,f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f=='mean_hr' else 1.0
                vals[f] = st.number_input(f, value=float(default))
        if st.button("Predict"):
            features = {k: float(v) for k,v in vals.items()}

    elif method == "Upload Image":
        uploaded = st.file_uploader("Image", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, width=360, caption="Preview")
            if st.button("Predict"):
                img = Image.open(uploaded).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}

    elif method == "Upload Video":
        vid = st.file_uploader("Video (mp4,mov,avi)", type=["mp4","mov","avi"])
        if vid:
            st.video(vid, start_time=0)
            if st.button("Predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vid.read()); tmp.flush(); tmp.close()
                try:
                    sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                    if sig is None:
                        st.error("Couldn't extract reliable signal — try a longer/clearer video.")
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error("Pulse peaks not reliable — try again.")
                        else:
                            rr_ms = (60000.0/np.array(hr_series))
                            feats = compute_hrv(rr_ms, hr_series)
                            features = feats
                            waveform = sig.tolist()
                except Exception as e:
                    st.error(f"Video processing failed: {e}")
                finally:
                    try: os.unlink(tmp.name)
                    except: pass

    elif method == "Webcam Image":
        cam = st.camera_input("Capture image")
        if cam:
            st.image(cam, width=360)
            if st.button("Predict"):
                img = Image.open(cam).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}

    elif method == "Webcam Video":
        st.write("Click to record webcam video (local only).")
        if st.button("Start Webcam Recording"):
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmpf.close()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Run locally and allow camera.")
            else:
                fps = 20.0
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read webcam.")
                else:
                    h,w = frame.shape[:2]
                    out = cv2.VideoWriter(tmpf.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
                    t0 = time.time()
                    progress = st.progress(0)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                        elapsed = time.time() - t0
                        progress.progress(min(100, int((elapsed/rec_seconds)*100)))
                        if elapsed >= rec_seconds:
                            break
                    out.release(); cap.release()
                    st.success("Recording finished — processing...")
                    try:
                        sig, fps = extract_mean_green_signal_from_video_file(tmpf.name, max_seconds=rec_seconds)
                        if sig is None:
                            st.error("Couldn't extract reliable signal from webcam video.")
                        else:
                            mean_hr, hr_series = get_hr_from_signal(sig, fps)
                            if hr_series is None:
                                st.error("Pulse not detected.")
                            else:
                                rr_ms = (60000.0/np.array(hr_series))
                                feats = compute_hrv(rr_ms, hr_series)
                                features = feats
                                waveform = sig.tolist()
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                    finally:
                        try: os.unlink(tmpf.name)
                        except: pass

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Dashboard")
    gauge_placeholder = st.empty()
    metrics_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# After prediction: show results
# ---------------------------
if features is not None:
    # ensure numeric floats for features dict
    feats = {k: float(features.get(k, np.nan)) if features.get(k, None) is not None else float("nan") for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob)
    hr_label = categorize_hr(feats.get("mean_hr", 0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get("mean_hr", 0.0))
    sentence = f"HR ≈ {feats.get('mean_hr', np.nan):.0f} bpm ({hr_label}). Stress: {stress_label} ({prob:.2f}). Heart-attack estimate: {risk_pct:.0f}% ({risk_cat})."

    # gauge
    if go is not None:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=feats.get("mean_hr", 60),
                                    gauge={'axis':{'range':[30,180]}, 'bar':{'color':'#ff4d6d'}},
                                    title={'text': "<b>Heart Rate (bpm)</b>"}))
        fig.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="white", font_color="#0b1721")
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        gauge_placeholder.info(f"HR: {feats.get('mean_hr', np.nan):.0f} bpm")

    # results card
    st.markdown(f"""
        <div class="card" style="margin-top:12px">
          <h3 style="color:#c41b23; margin-bottom:6px">Result</h3>
          <div class="muted">{sentence}</div>
          <div style="height:10px"></div>
          <div class="progress-wrap"><div class="progress-inner" style="width:{int(prob*100)}%;">{int(prob*100)}%</div></div>
          <div style="height:12px"></div>
          <div style="display:flex; gap:10px;">
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#c41b23">{stress_label}</b><div class="muted">Stress</div></div>
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#0b1721">{int(feats.get('mean_hr',0))} bpm</b><div class="muted">Heart Rate</div></div>
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#f97316">{int(risk_pct)}%</b><div class="muted">Attack Risk</div></div>
          </div>
        </div>
    """, unsafe_allow_html=True)

    # SHAP
    if show_shap:
        st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
        st.subheader("SHAP contributions")
        sh_df, sh_err = compute_shap_table(feats)
        if sh_df is not None:
            st.table(sh_df)
        else:
            st.info(sh_err)
        st.markdown("</div>", unsafe_allow_html=True)

    # AI explanation
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("AI Explanation")
    gpt_out = generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang=lang)
    st.write(gpt_out)
    st.markdown("</div>", unsafe_allow_html=True)

    # waveform
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("HRV waveform")
    if waveform is not None:
        st.line_chart(pd.DataFrame({"signal": waveform}))
    else:
        st.info("Waveform will appear for video/webcam inputs.")
    st.markdown("</div>", unsafe_allow_html=True)

    # butterfly
    st.markdown("""
    <div class='card' style='text-align:center; margin-top:12px'>
      <div class='butterfly'>
        <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
          <g transform="translate(60,60)">
            <path d="M-2,-2 C-30,-40 -60,-20 -40,10 C-20,40 10,20 0,0" fill="#ff7a59" opacity="0.9"/>
            <path d="M2,2 C30,40 60,20 40,-10 C20,-40 -10,-20 0,0" fill="#ff4d6d" opacity="0.9"/>
            <circle r="6" fill="#fff"/>
          </g>
        </svg>
      </div>
      <div style="height:6px"></div>
      <div class='muted'>Animated indicator — calm when green, flutter when stress rises.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# EXTENSION: Weekly Trend, PDF report, Live Waveform, Doctor Mode, Premium Home
# ============================

# Optional libs for PDF & plots
try:
    from fpdf import FPDF
except Exception:
    FPDF = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

HISTORY_FILE = ROOT / "history.json"
# If history.json is accidentally a directory, use fallback file inside it
if HISTORY_FILE.exists() and HISTORY_FILE.is_dir():
    HISTORY_FILE = HISTORY_FILE / "history.json"

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []

def sanitize_for_json(obj):
    # convert numpy types & nan -> None
    if obj is None:
        return None
    if isinstance(obj, (np.floating, float)):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    try:
        return float(obj)
    except Exception:
        return None

def save_history_entry(entry):
    data = load_history()
    # sanitize nested values
    e = {}
    for k,v in entry.items():
        if isinstance(v, dict):
            e[k] = {kk: sanitize_for_json(vv) for kk,vv in v.items()}
        else:
            e[k] = sanitize_for_json(v)
    data.append(e)
    try:
        HISTORY_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        # try to create parent dir and retry
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

# If we have current features from this run, save a history record automatically
if features is not None:
    try:
        now = datetime.datetime.now().isoformat()
        hr_val = features.get("mean_hr", None)
        prob_val, _ = predict_ensemble({k: float(features.get(k, math.nan)) if features.get(k, None) is not None else math.nan for k in FEATURE_ORDER})
        entry = {
            "time": now,
            "hr": float(hr_val) if (hr_val is not None and not math.isnan(float(hr_val))) else None,
            "stress_prob": float(prob_val),
            "features": {k: sanitize_for_json(features.get(k, None)) for k in FEATURE_ORDER}
        }
        save_history_entry(entry)
    except Exception:
        pass

# ---------- Weekly Trend Dashboard (robust timestamp parsing) ----------
def parse_time_safely(t):
    """Return a datetime or None - accepts iso strings, other strings, numeric epoch, or None."""
    if t is None:
        return None
    # if already datetime
    if isinstance(t, datetime.datetime):
        return t
    # numeric epoch
    if isinstance(t, (int, float, np.integer, np.floating)):
        try:
            return datetime.datetime.fromtimestamp(float(t))
        except Exception:
            return None
    # string inputs
    if isinstance(t, str):
        # try isoformat first
        for fmt in (None, "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                if fmt is None:
                    # try fromisoformat which accepts extended ISO
                    return datetime.datetime.fromisoformat(t)
                else:
                    return datetime.datetime.strptime(t, fmt)
            except Exception:
                continue
    return None

def compute_weekly_stats():
    data = load_history()
    if not data:
        return None
    now = datetime.datetime.now()
    week_ago = now - datetime.timedelta(days=7)

    valid_recent = []
    # accept entries that have either "time" or "timestamp" or "date"
    for d in data:
        tval = None
        if isinstance(d, dict):
            if "time" in d:
                tval = d.get("time")
            elif "timestamp" in d:
                tval = d.get("timestamp")
            elif "date" in d:
                tval = d.get("date")
        # try to parse
        dt = parse_time_safely(tval)
        if dt is not None and dt >= week_ago:
            valid_recent.append(d)
    # fallback: if no 7-day records, use the last N entries
    if not valid_recent:
        # pick last 20 numeric entries (if any)
        if isinstance(data, list) and len(data) > 0:
            valid_recent = data[-20:]
        else:
            valid_recent = []

    # extract hr and stress
    hrs = []
    probs = []
    for d in valid_recent:
        try:
            if isinstance(d, dict):
                if d.get("hr") is not None:
                    hrs.append(float(d.get("hr")))
                elif d.get("heart_rate") is not None:
                    hrs.append(float(d.get("heart_rate")))
                if d.get("stress_prob") is not None:
                    probs.append(float(d.get("stress_prob")))
                elif d.get("stress") is not None:
                    probs.append(float(d.get("stress")))
        except Exception:
            continue

    if len(hrs) == 0 and len(probs) == 0:
        return None

    stats = {
        "avg_hr": float(np.mean(hrs)) if len(hrs) > 0 else None,
        "avg_stress": float(np.mean(probs)) if len(probs) > 0 else None,
        "min_hr": float(np.min(hrs)) if len(hrs) > 0 else None,
        "max_hr": float(np.max(hrs)) if len(hrs) > 0 else None,
        "count": len(valid_recent),
        "recent": valid_recent
    }
    return stats

# ---------- PDF Report Generation ----------
def generate_pdf_report(filename="report.pdf", entry=None, waveform=None, lang="English"):
    if FPDF is None:
        return None, "FPDF not installed. Install via `pip install fpdf` to enable PDF reports."
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(196,27,35)
    pdf.cell(0, 10, "AI Heart Rate & Stress Analysis Report", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0,0,0)
    if entry is None:
        pdf.cell(0, 8, "No data available to create report.", ln=True)
    else:
        t = entry.get("time", "")
        pdf.cell(0, 8, f"Time: {t}", ln=True)
        hr = entry.get("hr", None)
        sp = entry.get("stress_prob", None)
        if hr is not None:
            pdf.cell(0, 8, f"Estimated Heart Rate: {hr:.1f} bpm", ln=True)
        if sp is not None:
            pdf.cell(0, 8, f"Stress Probability: {sp:.2f}", ln=True)
            label = "Low" if sp<0.33 else "Moderate" if sp<0.66 else "High"
            pdf.cell(0, 8, f"Stress Level: {label}", ln=True)
        pdf.ln(6)
        ai_text = generate_ai_explanation(entry.get("features", {}), sp if sp is not None else 0.0, 0, "Low", lang=lang)
        pdf.multi_cell(0, 6, "AI Explanation:")
        pdf.set_font("Arial", size=10)
        for line in ai_text.split("\n"):
            pdf.multi_cell(0, 5, line)
        pdf.set_font("Arial", size=11)
        pdf.ln(6)
        if waveform is not None and plt is not None:
            try:
                fig, ax = plt.subplots(figsize=(6,2))
                ax.plot(waveform, linewidth=1)
                ax.set_title("HRV Waveform (preview)")
                plt.tight_layout()
                img_temp = ROOT / "tmp_waveform.png"
                fig.savefig(str(img_temp), dpi=150)
                plt.close(fig)
                pdf.image(str(img_temp), w=180)
                try: img_temp.unlink()
                except: pass
            except Exception:
                pass
    out_path = ROOT / filename
    try:
        pdf.output(str(out_path))
        return out_path, None
    except Exception as e:
        return None, f"PDF generation failed: {e}"

# ---------- Live HRV Waveform Animation ----------
def play_live_waveform(wave, speed=0.03):
    if wave is None:
        st.info("No waveform available for live animation.")
        return
    if len(wave) > 5000:
        wave = wave[-2000:]
    placeholder = st.empty()
    df = pd.DataFrame({"signal": []})
    chart = placeholder.line_chart(df)
    for i, v in enumerate(wave):
        if i % 4 == 0:
            chart.add_rows(pd.DataFrame({"signal": [v]}))
        time.sleep(speed)
    st.success("Waveform playback finished.")

# ---------- Doctor Mode computations ----------
def compute_doctor_metrics(feats):
    rr_mean = feats.get("rr_mean", None)
    rr_std = feats.get("rr_std", None)
    sdnn = rr_std if rr_std is not None else None
    lf_hf = feats.get("lf_hf", None)
    mean_hr = feats.get("mean_hr", 70)
    prob, _ = predict_ensemble(feats)
    heart_age = 20 + max(0, (mean_hr - 60) * 0.4) + prob * 15
    heart_age = int(min(90, max(18, heart_age)))
    return {"sdnn": sdnn, "lf_hf": lf_hf, "heart_age": heart_age}

# ---------- Frequency-domain plotting (uses RR if available, or synthesizes RR from mean/std) ----------
def plot_frequency_domain(feats):
    """
    Try to plot PSD (LF/HF) from RR series or synthesized RR.
    Returns path to PNG image if created, otherwise None.
    """
    if plt is None:
        return None

    try:
        # If user provided an explicit rr_series in features (list/array), use it
        rr_series = None
        if isinstance(feats, dict):
            # look for raw rr arrays in keys
            for k in ("rr_series", "rr_ms", "rr_list", "rr_values"):
                if k in feats and feats[k] is not None:
                    arr = feats[k]
                    try:
                        rr_series = np.array(arr, dtype=float)
                        break
                    except Exception:
                        rr_series = None
            # otherwise check for rr_mean & rr_std and synthesize
            if rr_series is None and feats.get("rr_mean") is not None and feats.get("rr_std") is not None:
                try:
                    mu = float(feats.get("rr_mean"))
                    sigma = float(feats.get("rr_std"))
                    if math.isnan(mu) or mu <= 0:
                        rr_series = None
                    else:
                        # synthesize ~200 beats around mean with clamp
                        n = 256
                        rng = np.random.default_rng(seed=42)
                        rr_series = rng.normal(loc=mu, scale=max(1.0, sigma if not math.isnan(sigma) else 5.0), size=n)
                        rr_series = np.clip(rr_series, 300.0, 2000.0)  # ms bounds
                except Exception:
                    rr_series = None

        if rr_series is None or len(rr_series) < 4:
            return None

        # convert rr_ms to instantaneous heart rate series (bpm)
        inst_hr = 60000.0 / rr_series
        # resample/interpolate to uniform time grid if needed
        fs_interp = 4.0
        try:
            times = np.cumsum(rr_series) / 1000.0
            t_interp = np.arange(0, times[-1], 1.0/fs_interp)
            # build beat_times and inst_hr_samples
            beat_times = times[:-1] if len(times) > 1 else times
            inst_hr_samples = inst_hr[:len(beat_times)]
            if len(beat_times) >= 2:
                interp = np.interp(t_interp, beat_times, inst_hr_samples)
            else:
                interp = inst_hr
        except Exception:
            interp = inst_hr

        # PSD
        f, pxx = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
        # plot
        fig, ax = plt.subplots(figsize=(6,3))
        ax.semilogy(f, pxx, linewidth=1.2)
        ax.set_xlim(0, 0.6)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.set_title("Frequency domain (PSD) — LF/HF view")
        # mark LF and HF bands
        ax.axvspan(0.04, 0.15, alpha=0.12, color='orange', label='LF (0.04-0.15)')
        ax.axvspan(0.15, 0.4, alpha=0.08, color='green', label='HF (0.15-0.40)')
        ax.legend(loc='upper right')
        plt.tight_layout()

        out_img = ROOT / "tmp_freq.png"
        fig.savefig(str(out_img), dpi=150)
        plt.close(fig)
        return out_img
    except Exception:
        return None

# ---------- Premium Home Screen ----------
def premium_home_view(latest_entry=None, waveform=None, lang="English"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#c41b23; margin-bottom:6px;'>Premium Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
      <div style="font-size:120px; line-height:0.8;">❤️</div>
      <div class="muted">AI-Based Heart Rate & Stress Assistant</div>
    </div>
    """, unsafe_allow_html=True)
    if latest_entry:
        hr = latest_entry.get("hr", None)
        sp = latest_entry.get("stress_prob", None)
        if hr is not None:
            st.metric("Latest Heart Rate", f"{hr:.0f} bpm")
        if sp is not None:
            label = "Low" if sp<0.33 else "Moderate" if sp<0.66 else "High"
            st.metric("Latest Stress", f"{int(sp*100)}% ({label})")
    else:
        st.info("Run an analysis to populate live preview.")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Weekly Trend"):
        st.session_state["_show_panel"] = "Weekly Trend"
    if col2.button("Generate Report"):
        st.session_state["_show_panel"] = "Generate PDF Report"
    if col3.button("Live Waveform"):
        st.session_state["_show_panel"] = "Live Waveform"
    if col4.button("Doctor Mode"):
        st.session_state["_show_panel"] = "Doctor Mode"
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Sidebar: additional tools ----------
with st.sidebar:
    st.markdown("### Additional Tools")
    if 'panel' not in st.session_state:
        st.session_state['panel'] = None
    panel = st.selectbox("Open panel", ["None","Premium Home","Weekly Trend","Generate PDF Report","Live Waveform","Doctor Mode"], index=0)
    st.session_state['_show_panel'] = panel if panel != "None" else st.session_state.get('_show_panel', None)

# Determine latest entry
history = load_history()
latest = history[-1] if history else None

# React to panel selection
panel = st.session_state.get('_show_panel', None)

if panel == "Premium Home":
    premium_home_view(latest, waveform, lang=lang)

elif panel == "Weekly Trend":
    st.header("Weekly Trend Dashboard")
    stats = compute_weekly_stats()
    if stats is None:
        st.info("No history available. Run at least one test to save history.")
    else:
        st.metric("Avg HR (7d)", f"{stats['avg_hr']:.1f} bpm" if stats['avg_hr'] else "N/A")
        st.metric("Avg Stress (7d)", f"{int(stats['avg_stress']*100)}%" if stats['avg_stress'] else "N/A")
        st.metric("Min HR", f"{stats['min_hr']:.0f} bpm" if stats['min_hr'] else "N/A")
        st.metric("Max HR", f"{stats['max_hr']:.0f} bpm" if stats['max_hr'] else "N/A")
        recent = stats['recent']
        df = pd.DataFrame(recent)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        if not df.empty and 'stress_prob' in df.columns:
            st.bar_chart(df.set_index('time')['stress_prob'].apply(lambda x: x*100))
        else:
            st.info("Not enough data to plot stress bar graph.")

elif panel == "Generate PDF Report":
    st.header("Generate PDF Report")
    if latest is None:
        st.info("No data to generate report. Run analysis first.")
    else:
        st.write("Latest entry will be used for the report.")
        if st.button("Create PDF"):
            out, err = generate_pdf_report(filename="report_latest.pdf", entry=latest, waveform=waveform, lang=lang)
            if err:
                st.error(err)
            else:
                st.success(f"Report saved: {out}")
                with open(out, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name=out.name, mime="application/pdf")

elif panel == "Live Waveform":
    st.header("Live HRV Waveform Playback")
    if waveform is None:
        st.info("No waveform available from the last run. Use Upload Video or Webcam Video and press Predict.")
    else:
        speed = st.slider("Playback speed (smaller=slower)", 0.01, 0.2, 0.03)
        if st.button("Play Waveform"):
            play_live_waveform(waveform, speed=speed)

elif panel == "Doctor Mode":
    st.header("Doctor Mode — Advanced Metrics")
    if features is None:
        st.info("No features available. Run prediction (video/webcam/manual) to compute advanced metrics.")
    else:
        docs = compute_doctor_metrics({k: float(features.get(k, np.nan)) for k in FEATURE_ORDER})
        st.metric("SDNN (ms)", f"{docs['sdnn']:.1f}" if docs['sdnn'] else "N/A")
        st.metric("LF/HF", f"{docs['lf_hf']:.2f}" if docs['lf_hf'] else "N/A")
        st.metric("Estimated Heart Age", f"{docs['heart_age']} yrs")
        st.markdown("### Frequency-domain plot")
        fd = plot_frequency_domain({k: float(features.get(k, np.nan)) for k in FEATURE_ORDER})
        if fd:
            st.image(str(fd))
        else:
            st.info("Frequency-domain not available from current features (need RR series).")

# reset panel dropdown if user chooses None in sidebar
if panel == "None" and '_show_panel' in st.session_state:
    st.session_state['_show_panel'] = None

# end of app
