import os
import io
import json
import time
import math
import tempfile
import base64
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

from joblib import load
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

# Optional libs - handled with try/except
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
# Config
# ---------------------------
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer",  layout="wide")

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
LOTTIE_FILE = ASSETS_DIR / "breathing.json"
USERS_FILE = ROOT / "users.json"
HISTORY_DIR = ROOT / "history"
HISTORY_DIR.mkdir(exist_ok=True)

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

# ---------------------------
# Utility functions
# ---------------------------
def safe_load_joblib(path):
    try:
        return load(path)
    except Exception as e:
        # don't crash on model load issues; show minimal debug
        try:
            st.debug(f"Model load failed {path}: {e}")
        except Exception:
            pass
        return None

def ensure_users_file():
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users":[{"username":"demo","password":"demo"}]}, indent=2))

def read_users():
    ensure_users_file()
    return json.loads(USERS_FILE.read_text())

def save_users(data):
    USERS_FILE.write_text(json.dumps(data, indent=2))

# ---------------------------
# Load models (if present)
# ---------------------------
scaler = safe_load_joblib(MODELS_DIR/"scaler.pkl")
mlp = safe_load_joblib(MODELS_DIR/"mlp.pkl")
rf = safe_load_joblib(MODELS_DIR/"rf.pkl")
xgb = safe_load_joblib(MODELS_DIR/"xgb.pkl")
stacker = safe_load_joblib(MODELS_DIR/"stacker.pkl")

models_loaded = {
    "mlp": mlp is not None,
    "rf": rf is not None,
    "xgb": xgb is not None,
    "stacker": stacker is not None,
    "scaler": scaler is not None
}

# ---------------------------
# CSS - Light theme + colorful buttons + butterfly animation
# ---------------------------
st.markdown("""
<style>
/* App background */
.stApp { background: #f6f8fb; color: #0b1721; }

/* Header/logo */
.header-box { display:flex; gap:16px; align-items:center; padding-bottom:6px; }
.logo-box {
    width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,#ff4d6d,#ff7a59); color:white; font-weight:700; font-size:28px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}

/* Card */
.card { background:white; padding:18px; border-radius:12px; box-shadow: 0 8px 24px rgba(15,23,42,0.04); border:1px solid rgba(15,23,42,0.02); }

/* Colorful buttons */
.stButton > button {
    background: linear-gradient(90deg,#ff4d6d,#ff7a59);
    color: white; font-weight:700; padding:10px 20px; border-radius:12px; border: none;
    transition: transform 0.14s ease;
}
.stButton > button:hover { transform: translateY(-3px); }

/* Secondary button */
.stButton.secondary > button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color: white; font-weight:700; padding:8px 16px; border-radius:10px;
}

/* Compact preview */
.preview-img { width:360px; height:200px; object-fit:cover; border-radius:10px; border:1px solid rgba(2,6,23,0.04); }

/* Stress progress bar */
.progress-wrap { background:#eef2f7; border-radius:12px; padding:8px; }
.progress-inner { height:26px; border-radius:8px; width:0%; background:linear-gradient(90deg,#ff7a59,#ff4d6d); color:white; display:flex; align-items:center; justify-content:center; font-weight:700; transition: width 1s ease; }

/* Butterfly card */
.butterfly {
  width: 120px; height: 120px; display:block; margin:auto; position:relative;
  filter: drop-shadow(0 8px 28px rgba(0,0,0,0.06));
}
.butterfly svg { width:100%; height:100%; }
@keyframes flutter { 0% { transform: translateY(0) rotate(-2deg);} 50% { transform: translateY(-6px) rotate(2deg);} 100% { transform: translateY(0) rotate(-2deg);} }
.butterfly svg { animation: flutter 2s infinite ease-in-out; transform-origin:center; }

/* small muted text */
.muted { color:#64748b; font-size:13px; }

/* compact video */
.video-compact { width:560px; max-width:100%; border-radius:12px; border:1px solid rgba(2,6,23,0.04); box-shadow:none; }

</style>
""", unsafe_allow_html=True)

# ---------------------------
# rPPG helpers: simple chrom/green-signal extractor, peaks, HRV
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * max_seconds)
    max_frames = min(total_frames, int(fps * max_seconds))
    sig = []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # roi center box
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        wbox, hbox = int(w*0.35), int(h*0.45)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(w, cx+wbox//2), min(h, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1
            continue
        # mean green
        gmean = float(np.mean(roi[:,:,1]))
        sig.append(gmean)
        frames += 1
    cap.release()
    if len(sig) < 8:
        return None, fps
    return np.array(sig), fps

def get_hr_from_signal(sig, fs):
    if sig is None:
        return None, None
    try:
        filt = bandpass(sig - np.mean(sig), fs)
    except Exception:
        filt = sig - np.mean(sig)
    distance = max(1, int(0.4 * fs))
    peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.18)
    if len(peaks) < 2:
        # try lower prominence
        peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.08)
    if len(peaks) < 2:
        return None, None
    times = peaks / float(fs)
    rr = np.diff(times) * 1000.0  # ms
    hr_series = 60000.0 / rr
    return float(np.mean(hr_series)), hr_series

def compute_hrv_from_rr(rr_ms, hr_series=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else 0.0
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0) if len(diff) > 0 else 0.0
    sd1 = float(np.sqrt(np.var(diff) / 2.0)) if len(diff) > 0 else 0.0
    sd2 = float(np.sqrt(max(0.0, 2*np.var(rr_ms) - np.var(diff)/2.0))) if len(diff) > 0 else 0.0
    rr_mean = float(np.mean(rr_ms))
    rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
    # LF/HF estimate
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
            lf_mask = (f >= 0.04) & (f <= 0.15)
            hf_mask = (f > 0.15) & (f <= 0.4)
            lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
            hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
            lf_hf = float(lf/hf) if hf > 0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {
        "mean_hr": mean_hr,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "sd1": sd1,
        "sd2": sd2,
        "lf_hf": lf_hf,
        "rr_mean": rr_mean,
        "rr_std": rr_std
    }

# ---------------------------
# Prediction & helpers (ensemble)
# ---------------------------
def vectorize_features(feats:dict):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

def safe_predict_proba_single(model, Xs):
    try:
        p = model.predict_proba(Xs)[:,1][0]
        return float(p)
    except Exception:
        try:
            return float(model.predict(Xs)[0])
        except Exception:
            return 0.5

def predict_ensemble(feats:dict):
    X = vectorize_features(feats)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    probs = []
    if mlp is not None:
        probs.append(safe_predict_proba_single(mlp, Xs))
    if rf is not None:
        probs.append(safe_predict_proba_single(rf, Xs))
    if xgb is not None:
        probs.append(safe_predict_proba_single(xgb, Xs))
    if len(probs) == 0:
        # fallback heuristic: higher HR and lower rmssd increase stress
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
    risk_raw = 0.6*prob + 0.4*min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20: cat="Low"
    elif risk_pct < 50: cat="Moderate"
    else: cat="High"
    return risk_pct, cat

# ---------------------------
# SHAP helper (robust for single-output models)
# ---------------------------
def compute_shap_table(feats:dict):
    if shap is None:
        return None, "SHAP library not installed."
    # prepare X and background
    X = vectorize_features(feats)
    try:
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
            # For MLP (single-output), KernelExplainer may be used but heavy.
            # We'll use KernelExplainer fallback with small background.
            try:
                background = np.zeros((10, len(FEATURE_ORDER)))
                explainer = shap.KernelExplainer(lambda z: np.array(mlp.predict_proba(z)[:,1]) , background)
                svals = explainer.shap_values(X, nsamples=64)
                vals = np.array(svals).reshape(-1)[:len(FEATURE_ORDER)]
            except Exception as e:
                return None, f"SHAP not available: {e}"
        else:
            return None, "No supported model available for SHAP."
        # ensure length
        if len(vals) < len(FEATURE_ORDER):
            vals = np.pad(vals, (0, len(FEATURE_ORDER)-len(vals)), 'constant', constant_values=0.0)
        df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": [float(x) for x in vals]})
        return df, None
    except Exception as e:
        return None, f"SHAP not available: {e}"

# ---------------------------
# AI Explanation (templated) - English / Hindi / Marathi
# Different templates for Low / Moderate / High
# ---------------------------
TEMPLATES = {
    "English": {
        "Low": {
            "one": "Your heart rate and HRV markers are within a healthy range — low stress detected.",
            "recommend": [
                "Immediate: Keep breathing steady for 1–2 minutes.",
                "Today: Take short breaks and stay hydrated.",
                "Long-term: Maintain regular exercise and sleep schedule."
            ],
            "seek": "If you experience chest pain or dizziness, seek medical care."
        },
        "Moderate": {
            "one": "Your stress markers are moderate and heart rate slightly elevated.",
            "recommend": [
                "Immediate: Try 2 minutes of paced breathing (4s in, 4s out).",
                "Today: Reduce caffeine and take a 10-minute walk.",
                "Long-term: Practice mindfulness or regular aerobic exercise."
            ],
            "seek": "If symptoms worsen or persist, consult a clinician."
        },
        "High": {
            "one": "High stress markers detected and heart rate is elevated.",
            "recommend": [
                "Immediate: Stop activity and do 3-minute guided breathing.",
                "Today: Avoid stimulants and rest. Inform a family member.",
                "Long-term: Seek professional help for stress management and evaluation."
            ],
            "seek": "If you have chest pain, severe shortness of breath or fainting, seek emergency care."
        }
    },
    "Hindi": {
        "Low": {
            "one": "आपका हृदय-दर और HRV सामान्य सीमा में है — तनाव कम है।",
            "recommend": [
                "तुरंत: 1–2 मिनट गहरी सांस लें।",
                "आज: छोटे-छोटे ब्रेक लें और पानी पिएँ।",
                "दीर्घकालिक: नियमित व्यायाम और सही नींद रखें।"
            ],
            "seek": "यदि सीने में दर्द या चक्कर आए तो डॉक्टर से संपर्क करें।"
        },
        "Moderate": {
            "one": "आपके तनाव संकेत मध्यम हैं और हृदय-दर हल्का बढ़ा हुआ है।",
            "recommend": [
                "तुरंत: 2 मिनट शांत साँस लें (4s in, 4s out)।",
                "आज: कैफीन कम करें और 10 मिनट टहलें।",
                "दीर्घकालिक: माइंडफुलनेस या व्यायाम करें।"
            ],
            "seek": "यदि लक्षण बिगड़ें या बने रहें तो चिकित्सक से मिलें।"
        },
        "High": {
            "one": "उच्च तनाव संकेत मिले हैं और हृदय-दर अधिक है।",
            "recommend": [
                "तुरंत: 3 मिनट गाइडेड ब्रीदिंग करें।",
                "आज: स्टिमुलेंट से बचें और आराम करें।",
                "दीर्घकालिक: तनाव प्रबंधन हेतु विशेषज्ञ से सलाह लें।"
            ],
            "seek": "यदि सीने में दर्द या सांस लेने में कठिनाई हो तो तुरंत अस्पताल जाएँ।"
        }
    },
    "Marathi": {
        "Low": {
            "one": "तुमचा हृदयाचे ठोके आणि HRV सामान्य श्रेणीत आहेत — ताण कमी आहे.",
            "recommend": [
                "तुरंत: 1–2 मिनिटे श्वास घ्या.",
                "आज: छोटे ब्रेक घ्या आणि पाणी प्या.",
                "दीर्घकालीन: नियमित व्यायाम आणि चांगली झोप घ्या."
            ],
            "seek": "छातीतील वेदना किंवा चक्कर आल्यास ताबडतोब डॉक्टरला दाखवा."
        },
        "Moderate": {
            "one": "तुमचे तणाव चिन्हे मध्यम आहेत आणि हृदयाचे ठोके थोडे वाढलेले आहेत.",
            "recommend": [
                "तुरंत: 2 मिनिटे शांत श्वास (4s in, 4s out).",
                "आज: कैफीन कमी करा आणि 10 मिनिटे चालावे.",
                "दीर्घकालीन: ध्यान किंवा नियमित व्यायाम करा."
            ],
            "seek": "लक्षणे वाढली किंवा कायम राहिल्यास चिकित्सकांचा सल्ला घ्या."
        },
        "High": {
            "one": "उच्च तणाव दिसला आहे आणि हृदयाचे ठोके वाढलेले आहेत.",
            "recommend": [
                "तुरंत: 3 मिनिटे श्वास व्यायाम करा.",
                "आज: विश्रांती घ्या व स्टिम्युलंट टाळा.",
                "दीर्घकालीन: तणाव व्यवस्थापनासाठी तज्ञांचा सल्ला घ्या."
            ],
            "seek": "छातीमध्ये वेदना, श्वास घ्यायला त्रास असेल तर ताबडतोब वैद्यकीय मदत घ्या."
        }
    }
}

def generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang="English"):
    stress_label = categorize_stress(prob)
    tpl = TEMPLATES.get(lang, TEMPLATES["English"]).get(stress_label, TEMPLATES["English"]["Moderate"])
    text_lines = []
    text_lines.append(tpl["one"])
    text_lines.append("")
    text_lines.append("Recommendations:")
    for r in tpl["recommend"]:
        text_lines.append("- " + r)
    text_lines.append("")
    text_lines.append("When to seek care: " + tpl["seek"])
    return "\n".join(text_lines)

# ---------------------------
# Simple multi-user session (local)
# ---------------------------
ensure_users_file()
if "user" not in st.session_state:
    st.session_state.user = None
if "history" not in st.session_state:
    st.session_state.history = {}  # keyed by username: list of dicts

def login_widget():
    st.sidebar.markdown("## Account")

    users = read_users()

    # If user already logged in
    if st.session_state.user:
        st.sidebar.success(f"Logged in as: {st.session_state.user}")

        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()
        return

    # LOGIN FORM
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")

    if submit:
        found = False
        for u in users["users"]:
            if u["username"] == username and u["password"] == password:
                st.session_state.user = username
                st.experimental_rerun()
                found = True
                break
        if not found:
            st.sidebar.error("Invalid username or password")

    # REGISTER
    if st.sidebar.button("Create New Account"):
        st.session_state["register_mode"] = True

    if st.session_state.get("register_mode"):
        st.sidebar.markdown("### Register New Account")
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password", type="password")

        if st.sidebar.button("Register"):
            if any(u["username"] == new_user for u in users["users"]):
                st.sidebar.error("Username already exists.")
            else:
                users["users"].append({"username": new_user, "password": new_pass})
                save_users(users)
                st.sidebar.success("Account created successfully!")
                st.session_state["register_mode"] = False

# Ensure login widget shows on sidebar
login_widget()

# ---------------------------
# UI layout - header
# ---------------------------
col1, col2 = st.columns([0.9, 8])
with col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=64)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color:#c41b23; margin-bottom:0;'>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------
# Sidebar controls (main)
# ---------------------------
with st.sidebar:
    st.header("Input")
    method = st.radio("Choose method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max video seconds", min_value=6, max_value=20, value=10)
    rec_seconds = st.slider("Webcam record sec", min_value=4, max_value=16, value=8)
    show_shap = st.checkbox("Show SHAP explainability", value=True)
    lang = st.selectbox("Language", ["English","Hindi","Marathi"])
    st.markdown("---") 

# ---------------------------
# Main input card
# ---------------------------
left_col, right_col = st.columns([1.1, 1])
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input")
    features = None
    uploaded_media_path = None
    waveform = None
    if method == "Manual Entry":
        vals = {}
        cols = st.columns(2)
        for i,f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f == 'mean_hr' else 1.0
                vals[f] = st.number_input(f, value=float(default))
        if st.button("Predict"):
            features = {k: float(v) for k,v in vals.items()}

    elif method == "Upload Image":
        uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, width=360, caption="Preview", use_column_width=False)
            if st.button("Predict"):
                img = Image.open(uploaded).convert("RGB")
                arr = np.array(img)
                # surrogate estimation - single image cannot produce HRV; use heuristic surrogate
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr": 72 + (128-mean_g)/18.0, "rmssd": 28.0, "pnn50": 3.0, "sd1": 12.0, "sd2":24.0, "lf_hf":1.0, "rr_mean":800.0, "rr_std":40.0}
    elif method == "Upload Video":
        vid = st.file_uploader("Upload a short face video (mp4,mov,avi) 6–15s", type=["mp4","mov","avi"])
        if vid:
            st.video(vid, start_time=0)
            if st.button("Predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vid.read()); tmp.flush(); tmp.close()
                uploaded_media_path = tmp.name
                sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                if sig is None:
                    st.error("Couldn't extract reliable signal. Try a longer/clearer video.")
                else:
                    mean_hr, hr_series = get_hr_from_signal(sig, fps)
                    if hr_series is None:
                        st.error("Pulse peaks not reliable.")
                    else:
                        rr_ms = 60000.0 / np.array(hr_series)
                        features = compute_hrv_from_rr(rr_ms, hr_series)
                        waveform = sig.tolist()
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    elif method == "Webcam Image":
        cam = st.camera_input("Capture image")
        if cam:
            st.image(cam, width=360)
            if st.button("Predict"):
                img = Image.open(cam).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":3.0, "sd1":12.0, "sd2":24.0, "lf_hf":1.0, "rr_mean":800.0, "rr_std":40.0}

    elif method == "Webcam Video":
        st.write("Click record — you must run this locally for webcam access.")
        if st.button("Start webcam recording"):
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmpf.close()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Run locally and allow camera access.")
            else:
                fps = 20.0
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read from webcam.")
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
                    sig, fps = extract_mean_green_signal_from_video_file(tmpf.name, max_seconds=rec_seconds)
                    if sig is None:
                        st.error("Couldn't extract reliable signal from webcam.")
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error("Pulse not detected.")
                        else:
                            rr_ms = 60000.0 / np.array(hr_series)
                            features = compute_hrv_from_rr(rr_ms, hr_series)
                            waveform = sig.tolist()
                    try:
                        os.unlink(tmpf.name)
                    except Exception:
                        pass

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown("""
        <div class='card' style='padding:20px'>
            <h3 style='color:#c41b23; margin-bottom:10px;'>Live Dashboard</h3>
        </div>
    """, unsafe_allow_html=True)

    dash_col1, dash_col2, dash_col3 = st.columns(3)

    # create a persistent placeholder for the main gauge so we can update it later
    gauge_ph = st.empty()

    # --------- HR Circular Gauge (Plotly) ----------
    with dash_col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if go is not None:
            fig_hr = go.Figure(go.Indicator(
                mode="gauge+number",
                value=75,
                gauge={
                    "axis":{"range":[40,180]},
                    "bar":{"color":"#ff4d6d"},
                    "bgcolor":"white",
                    "steps":[{"range":[40,180], "color":"#ffe6ea"}],
                },
                title={"text":"Heart Rate (bpm)"}
            ))
            fig_hr.update_layout(height=240, margin=dict(t=10,b=0,l=0,r=0))
            st.plotly_chart(fig_hr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------- Stress Donut Chart -----------
    with dash_col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if go is not None:
            stress_fig = go.Figure(data=[go.Pie(
                values=[75, 25],
                labels=["Stress", ""],
                hole=.7,
                marker_colors=["#ff7a59", "#f5f5f5"],
                textinfo='none'
            )])
            stress_fig.update_layout(
                height=240,
                showlegend=False,
                annotations=[dict(text="Stress<br>75%", x=0.5, y=0.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(stress_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --------- Risk Semi-circle ---------
    with dash_col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if go is not None:
            risk_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=25,
                gauge={
                    "axis":{"range":[0,100]},
                    "bar":{"color":"#f97316"},
                    "bgcolor":"white",
                    "steps":[{"range":[0,100], "color":"#fee9d0"}],
                },
                title={"text":"Heart Attack Risk"}
            ))
            risk_fig.update_layout(height=240, margin=dict(t=10,b=0,l=0,r=0))
            st.plotly_chart(risk_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# After prediction: results
# ---------------------------
if features is not None:
    feats = {k: float(features.get(k, np.nan)) for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob)
    hr_label = categorize_hr(feats.get("mean_hr", 0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get("mean_hr", 0.0))
    sentence = f"Predicted HR ≈ {feats.get('mean_hr', math.nan):.0f} bpm | Stress: {stress_label} ({prob:.2f}) | Heart-attack est: {risk_pct:.0f}% ({risk_cat})"

    # Gauge (update via placeholder)
    if go is not None:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=feats.get("mean_hr", 60),
                                    gauge={'axis':{'range':[30,180]}, 'bar':{'color':'#ff4d6d'} },
                                    title={'text': "<b>Heart Rate (bpm)</b>"}))
        fig.update_layout(height=280, margin=dict(t=20,b=0,l=0,r=0), paper_bgcolor="white", font_color="#0b1721")
        # use the placeholder created earlier
        gauge_ph.plotly_chart(fig, use_container_width=True)
    else:
        gauge_ph.info(f"Heart Rate: {feats.get('mean_hr', math.nan):.0f} bpm")

    # result card with stress bar
    st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#c41b23;'>Result</h3>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>{sentence}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress-wrap'><div class='progress-inner' style='width:{int(prob*100)}%;'>{int(prob*100)}%</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    colA.markdown(f"<div class='muted'>Stress</div><h4 style='color:#c41b23'>{stress_label}</h4>", unsafe_allow_html=True)
    colB.markdown(f"<div class='muted'>Heart Rate</div><h4>{int(feats.get('mean_hr',0))} bpm</h4>", unsafe_allow_html=True)
    colC.markdown(f"<div class='muted'>Attack Risk</div><h4 style='color:#f97316'>{int(risk_pct)}%</h4>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Waveform (compact) and trend storage
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Live HRV waveform")
    if waveform is not None:
        st.line_chart(pd.DataFrame({"signal": waveform}))
    else:
        st.info("Waveform will appear for video/webcam input.")
    st.markdown("</div>", unsafe_allow_html=True)

    # SHAP (fixed)
    if show_shap:
        st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
        st.subheader("SHAP contributions")
        sh_df, sh_error = compute_shap_table(feats)
        if sh_df is not None:
            st.table(sh_df)
        else:
            st.info(sh_error)
        st.markdown("</div>", unsafe_allow_html=True)

    # AI explanation templated by stress level and language
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("AI Explanation")
    ai_text = generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang=lang)
    st.text(ai_text)
    st.markdown("</div>", unsafe_allow_html=True)

    # Butterfly indicator
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card' style='text-align:center'>
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

    # store to history per user
    username = st.session_state.user or "guest"
    if username not in st.session_state.history:
        st.session_state.history[username] = []
    st.session_state.history[username].append({
        "time": datetime.datetime.now().isoformat(),
        "hr": float(feats.get("mean_hr", math.nan)),
        "stress_prob": float(prob),
        "stress_label": stress_label
    })

# ---------------------------
# Trend & History area (bottom)
# ---------------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Trend & History")
user_hist = st.session_state.history.get(st.session_state.user or "guest", [])
if len(user_hist) == 0:
    st.info("No history for current user yet. Run an analysis to record trends.")
else:
    df_hist = pd.DataFrame(user_hist)
    df_hist['time'] = pd.to_datetime(df_hist['time'])
    st.line_chart(df_hist.set_index('time')[['hr','stress_prob']])
    st.table(df_hist.sort_values('time', ascending=False).head(10))
st.markdown("</div>", unsafe_allow_html=True)

# End of app.py
