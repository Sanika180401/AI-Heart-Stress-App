import os
import tempfile
import time
import math
import json
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
    features = None
    waveform = None

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
if 'features' in locals() and features is not None:
    feats = {k: float(features.get(k, np.nan)) for k in FEATURE_ORDER}
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

# End of file
