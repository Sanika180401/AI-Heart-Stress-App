# app.py — Premium Light Theme UI (final)
# - Light theme with red heart accent + colorful buttons
# - Compact previews for uploaded media
# - Webcam video recording (cv2) for local runs
# - Robust OpenAI GPT integration (hidden key via env or openai_key.txt)
# - SHAP explainability guarded
# - Requires: models/scaler.pkl and models/stacker.pkl in models/
# - Optional: assets/logo.png (displayed), assets/breathing.json (lottie)

import os
import time
import tempfile
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image
import cv2

# optional libs
try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

try:
    import shap
except Exception:
    shap = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------------------
# Config + Light theme CSS
# ---------------------------
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer", layout="wide")

st.markdown(
    """
    <style>
    /* Light background */
    .stApp { background: #f7f9fb; color: #0b1721; }
    .header { display:flex; align-items:center; gap:16px; margin-bottom:8px; }
    .logo-box { width:70px; height:70px; border-radius:12px; background:linear-gradient(135deg,#ff4d6d,#ff7a59); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:28px; box-shadow: 0 6px 18px rgba(0,0,0,0.08);}
    h1 { margin:0; color:#c41b23; }
    .subtitle { color:#475569; margin-top:4px; font-size:13px; }

    /* cards */
    .card { background: white; border-radius:12px; padding:16px; box-shadow: 0 6px 20px rgba(15,23,42,0.06); border: 1px solid rgba(15,23,42,0.03); }

    /* colorful buttons */
    .stButton > button {
        background: linear-gradient(90deg,#ff4d6d,#ff7a59);
        color: white;
        border-radius:10px;
        padding:10px 18px;
        font-weight:600;
        border: none;
    }
    .stButton > button:hover { transform: translateY(-2px); }

    /* secondary buttons */
    .stButton.secondary > button {
        background: linear-gradient(90deg,#06b6d4,#3b82f6);
        color: white;
        border-radius:10px;
    }

    /* compact preview */
    .preview-img { width:360px; height:200px; object-fit:cover; border-radius:10px; border:1px solid rgba(2,6,23,0.05); }

    /* stress bar */
    .stress-outer { width:100%; background:#eef2f7; border-radius:10px; padding:6px; }
    .stress-inner { height:26px; border-radius:8px; width:0%; background:linear-gradient(90deg,#ff7a59,#ff4d6d); color:white; display:flex; align-items:center; justify-content:flex-end; padding-right:8px; font-weight:700; transition: width 1s ease; }

    .muted { color:#64748b; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Paths and load models
# ---------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
LOTTIE_BREATH = os.path.join(ASSETS_DIR, "breathing.json")

def load_model_safe(name):
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        try:
            return load(p)
        except Exception:
            return None
    return None

# load required model artifacts
scaler = load_model_safe("scaler.pkl")
stacker = load_model_safe("stacker.pkl")
mlp = load_model_safe("mlp.pkl")
rf = load_model_safe("rf.pkl")
xgb = load_model_safe("xgb.pkl")

# ---------------------------
# Gemini Free API (AI Explanation)
# ---------------------------
import google.generativeai as genai

def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        keyfile = os.path.join(ROOT, "gemini_key.txt")
        if os.path.exists(keyfile):
            with open(keyfile, "r") as f:
                key = f.read().strip()
    if not key:
        return None
    try:
        genai.configure(api_key=key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except:
        return None

gem_client = get_gemini_client()

def call_gpt(feats, prob, risk_pct, risk_cat, lang="English"):
    if gem_client is None:
        return "Gemini AI unavailable — API key missing."

    prompt = f"""
You are a concise medical assistant.

Language: {lang}

HRV: {feats}
Stress Probability: {prob:.2f}
Estimated heart attack risk: {risk_pct:.0f}% ({risk_cat})

Provide:
1. One-line health explanation
2. Three recommendations: immediate, today, long-term
3. One-line: when to seek medical care

Respond ONLY in {lang}.
"""

    try:
        resp = gem_client.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Gemini AI Error: {e}"

# ---------------------------
# rPPG helpers (compact)
# ---------------------------
from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_mean_green_signal_from_video_file(video_path, max_seconds=10, resize_width=360):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")
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
        return None, None
    times = peaks / float(fs)
    rr = np.diff(times) * 1000.0
    hr_series = 60000.0/rr
    return float(np.mean(hr_series)), hr_series

def compute_hrv_from_rr(rr_ms, hr_series=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff) / 2.0))
    sd2 = float(np.sqrt(max(0.0, 2*np.var(rr_ms) - np.var(diff)/2.0)))
    rr_mean = float(np.mean(rr_ms)); rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
    # lf/hf approx
    lf_hf = 0.0
    try:
        fs_interp=4.0
        times=np.cumsum(rr_ms)/1000.0
        if len(times) >= 4:
            t_interp = np.arange(0, times[-1], 1.0/fs_interp)
            inst_hr = 60000.0/rr_ms
            beat_times = times[:-1]
            interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
            f,p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
            lf_mask = (f>=0.04) & (f<=0.15)
            hf_mask = (f>0.15) & (f<=0.4)
            lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
            hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
            lf_hf = float(lf/hf) if hf > 0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {"mean_hr":mean_hr,"rmssd":rmssd,"pnn50":pnn50,"sd1":sd1,"sd2":sd2,"lf_hf":lf_hf,"rr_mean":rr_mean,"rr_std":rr_std}

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

# Ensemble prediction
def predict_ensemble(feats):
    X = vectorize_features(feats)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    probs=[]
    def safe_prob(m):
        try:
            return float(m.predict_proba(Xs)[:,1][0])
        except Exception:
            try:
                return float(m.predict(Xs)[0])
            except Exception:
                return 0.5
    if mlp is not None: probs.append(safe_prob(mlp))
    if rf is not None: probs.append(safe_prob(rf))
    if xgb is not None: probs.append(safe_prob(xgb))
    if len(probs)==0:
        return 0.5, probs
    meta = np.array(probs).reshape(1,-1)
    try:
        final = float(stacker.predict_proba(meta)[:,1][0]) if stacker is not None else float(np.mean(probs))
    except Exception:
        final = float(np.mean(probs))
    return final, probs

def categorize_stress(p):
    if p < 0.35: return "Low"
    if p < 0.65: return "Moderate"
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

# GPT helper (robust)
def call_gpt(feats, prob, risk_pct, risk_cat, lang):
    client = openai_client
    if client is None:
        return "AI assistant unavailable — no OpenAI key."
    prompt = f"""You are a friendly concise health assistant.
HRV: {feats}
Stress probability: {prob:.2f}
Estimated heart-attack chance: {risk_pct:.0f}% ({risk_cat})

Give:
1) One-sentence explanation,
2) Three simple recommendations (immediate, today, long-term),
3) One-sentence when to seek medical care.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=250,
            temperature=0.7
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            # some clients return different structure
            return str(resp)
    except Exception as e:
        return f"OpenAI call error: {e}"

# ---------------------------
# UI Layout - header
# ---------------------------
col_logo, col_title = st.columns([0.8, 6])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("<h1>AI Heart & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>', unsafe_allow_html=True)

st.write("")

# left sidebar controls
with st.sidebar:
    st.header("Input & Settings")
    lang = st.selectbox("Explanation Language", ["English", "Hindi", "Marathi"])
    method = st.radio("Choose method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max video seconds", 6, 20, 10)
    rec_seconds = st.slider("Webcam recording sec", 4, 12, 8)
    show_shap = st.checkbox("Show SHAP explainability", True)
    # show OpenAI status
    if openai_client is None:
        st.markdown("**AI assistant:** <span style='color:#ff4d6d'>Unavailable</span>", unsafe_allow_html=True)
    else:
        st.markdown("**AI assistant:** <span style='color:#10b981'>Available</span>", unsafe_allow_html=True)

st.write("")

# main cards
left, right = st.columns([1, 1.4])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input")
    features = None

    if method == "Manual Entry":
        vals = {}
        cols = st.columns(2)
        for i, f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f=='mean_hr' else 1.0
                vals[f] = st.number_input(f, value=float(default))
        if st.button("Predict"):
            features = vals

    elif method == "Upload Image":
        uploaded = st.file_uploader("Image", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, width=360, caption="Preview", output_format="auto")
            if st.button("Predict"):
                img = Image.open(uploaded).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}

    elif method == "Upload Video":
        vid = st.file_uploader("Video (mp4,mov)", type=["mp4","mov","avi"])
        if vid:
            # compact preview
            st.video(vid, start_time=0)
            if st.button("Predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vid.read()); tmp.flush(); tmp.close()
                try:
                    sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                    if sig is None:
                        st.error("Couldn't extract a reliable signal — use a longer/clearer video.")
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error("Pulse peaks not reliable — try again.")
                        else:
                            rr_ms = np.diff(np.array([0] + list(60000.0/np.array(hr_series))))  # approx
                            # better: convert hr_series -> rr_ms
                            rr_ms = (60000.0/np.array(hr_series))
                            feats = compute_hrv_from_rr(rr_ms, hr_series)
                            features = feats
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
                st.error("Cannot access webcam. Ensure camera is allowed and you're running locally.")
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
                                feats = compute_hrv_from_rr(rr_ms, hr_series)
                                features = feats
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
# After prediction: display results
# ---------------------------
if 'features' in locals() and features is not None:
    feats = {k: float(features.get(k, np.nan)) for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob)
    hr_label = categorize_hr(feats.get("mean_hr", 0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get("mean_hr", 0.0))
    sentence = f"HR ≈ {feats.get('mean_hr', np.nan):.0f} bpm ({hr_label}). Stress: {stress_label} ({prob:.2f}). Heart-attack estimate: {risk_pct:.0f}% ({risk_cat})."

    # show gauge (plotly)
    if go is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=feats.get("mean_hr", 60),
            gauge={'axis':{'range':[30,180]}, 'bar':{'color':'#ff4d6d'}},
            title={'text': "<b>Heart Rate (bpm)</b>"}
        ))
        fig.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="white", font_color="#0b1721")
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        gauge_placeholder.info(f"HR: {feats.get('mean_hr', np.nan):.0f} bpm")

    # neon result card (light)
    st.markdown(
        f"""
        <div class="card" style="margin-top:12px">
          <h3 style="color:#c41b23; margin-bottom:6px">Result</h3>
          <div class="muted">{sentence}</div>
          <div style="height:10px"></div>
          <div class="stress-outer"><div class="stress-inner" style="width:{int(prob*100)}%;">{int(prob*100)}%</div></div>
          <div style="height:12px"></div>
          <div style="display:flex; gap:10px;">
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#c41b23">{stress_label}</b><div class="small muted">Stress</div></div>
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#0b1721">{int(feats.get('mean_hr',0))} bpm</b><div class="small muted">Heart Rate</div></div>
            <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#f97316">{int(risk_pct)}%</b><div class="small muted">Attack Risk</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP table
    if show_shap and shap is not None and (rf is not None or mlp is not None or xgb is not None):
        st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
        st.subheader("SHAP contributions")
        try:
            model_for_shap = rf or mlp or stacker
            explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(np.zeros((1,len(FEATURE_ORDER)))))
            svals = explainer(vectorize_features(feats))
            shap_vals = np.array(svals.values).reshape(-1)[:len(FEATURE_ORDER)]
            sh_df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": shap_vals})
            st.table(sh_df)
        except Exception as e:
            st.info(f"SHAP not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # GPT Explanation
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("AI Explanation (GPT)")
    gpt_out = call_gpt(feats, prob, risk_pct, risk_cat, lang)
    st.write(gpt_out)
    st.markdown("</div>", unsafe_allow_html=True)

# small spacer
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
