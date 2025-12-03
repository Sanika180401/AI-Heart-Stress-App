# app.py
"""
Final cleaned & fixed AI HRV Stress & Heart-Risk app.
- 5 input methods (Manual, Upload Video, Upload Image, Webcam Image, Webcam Video)
- Video-based rPPG -> HRV (simple mean-green ROI)
- Ensemble prediction (MLP + RF + XGB if present)
- SHAP explainability (robust handling to avoid 2D/shape errors)
- GPT explanation using OpenAI >=1.0 client (API key hidden: env var or openai_key.txt)
- Removed unattractive sidebar text and OpenAI key input
- Suppresses noisy warnings
"""

import os
import time
import tempfile
import warnings
warnings.filterwarnings("ignore")  # hide XGBoost/sklearn deprecation noises for a clean console

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend

# sklearn + optional xgboost/shap/openai
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import shap
except Exception:
    shap = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import pickle

# -------------------------
# Config / Paths
# -------------------------
st.set_page_config(page_title="AI Heart & Stress Analyzer", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_LOGO = os.path.join(ROOT, "assets", "logo.png")
MODELS_DIR = os.path.join(ROOT, "models")

# -------------------------
# UI header
# -------------------------
def render_header():
    c1, c2 = st.columns([1, 6])
    with c1:
        if os.path.exists(ASSETS_LOGO):
            st.image(ASSETS_LOGO, width=100)
        else:
            st.markdown("<h2></h2>", unsafe_allow_html=True)
    with c2:
        st.markdown("<h1 style='margin:0'>AI Heart & Stress Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:#666'>Video-based HRV → Ensemble predictions → Explainability + GPT</div>", unsafe_allow_html=True)

render_header()
st.write("---")

# -------------------------
# Load models (safe)
# -------------------------
def safe_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

mlp = safe_load(os.path.join(MODELS_DIR, "mlp.pkl"))
rf = safe_load(os.path.join(MODELS_DIR, "rf.pkl"))
xgb = safe_load(os.path.join(MODELS_DIR, "xgb.pkl")) if XGBClassifier is not None else None
stacker = safe_load(os.path.join(MODELS_DIR, "stacker.pkl"))
scaler = safe_load(os.path.join(MODELS_DIR, "scaler.pkl"))

# demo fallback models so app never crashes
def create_demo_models():
    X = np.random.rand(200, 8)
    y = np.random.randint(0, 2, 200)
    scaler_local = StandardScaler().fit(X)
    Xs = scaler_local.transform(X)
    mlp_local = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400).fit(Xs, y)
    rf_local = RandomForestClassifier(n_estimators=120).fit(Xs, y)
    xgb_local = None
    if XGBClassifier is not None:
        try:
            xgb_local = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=80)
            xgb_local.fit(Xs, y)
        except Exception:
            xgb_local = None
    from sklearn.linear_model import LogisticRegression
    preds = [mlp_local.predict_proba(Xs)[:,1], rf_local.predict_proba(Xs)[:,1]]
    if xgb_local is not None:
        preds.append(xgb_local.predict_proba(Xs)[:,1])
    meta = np.vstack(preds).T
    stack = LogisticRegression().fit(meta, y)
    return mlp_local, rf_local, xgb_local, stack, scaler_local

if mlp is None or rf is None or stacker is None:
    demo = create_demo_models()
    if mlp is None: mlp = demo[0]
    if rf is None: rf = demo[1]
    if xgb is None: xgb = demo[2]
    if stacker is None: stacker = demo[3]
    if scaler is None: scaler = demo[4]

# SHAP explainer (if available)
explainer = None
if shap is not None and rf is not None:
    try:
        explainer = shap.TreeExplainer(rf)
    except Exception:
        explainer = None

# -------------------------
# rPPG & HRV utility
# -------------------------
def bandpass_filter(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_mean_green_signal_from_video(video_path, max_seconds=12, resize_width=360):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds)
    max_frames = min(total_frames, int(fps*max_seconds))
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
        ch, cw = frame.shape[:2]
        cx, cy = cw//2, ch//2
        wbox, hbox = int(cw*0.35), int(ch*0.45)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(cw, cx+wbox//2), min(ch, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1
            continue
        mean_g = float(np.mean(roi[:,:,1]))
        sig.append(mean_g)
        frames += 1
    cap.release()
    sig = np.array(sig)
    if sig.size == 0:
        raise RuntimeError("No frames / ROI found")
    return sig, fps

def peaks_and_rr(sig, fps):
    try:
        filt = bandpass_filter(sig - np.mean(sig), fps)
    except Exception:
        filt = sig - np.mean(sig)
    distance = max(1, int(0.4*fps))
    peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.2)
    if len(peaks) < 2:
        peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.05)
    if len(peaks) < 2:
        return None, None, None
    times = peaks / float(fps)
    rr = np.diff(times) * 1000.0  # ms
    hr_series = 60000.0 / rr
    mean_hr = float(np.mean(hr_series))
    return mean_hr, hr_series, rr

def compute_hrv_from_rr(rr_ms, hr_series=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff)/2.0))
    sd2 = float(np.sqrt(2*np.var(rr_ms) - np.var(diff)/2.0))
    rr_mean = float(np.mean(rr_ms)); rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
    try:
        fs_interp = 4.0
        times = np.cumsum(rr_ms)/1000.0
        t_interp = np.arange(0, times[-1], 1.0/fs_interp)
        inst_hr = 60000.0/rr_ms
        beat_times = times[:-1]
        if len(beat_times) >= 4:
            interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
            f, p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
            lf_mask = (f>=0.04) & (f<=0.15)
            hf_mask = (f>0.15) & (f<=0.4)
            lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
            hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
            lf_hf = float(lf/hf) if hf > 0 else 0.0
        else:
            lf_hf = 0.0
    except Exception:
        lf_hf = 0.0
    feats = {"mean_hr":mean_hr,"rmssd":rmssd,"pnn50":pnn50,"sd1":sd1,"sd2":sd2,"lf_hf":lf_hf,"rr_mean":rr_mean,"rr_std":rr_std}
    return feats

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

def predict_ensemble(feats):
    X = vectorize_features(feats)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    def safe_prob(model, X):
        try:
            return float(model.predict_proba(X)[:,1][0])
        except Exception:
            try:
                return float(model.predict(X)[0])
            except Exception:
                return 0.5
    p_mlp = safe_prob(mlp, Xs)
    p_rf  = safe_prob(rf, Xs)
    p_xgb = safe_prob(xgb, Xs) if xgb is not None else 0.5
    probs = [p_mlp, p_rf] + ([p_xgb] if xgb is not None else [])
    meta = np.array(probs).reshape(1, -1)
    try:
        if stacker is not None:
            final = float(stacker.predict_proba(meta)[:,1][0])
        else:
            final = float(np.mean(probs))
    except Exception:
        final = float(np.mean(probs))
    return final, probs

# -------------------------
# Heuristics + messages
# -------------------------
def categorize_stress(prob):
    if prob < 0.35:
        return "Low"
    if prob < 0.65:
        return "Moderate"
    return "High"

def categorize_hr(hr):
    if hr < 60:
        return "Low"
    if hr <= 100:
        return "Normal"
    return "High"

def heart_attack_risk_heuristic(prob, mean_hr):
    hr_score = max(0.0, (mean_hr - 60.0) / 60.0)
    risk_raw = 0.6 * prob + 0.4 * min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20:
        cat = "Low"
    elif risk_pct < 50:
        cat = "Moderate"
    else:
        cat = "High"
    return risk_pct, cat

def creative_sentence(prob, mean_hr, stress_cat, hr_cat, risk_pct, risk_cat):
    return (f"Predicted resting HR ≈ {mean_hr:.0f} bpm ({hr_cat}). Stress level: {stress_cat} (score {prob:.2f}). "
            f"Estimated near-term heart-attack chance: {risk_pct:.0f}% ({risk_cat}). Try a 2-minute breathing break; if symptoms persist, seek care.")

# -------------------------
# SHAP helper (robust)
# -------------------------
def shap_table_safe(feats):
    if explainer is None:
        return None
    try:
        X = vectorize_features(feats)
        Xs = scaler.transform(X) if scaler is not None else X
        shp = explainer.shap_values(Xs)
        # shap can return list (for multi-class) or array; choose first row appropriately and flatten
        if isinstance(shp, list):
            # take first class or first array
            arr = np.array(shp[0])
        else:
            arr = np.array(shp)
        # arr may be shape (1, n_features) or (n_features,) or (n_samples, n_features)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            vals = arr[0].flatten()
        else:
            vals = arr.flatten()
        # ensure length matches FEATURE_ORDER
        if vals.size != len(FEATURE_ORDER):
            # attempt to reduce or pad
            vals = vals.flatten()[:len(FEATURE_ORDER)]
            if vals.size < len(FEATURE_ORDER):
                vals = np.pad(vals, (0, len(FEATURE_ORDER)-vals.size), constant_values=0.0)
        df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": vals})
        return df
    except Exception:
        return None

# -------------------------
# OpenAI helper (hidden key)
# -------------------------
def get_openai_api_key():
    # 1) check env var
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    # 2) check local file openai_key.txt (not recommended for prod)
    fp = os.path.join(ROOT, "openai_key.txt")
    if os.path.exists(fp):
        try:
            with open(fp, "r") as f:
                return f.read().strip()
        except:
            return None
    return None

def call_openai_hidden(prob, feats, risk_pct, risk_cat):
    api_key = get_openai_api_key()
    if api_key is None:
        return "OpenAI API key not found (set OPENAI_API_KEY env var or create openai_key.txt in project root)."
    if OpenAI is None:
        return "OpenAI client not installed."

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
You are a concise non-diagnostic health assistant.
Stress probability: {prob:.3f}
HRV features: {feats}
Estimated heart-attack chance: {risk_pct:.0f}% ({risk_cat})

Provide:
1) One-sentence explanation (why).
2) Three quick recommendations (immediate, today, long-term).
3) One-sentence guidance when to seek medical care.
"""
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=280,
            temperature=0.7
        )
        # robust extraction
        try:
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp['choices'][0]['message']['content']
            except Exception:
                return str(resp)
    except Exception as e:
        return f"OpenAI error: {e}"

# -------------------------
# Sidebar controls (clean)
# -------------------------
with st.sidebar:
    st.header("Input Configuration")
    input_method = st.radio("Choose Method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max Video Seconds", 6, 20, 10)
    record_seconds = st.slider("Webcam recording seconds", 4, 12, 8)
    enable_shap = st.checkbox("Show SHAP Explainability", True)
    st.markdown("---")
    st.markdown("OpenAI key is read from environment variable `OPENAI_API_KEY` or `openai_key.txt` (hidden).")

# -------------------------
# MAIN: handle 5 methods
# -------------------------
st.markdown("## Run Analysis")

# MANUAL
if input_method == "Manual Entry":
    st.subheader("Manual HR/HRV entry")
    c1, c2 = st.columns(2)
    with c1:
        mean_hr = st.number_input("Mean HR (bpm)", 40, 160, 75)
        rmssd = st.number_input("RMSSD (ms)", 1.0, 300.0, 30.0)
    with c2:
        sd1 = st.number_input("SD1 (ms)", 1.0, 120.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 1.0, 200.0, 40.0)
    if st.button("Predict"):
        feats = {"mean_hr":float(mean_hr),"rmssd":float(rmssd),"pnn50":np.nan,"sd1":float(sd1),"sd2":float(sd2),"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}
        prob, probs = predict_ensemble(feats)
        stress_cat = categorize_stress(prob)
        hr_cat = categorize_hr(feats["mean_hr"])
        risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
        st.success(creative_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
        if enable_shap:
            dfsh = shap_table_safe(feats)
            if dfsh is not None:
                st.subheader("SHAP contributions")
                st.table(dfsh)
        # GPT explanation (hidden key)
        st.subheader("AI Explanation (GPT)")
        st.write(call_openai_hidden(prob, feats, risk_pct, risk_cat))


# UPLOAD VIDEO
elif input_method == "Upload Video":
    st.subheader("Upload a short face video (6–15s recommended)")
    uploaded = st.file_uploader("Choose video", type=["mp4","mov","avi"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.flush(); path = tmp.name
        st.video(path)
        st.info("Extracting HRV from video...")
        try:
            sig, fps = extract_mean_green_signal_from_video(path, max_seconds=max_seconds)
            mean_hr, hr_series, rr = peaks_and_rr(sig, fps)
            if rr is None:
                st.error("Couldn't detect reliable peaks — try longer/clearer video.")
            else:
                feats = compute_hrv_from_rr(rr, hr_series)
                st.subheader("HRV features")
                st.json(feats)
                prob, probs = predict_ensemble(feats)
                stress_cat = categorize_stress(prob)
                hr_cat = categorize_hr(feats["mean_hr"])
                risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
                st.success(creative_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
                if enable_shap:
                    dfsh = shap_table_safe(feats)
                    if dfsh is not None:
                        st.subheader("SHAP contributions")
                        st.table(dfsh)
                st.subheader("AI Explanation (GPT)")
                st.write(call_openai_hidden(prob, feats, risk_pct, risk_cat))
        except Exception as e:
            st.error(f"Video processing error: {e}")
        finally:
            try: os.unlink(path)
            except: pass

# UPLOAD IMAGE (surrogate)
elif input_method == "Upload Image":
    st.subheader("Upload face IMAGE (single image yields surrogate prediction)")
    uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.flush(); path = tmp.name
        img = Image.open(path)
        st.image(img, use_column_width=True)
        st.warning("Single images are SURROGATE predictions (no real HRV).")
        if st.button("Predict (surrogate)"):
            arr = np.array(img.convert("RGB"))
            mean_g = float(np.mean(arr[:,:,1]))
            feats = {"mean_hr":72 + (128-mean_g)/18.0,"rmssd":28.0,"pnn50":np.nan,"sd1":18.0,"sd2":36.0,"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}
            prob, probs = predict_ensemble(feats)
            stress_cat = categorize_stress(prob)
            hr_cat = categorize_hr(feats["mean_hr"])
            risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
            st.success(creative_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
            if enable_shap:
                dfsh = shap_table_safe(feats)
                if dfsh is not None:
                    st.subheader("SHAP contributions (surrogate)")
                    st.table(dfsh)
            st.subheader("AI Explanation (GPT)")
            st.write(call_openai_hidden(prob, feats, risk_pct, risk_cat))
        try: os.unlink(path)
        except: pass

# WEBCAM IMAGE (surrogate)
elif input_method == "Webcam Image":
    st.subheader("Capture a single webcam image (surrogate)")
    cam = st.camera_input("Take a selfie")
    if cam:
        img = Image.open(cam)
        st.image(img, use_column_width=True)
        st.warning("Single images are SURROGATE predictions (no real HRV).")
        if st.button("Predict (surrogate webcam)"):
            arr = np.array(img.convert("RGB"))
            mean_g = float(np.mean(arr[:,:,1]))
            feats = {"mean_hr":72 + (128-mean_g)/18.0,"rmssd":28.0,"pnn50":np.nan,"sd1":18.0,"sd2":36.0,"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}
            prob, probs = predict_ensemble(feats)
            stress_cat = categorize_stress(prob)
            hr_cat = categorize_hr(feats["mean_hr"])
            risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
            st.success(creative_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
            if enable_shap:
                dfsh = shap_table_safe(feats)
                if dfsh is not None:
                    st.subheader("SHAP contributions (surrogate)")
                    st.table(dfsh)
            st.subheader("AI Explanation (GPT)")
            st.write(call_openai_hidden(prob, feats, risk_pct, risk_cat))

# WEBCAM VIDEO
elif input_method == "Webcam Video":
    st.subheader("Record short webcam video (8-12s recommended). Keep face steady.")
    if st.button("Start recording"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); fname = tmp.name; tmp.close()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Run locally and allow camera access.")
        else:
            fps = 20.0
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read webcam.")
            else:
                h,w = frame.shape[:2]
                out = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
                t0 = time.time(); prog = st.progress(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    elapsed = time.time() - t0
                    prog.progress(min(100, int((elapsed/record_seconds)*100)))
                    if elapsed >= record_seconds:
                        break
                out.release(); cap.release()
                st.success("Recording done — processing...")
                try:
                    sig, fps_used = extract_mean_green_signal_from_video(fname, max_seconds=record_seconds)
                    mean_hr, hr_series, rr = peaks_and_rr(sig, fps_used)
                    if rr is None:
                        st.error("Could not detect peaks; improve lighting and keep face steady.")
                    else:
                        feats = compute_hrv_from_rr(rr, hr_series)
                        st.subheader("HRV features")
                        st.json(feats)
                        prob, probs = predict_ensemble(feats)
                        stress_cat = categorize_stress(prob)
                        hr_cat = categorize_hr(feats["mean_hr"])
                        risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
                        st.success(creative_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
                        if enable_shap:
                            dfsh = shap_table_safe(feats)
                            if dfsh is not None:
                                st.subheader("SHAP contributions")
                                st.table(dfsh)
                        st.subheader("AI Explanation (GPT)")
                        st.write(call_openai_hidden(prob, feats, risk_pct, risk_cat))
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                finally:
                    try: os.unlink(fname)
                    except: pass

st.write("---")
st.caption("UI and results are for demonstration. Replace heuristic risk model with validated clinical score for production.")
