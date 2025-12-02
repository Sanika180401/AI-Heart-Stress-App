import os, time, tempfile
import warnings
warnings.filterwarnings("ignore")  # hide convergence, deprecation warnings for a clean console

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2

# signal processing
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
import math

# ML
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# explainability + LLM optional
try:
    import shap
except Exception:
    shap = None

try:
    import openai
except Exception:
    openai = None

# -------------------------
# CONFIG & PATHS
# -------------------------
st.set_page_config(page_title="AI HRV Stress & Risk Monitor", layout="wide")
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_LOGO = os.path.join(APP_ROOT, "assets", "logo.png")
MODELS_DIR = os.path.join(APP_ROOT, "models")

# Helper: safe model loader
import pickle
def safe_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

mlp = safe_load(os.path.join(MODELS_DIR, "mlp.pkl"))
rf  = safe_load(os.path.join(MODELS_DIR, "rf.pkl"))
xgb = safe_load(os.path.join(MODELS_DIR, "xgb.pkl")) if XGBClassifier is not None else None
stacker = safe_load(os.path.join(MODELS_DIR, "stacker.pkl"))
scaler = safe_load(os.path.join(MODELS_DIR, "scaler.pkl"))

# if missing, create lightweight demo models so app always runs
def create_demo_models():
    from sklearn.preprocessing import StandardScaler
    Xd = np.random.rand(120,8)
    yd = np.random.randint(0,2,120)
    scaler_local = StandardScaler().fit(Xd)
    Xs = scaler_local.transform(Xd)
    mlp_local = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, tol=1e-4).fit(Xs, yd)
    rf_local = RandomForestClassifier(n_estimators=100).fit(Xs, yd)
    xgb_local = None
    if XGBClassifier is not None:
        try:
            xgb_local = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100).fit(Xs, yd)
        except Exception:
            xgb_local = None
    # stacking meta
    from sklearn.linear_model import LogisticRegression
    probs = [mlp_local.predict_proba(Xs)[:,1], rf_local.predict_proba(Xs)[:,1]]
    if xgb_local is not None:
        probs.append(xgb_local.predict_proba(Xs)[:,1])
    meta = np.vstack(probs).T
    stack = LogisticRegression().fit(meta, yd)
    return mlp_local, rf_local, xgb_local, stack, scaler_local

if mlp is None or rf is None or stacker is None:
    demo = create_demo_models()
    mlp = demo[0] if mlp is None else mlp
    rf  = demo[1] if rf  is None else rf
    if xgb is None: xgb = demo[2]
    stacker = demo[3] if stacker is None else stacker
    if scaler is None: scaler = demo[4]

# SHAP explainer if possible
explainer = None
if shap is not None and rf is not None:
    try:
        explainer = shap.TreeExplainer(rf)
    except Exception:
        explainer = None

# -------------------------
# Signal & HRV Helpers
# -------------------------
def bandpass_filter(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_mean_green_signal_from_video(video_path, max_seconds=12, resize_width=360):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        # center crop ROI to approximate face region
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
    # LF/HF using numpy.trapz to avoid deprecated trapz warnings
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

# -------------------------
# features -> model -> outputs
# -------------------------
FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

def get_model_probability(feats):
    X = vectorize_features(feats)
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    # safe predict_proba
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
# Heuristics for categories and heart-attack chance
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
    # simple transparent formula: combine stress probability and HR relative to normal
    hr_score = max(0.0, (mean_hr - 60.0) / 60.0)  # 0 at 60bpm, 1 at 120bpm
    risk_raw = 0.6 * prob + 0.4 * min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20:
        cat = "Low"
    elif risk_pct < 50:
        cat = "Moderate"
    else:
        cat = "High"
    return risk_pct, cat

def creative_summary_sentence(prob, mean_hr, stress_cat, hr_cat, risk_pct, risk_cat):
    # creative, non-diagnostic sentence
    return (f"Predicted resting heart rate ≈ {mean_hr:.0f} bpm ({hr_cat}). "
            f"Stress level estimated as {stress_cat} (score {prob:.2f}). "
            f"Estimated short-term heart-attack chance: {risk_pct:.0f}% ({risk_cat}). "
            f"Suggestions: short breathing break, hydrate, and monitor symptoms. This is a research demo — not medical advice.")

# -------------------------
# SHAP table helper
# -------------------------
def shap_table(feats):
    if explainer is None:
        return None
    try:
        X = vectorize_features(feats)
        Xs = scaler.transform(X) if scaler is not None else X
        shp = explainer.shap_values(Xs)
        if isinstance(shp, list) and len(shp) > 0:
            vals = np.array(shp[0]).flatten()
        else:
            vals = np.array(shp).flatten()
        df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": vals})
        return df
    except Exception:
        return None

# -------------------------
# OpenAI helper & prompt (optional)
# -------------------------
def call_openai(api_key, prob, feats, risk_pct, risk_cat):
    if openai is None:
        return "OpenAI SDK not installed."
    if not api_key:
        return "No API key provided in sidebar."
    openai.api_key = api_key
    prompt = f"""
You are a friendly, concise health assistant (non-diagnostic).
Model outputs:
- Stress probability: {prob:.3f}
- HRV features: {feats}
- Estimated heart-attack chance: {risk_pct:.0f}% ({risk_cat})

Provide:
1) One-sentence explanation of why the model produced this result.
2) 3 practical recommendations (Immediate: 1-2min; Short term: today; Long-term: weeks).
3) One sentence warning when to seek medical care.

Keep the tone supportive and non-alarming.
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are a helpful, non-diagnostic medical assistant."},{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.7
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI call error: {e}"

# -------------------------
# UI: Header and side controls
# -------------------------
def render_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        if os.path.exists(ASSETS_LOGO):
            st.image(ASSETS_LOGO, width=90)
        else:
            st.markdown("<h2> AI HRV Stress Monitor</h2>", unsafe_allow_html=True)
    with col2:
        st.markdown("### Real-time stress & heart-risk monitoring (demo). Use video for real HRV.")
render_header()
st.write("---")

with st.sidebar:
    st.header("Controls")
    method = st.selectbox("Choose input method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max seconds (video)", 6, 20, 10)
    record_seconds = st.slider("Record seconds (webcam)", 4, 12, 8)
    enable_shap = st.checkbox("Enable SHAP (if available)", True)
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    st.markdown("---")
    st.markdown("Models (optional): place trained models in models/ (mlp.pkl, rf.pkl, xgb.pkl, stacker.pkl, scaler.pkl)")

# -------------------------
# Main UI: handle 5 methods
# -------------------------
if method == "Manual Entry":
    st.subheader("Manual input of HR/HRV (useful for testing)")
    col1, col2 = st.columns(2)
    with col1:
        mean_hr = st.number_input("Mean HR (bpm)", 40, 160, 75)
        rmssd = st.number_input("RMSSD (ms)", 1.0, 300.0, 30.0)
    with col2:
        sd1 = st.number_input("SD1 (ms)", 1.0, 120.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 1.0, 200.0, 40.0)
    if st.button("Predict (Manual)"):
        feats = {"mean_hr":mean_hr,"rmssd":rmssd,"pnn50":np.nan,"sd1":sd1,"sd2":sd2,"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}
        prob, probs = get_model_probability(feats)
        stress_cat = categorize_stress(prob)
        hr_cat = categorize_hr(mean_hr)
        risk_pct, risk_cat = heart_attack_risk_heuristic(prob, mean_hr)
        sent = creative_summary_sentence(prob, mean_hr, stress_cat, hr_cat, risk_pct, risk_cat)
        st.success(sent)
        st.write("Stress category:", stress_cat, " | HR category:", hr_cat, "| Risk:", f"{risk_pct:.0f}% ({risk_cat})")
        if enable_shap:
            dfsh = shap_table(feats)
            if dfsh is not None:
                st.subheader("SHAP contributions")
                st.table(dfsh)
        if openai_key:
            with st.spinner("Calling OpenAI for explanation..."):
                txt = call_openai(openai_key, prob, feats, risk_pct, risk_cat)
                st.subheader("AI recommendation (OpenAI)")
                st.write(txt)

elif method == "Upload Video":
    st.subheader("Upload a short face video (8-15s recommended)")
    uploaded = st.file_uploader("Upload video", type=["mp4","avi","mov"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read()); tmp.flush(); path = tmp.name
        st.info("Processing video for rPPG & HRV...")
        try:
            sig, fps = extract_mean_green_signal_from_video(path, max_seconds=max_seconds)
            mean_hr, hr_series, rr = peaks_and_rr(sig, fps)
            if rr is None:
                st.error("Couldn't detect peaks reliably. Try a longer/clearer video.")
            else:
                feats = compute_hrv_from_rr(rr, hr_series)
                st.subheader("HRV features")
                st.json(feats)
                prob, probs = get_model_probability(feats)
                stress_cat = categorize_stress(prob)
                hr_cat = categorize_hr(feats["mean_hr"])
                risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
                st.metric("Stress probability", f"{prob:.3f}")
                st.write("Stress:", stress_cat, "| HR:", hr_cat, "| Heart-attack risk:", f"{risk_pct:.0f}% ({risk_cat})")
                st.write(creative_summary_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
                if enable_shap:
                    dfsh = shap_table(feats)
                    if dfsh is not None:
                        st.subheader("SHAP contributions")
                        st.table(dfsh)
                if openai_key:
                    with st.spinner("Calling OpenAI..."):
                        txt = call_openai(openai_key, prob, feats, risk_pct, risk_cat)
                        st.subheader("AI recommendation (OpenAI)")
                        st.write(txt)
        except Exception as e:
            st.error(f"Video processing failed: {e}")
        finally:
            try: os.unlink(path)
            except: pass

elif method == "Upload Image":
    st.subheader("Upload a face IMAGE (single image cannot provide real HRV).")
    uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(uploaded.read()); tmp.flush(); path = tmp.name
        img = Image.open(path)
        st.image(img, caption="Uploaded image")
        st.warning("Image-based predictions are SURROGATES — they do not reflect real HRV. Use video/webcam for true HRV.")
        if st.button("Run surrogate prediction on this image"):
            arr = np.array(img.convert("RGB"))
            mean_g = float(np.mean(arr[:,:,1]))
            h,w,_ = arr.shape
            area_proxy = (h*w) / (640*480)
            # create heuristic surrogate features
            feats = {"mean_hr": 72 + (128-mean_g)/18.0, "rmssd": 28.0 - area_proxy*4.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}
            prob, _ = get_model_probability(feats)
            stress_cat = categorize_stress(prob)
            hr_cat = categorize_hr(feats["mean_hr"])
            risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
            st.metric("Stress probability (surrogate)", f"{prob:.3f}")
            st.write(creative_summary_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
            if enable_shap:
                dfsh = shap_table(feats)
                if dfsh is not None:
                    st.subheader("SHAP (surrogate)")
                    st.table(dfsh)
        try: os.unlink(path)
        except: pass

elif method == "Webcam Image":
    st.subheader("Capture a single webcam image (surrogate prediction)")
    cam = st.camera_input("Take a selfie")
    if cam is not None:
        img = Image.open(cam)
        st.image(img, caption="Captured image")
        st.warning("Single image cannot provide HRV — result is a SURROGATE.")
        if st.button("Run surrogate prediction on webcam image"):
            arr = np.array(img.convert("RGB"))
            mean_g = float(np.mean(arr[:,:,1]))
            h,w,_ = arr.shape
            area_proxy = (h*w) / (640*480)
            feats = {"mean_hr": 72 + (128-mean_g)/18.0, "rmssd": 28.0 - area_proxy*4.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}
            prob, _ = get_model_probability(feats)
            stress_cat = categorize_stress(prob)
            hr_cat = categorize_hr(feats["mean_hr"])
            risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
            st.metric("Stress probability (surrogate)", f"{prob:.3f}")
            st.write(creative_summary_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
            if enable_shap:
                dfsh = shap_table(feats)
                if dfsh is not None:
                    st.subheader("SHAP (surrogate)")
                    st.table(dfsh)

elif method == "Webcam Video":
    st.subheader("Record short webcam video (8-12s recommended). Keep face steady.")
    if st.button("Start recording"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fname = tmp.name; tmp.close()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Make sure you run locally and camera is available.")
        else:
            fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam.")
            else:
                h,w = frame.shape[:2]
                out = cv2.VideoWriter(fname, fourcc, fps, (w,h))
                t0 = time.time(); progress = st.progress(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    elapsed = time.time() - t0
                    progress.progress(min(100, int((elapsed/record_seconds)*100)))
                    if elapsed >= record_seconds:
                        break
                out.release(); cap.release()
                st.success("Recording done — processing...")
                try:
                    sig, fps_used = extract_mean_green_signal_from_video(fname, max_seconds=record_seconds)
                    mean_hr, hr_series, rr = peaks_and_rr(sig, fps_used)
                    if rr is None:
                        st.error("Could not detect peaks. Try better lighting and less motion.")
                    else:
                        feats = compute_hrv_from_rr(rr, hr_series)
                        st.subheader("HRV features")
                        st.json(feats)
                        prob, _ = get_model_probability(feats)
                        stress_cat = categorize_stress(prob)
                        hr_cat = categorize_hr(feats["mean_hr"])
                        risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats["mean_hr"])
                        st.metric("Stress probability", f"{prob:.3f}")
                        st.write(creative_summary_sentence(prob, feats["mean_hr"], stress_cat, hr_cat, risk_pct, risk_cat))
                        if enable_shap:
                            dfsh = shap_table(feats)
                            if dfsh is not None:
                                st.subheader("SHAP contributions")
                                st.table(dfsh)
                        if openai_key:
                            with st.spinner("Calling OpenAI..."):
                                txt = call_openai(openai_key, prob, feats, risk_pct, risk_cat)
                                st.subheader("AI recommendation (OpenAI)")
                                st.write(txt)
                except Exception as e:
                    st.error("Processing failed: " + str(e))
                finally:
                    try: os.unlink(fname)
                    except: pass

st.markdown("---")
st.info("This tool is a research/demo system. Not medical advice. For real clinical risk, use validated medical tests and consult a professional.")
