import os
import time
import math
import tempfile
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# signal + CV
import cv2
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend

# ML libs
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# explainability & LLM
try:
    import shap
except Exception:
    shap = None

try:
    import openai
except Exception:
    openai = None

# ---------------------------
# Config & utility
# ---------------------------
st.set_page_config(page_title="AI HRV Stress Monitor", layout="wide", initial_sidebar_state="expanded")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS_LOGO = os.path.join(APP_ROOT, "assets", "logo.png")
ROOT_LOGO = os.path.join(APP_ROOT, "logo.png")
MODELS_DIR = os.path.join(APP_ROOT, "models")
MLP_PATH = os.path.join(MODELS_DIR, "mlp.pkl")
RF_PATH = os.path.join(MODELS_DIR, "rf.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb.pkl")
STACKER_PATH = os.path.join(MODELS_DIR, "stacker.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# helper: load pickles safely
def safe_load(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load {os.path.basename(path)}: {e}")
            return None
    return None

# ---------------------------
# Top bar / logo & title
# ---------------------------
def render_header():
    logo = None
    if os.path.exists(ASSETS_LOGO):
        logo = ASSETS_LOGO
    elif os.path.exists(ROOT_LOGO):
        logo = ROOT_LOGO

    col1, col2 = st.columns([1, 6])
    with col1:
        if logo:
            st.image(logo, width=110)
        else:
            st.markdown("<div style='font-size:26px; font-weight:700;'></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='margin:0'>AI Heart Rate & Stress Monitoring</h1>", unsafe_allow_html=True)
        st.markdown("rPPG → HRV → Ensemble (MLP+RF+XGB) → SHAP → LLM explanations")
render_header()
st.write("---")

# ---------------------------
# load models if available
# ---------------------------
mlp = safe_load(MLP_PATH)
rf = safe_load(RF_PATH)
xgb = safe_load(XGB_PATH) if XGBClassifier is not None else None
stacker = safe_load(STACKER_PATH)
scaler = safe_load(SCALER_PATH)

# create demo models if not present (so app still runs)
def create_demo_models():
    X_demo = np.random.rand(120, 8)
    y_demo = np.random.randint(0, 2, 120)
    scaler_local = None
    try:
        from sklearn.preprocessing import StandardScaler
        scaler_local = StandardScaler().fit(X_demo)
        Xs = scaler_local.transform(X_demo)
    except Exception:
        Xs = X_demo
    mlp_local = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300).fit(Xs, y_demo)
    rf_local = RandomForestClassifier(n_estimators=100).fit(Xs, y_demo)
    xgb_local = None
    if XGBClassifier is not None:
        try:
            xgb_local = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100).fit(Xs, y_demo)
        except Exception:
            xgb_local = None
    # simple stacking via logistic
    from sklearn.linear_model import LogisticRegression
    meta_X = np.vstack([mlp_local.predict_proba(Xs)[:,1], rf_local.predict_proba(Xs)[:,1]] + ([xgb_local.predict_proba(Xs)[:,1]] if xgb_local is not None else []) ).T
    stack = LogisticRegression().fit(meta_X, y_demo)
    return mlp_local, rf_local, xgb_local, stack, scaler_local

if mlp is None or rf is None or stacker is None:
    demo_mlps = create_demo_models()
    mlp = demo_mlps[0] if mlp is None else mlp
    rf = demo_mlps[1] if rf is None else rf
    if xgb is None: xgb = demo_mlps[2]
    stacker = demo_mlps[3] if stacker is None else stacker
    if scaler is None:
        scaler = demo_mlps[4]
else:
    # use the loaded scaler if present
    if scaler is None:
        scaler = safe_load(SCALER_PATH)

# SHAP explainer (tree)
explainer = None
if shap is not None and rf is not None:
    try:
        explainer = shap.TreeExplainer(rf)
    except Exception:
        explainer = None

# ---------------------------
# signal & HRV helpers
# ---------------------------
def bandpass_filter(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def extract_signal_from_video(video_path, max_seconds=12, resize_width=320):
    """
    Primitive rPPG: mean-green in center face region (faster & simple).
    Returns raw signal (np.array) and fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds)
    max_frames = min(total_frames, int(fps*max_seconds))
    raw = []
    frames = 0
    last_roi = None
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # center crop as fallback
        ch, cw = frame.shape[:2]
        cx, cy = cw//2, ch//2
        wbox, hbox = int(cw*0.35), int(ch*0.4)
        x1, y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2, y2 = min(cw, cx+wbox//2), min(ch, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1
            continue
        mean_g = float(np.mean(roi[:,:,1]))
        raw.append(mean_g)
        frames += 1
    cap.release()
    raw = np.array(raw)
    if raw.size == 0:
        raise RuntimeError("No usable frames extracted from video.")
    return raw, fps

def estimate_hr_rr_from_signal(sig, fps):
    # filter
    try:
        filt = bandpass_filter(sig - np.mean(sig), fps)
    except Exception:
        filt = sig - np.mean(sig)
    # detect peaks
    distance = max(1, int(0.4*fps))
    peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.2)
    if len(peaks) < 2:
        peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.05)
    if len(peaks) < 2:
        return None, None, None
    times = peaks / float(fps)
    rr = np.diff(times) * 1000.0
    hr_series = 60000.0 / rr
    hr_mean = float(np.mean(hr_series))
    return hr_mean, hr_series, rr

def compute_hrv_features_from_rr(rr_ms, hr_series=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff)/2.0))
    sd2 = float(np.sqrt(2*np.var(rr_ms) - np.var(diff)/2.0))
    rr_mean = float(np.mean(rr_ms))
    rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
    # LF/HF approx (may be unreliable on short recordings)
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
            lf_hf = float(lf/hf) if hf>0 else 0.0
        else:
            lf_hf = 0.0
    except Exception:
        lf_hf = 0.0
    feats = {
        "mean_hr": mean_hr,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "sd1": sd1,
        "sd2": sd2,
        "lf_hf": lf_hf,
        "rr_mean": rr_mean,
        "rr_std": rr_std
    }
    return feats

# ---------------------------
# Classification helpers
# ---------------------------
FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def features_to_vector(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1, -1)

def classify_features(feats):
    X = features_to_vector(feats)
    used = "heuristic"
    # if scaler available, scale
    try:
        Xs = scaler.transform(X) if scaler is not None else X
    except Exception:
        Xs = X
    # get base probs
    try:
        p_mlp = mlp.predict_proba(Xs)[:,1][0] if hasattr(mlp, "predict_proba") else float(mlp.predict(Xs)[0])
    except Exception:
        p_mlp = 0.5
    try:
        p_rf = rf.predict_proba(Xs)[:,1][0] if hasattr(rf, "predict_proba") else float(rf.predict(Xs)[0])
    except Exception:
        p_rf = 0.5
    p_xgb = 0.5
    if xgb is not None:
        try:
            p_xgb = xgb.predict_proba(Xs)[:,1][0] if hasattr(xgb, "predict_proba") else float(xgb.predict(Xs)[0])
        except Exception:
            p_xgb = 0.5
    # meta stacking if stacker available
    probs = [p_mlp, p_rf] + ([p_xgb] if xgb is not None else [])
    meta = np.array(probs).reshape(1, -1)
    if stacker is not None:
        try:
            final_prob = float(stacker.predict_proba(meta)[:,1][0])
            used = "stacker"
        except Exception:
            final_prob = float(np.mean(probs))
    else:
        final_prob = float(np.mean(probs))
    label = "Stress" if final_prob >= 0.5 else "No Stress"
    return final_prob, label, used

# ---------------------------
# Adaptive threshold
# ---------------------------
if 'baseline_probs' not in st.session_state:
    st.session_state['baseline_probs'] = []

def adaptive_decision(prob, k=0.9, min_thresh=0.35, max_thresh=0.75):
    baseline = st.session_state['baseline_probs']
    if len(baseline) < 12:
        thresh = 0.5
    else:
        mu = float(np.mean(baseline)); sigma = float(np.std(baseline))
        thresh = float(np.clip(mu + k*sigma, min_thresh, max_thresh))
    baseline.append(prob)
    if len(baseline) > 500:
        baseline.pop(0)
    st.session_state['baseline_probs'] = baseline
    return prob >= thresh, float(thresh)

# ---------------------------
# SHAP explain
# ---------------------------
def shap_table_for_features(feats):
    try:
        X = features_to_vector(feats)
        Xs = scaler.transform(X) if scaler is not None else X
        if explainer is None:
            return None
        shp = explainer.shap_values(Xs)
        # shap_values shape differs by version; handle gracefully
        if isinstance(shp, list) and len(shp) > 0:
            vals = np.array(shp[0]).flatten()
        else:
            vals = np.array(shp).flatten()
        df = pd.DataFrame({"feature": FEATURE_ORDER, "shap": vals})
        return df
    except Exception:
        return None

# ---------------------------
# OpenAI integration
# ---------------------------
def call_openai_api(api_key, payload):
    if openai is None:
        return "OpenAI library not installed."
    if not api_key:
        return "No API key provided."
    openai.api_key = api_key
    system_prompt = "You are a helpful non-diagnostic health assistant."
    user_prompt = f"""Model outputs:
Probability: {payload.get('prob'):.3f}
Label: {payload.get('label')}
Top features: {payload.get('top_features')}
HRV features: {payload.get('feats')}

Provide:
1) One-sentence explanation.
2) 3 short recommendations: immediate (1-2 min), short-term (day), long-term (weeks).
3) When to seek medical help (1 sentence).
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ---------------------------
# UI Controls
# ---------------------------
with st.sidebar:
    st.header("Settings & Controls")
    input_method = st.radio("Input method", ["Manual Entry", "Upload (image/video)", "Webcam Record"])
    max_video_seconds = st.slider("Max seconds to process (video)", 6, 20, 10)
    record_seconds = st.slider("Webcam record seconds", 4, 12, 8)
    enable_shap = st.checkbox("Enable SHAP (if available)", True)
    openai_key = st.text_input("OpenAI API Key (optional)", type="password")
    st.markdown("---")
    st.markdown("Model files (optional): place mlp.pkl, rf.pkl, xgb.pkl, stacker.pkl, scaler.pkl in `models/`")
    st.markdown("Note: For robust HRV use well-lit 8-12s video, minimal motion.")

# ---------------------------
# Main interactive area
# ---------------------------
st.markdown("<style>.stButton>button{background-color:#4f46e5;color:white}</style>", unsafe_allow_html=True)

if input_method == "Manual Entry":
    st.subheader("Manual Entry (quick)")
    col1, col2 = st.columns(2)
    with col1:
        mean_hr = st.number_input("Mean HR (bpm)", 40, 160, 75)
        rmssd = st.number_input("RMSSD (ms)", 1.0, 300.0, 30.0)
    with col2:
        sd1 = st.number_input("SD1 (ms)", 1.0, 100.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 1.0, 200.0, 40.0)
    if st.button("Predict (Manual)"):
        feats = {'mean_hr':float(mean_hr),'rmssd':float(rmssd),'pnn50':np.nan,'sd1':float(sd1),'sd2':float(sd2),'lf_hf':np.nan,'rr_mean':np.nan,'rr_std':np.nan}
        prob, label, used = classify_features(feats)
        decision, thresh = adaptive_decision(prob)
        st.metric("Stress probability", f"{prob:.3f}")
        st.write(f"Adaptive label (threshold={thresh:.2f}): **{'Stress' if decision else 'No Stress'}**")
        if enable_shap:
            dfsh = shap_table_for_features(feats)
            if dfsh is not None:
                st.subheader("SHAP contributions")
                st.table(dfsh)
        if openai_key:
            payload = {'prob':prob,'label':label,'top_features':None,'feats':feats}
            out = call_openai_api(openai_key, payload)
            st.subheader("AI Explanation")
            st.write(out)

elif input_method == "Upload (image/video)":
    st.subheader("Upload a face image or a short face video (8-15s recommended)")
    uploaded = st.file_uploader("Choose file", type=["jpg","jpeg","png","mp4","mov","avi"])

    if uploaded is not None:
        # save temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.flush()
        fname = tmp.name

        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext in [".mp4", ".mov", ".avi"]:
            st.info("Processing video for rPPG & HRV (this can take a few seconds)...")
            try:
                sig, fps = extract_signal_from_video(fname, max_seconds=max_video_seconds)
                hr_mean, hr_series, rr = estimate_hr_rr_from_signal(sig, fps)
                if hr_mean is None:
                    st.error("Could not detect reliable pulse peaks. Try longer/clearer video with stable face.")
                else:
                    feats = compute_hrv_features_from_rr(rr, hr_series)
                    st.subheader("HRV Features")
                    st.json(feats)
                    prob, label, used = classify_features(feats)
                    decision, thresh = adaptive_decision(prob)
                    st.metric("Stress probability", f"{prob:.3f}")
                    st.write(f"Adaptive label (threshold={thresh:.2f}): **{'Stress' if decision else 'No Stress'}**")
                    if enable_shap:
                        dfsh = shap_table_for_features(feats)
                        if dfsh is not None:
                            st.subheader("SHAP contributions")
                            st.table(dfsh)
                    # LLM
                    payload = {'prob':prob,'label':label,'top_features':None,'feats':feats}
                    if openai_key:
                        with st.spinner("Calling OpenAI..."):
                            out = call_openai_api(openai_key, payload)
                            st.subheader("AI Explanation")
                            st.write(out)
                    else:
                        st.info("Provide OpenAI API key in sidebar to enable natural language explanations.")
            except Exception as e:
                st.error(f"Video processing error: {e}")
            finally:
                try: os.unlink(fname)
                except: pass

        else:
            # image file
            img = Image.open(fname)
            st.image(img, caption="Uploaded image", use_column_width=True)
            st.warning("Single images cannot provide HRV/rPPG. You may either upload a short video or use Webcam Record for real HRV.")
            st.markdown("### Demo prediction from image (surrogate features)")
            if st.button("Run demo prediction on this image"):
                # make surrogate features from image: mean green intensity and face size proxy
                try:
                    arr = np.array(img.convert("RGB"))
                    mean_g = float(np.mean(arr[:,:,1]))
                    h,w,_ = arr.shape
                    face_area_proxy = (h*w) / (640*480)
                    # construct plausible HRV surrogate values using heuristics
                    feats = {'mean_hr': 75 + (128-mean_g)/20.0, 'rmssd': 30.0 - (face_area_proxy*5.0), 'pnn50':np.nan, 'sd1':20.0 - face_area_proxy*2.0, 'sd2':40.0, 'lf_hf':np.nan, 'rr_mean':np.nan, 'rr_std':np.nan}
                except Exception:
                    feats = {'mean_hr':75,'rmssd':30,'pnn50':np.nan,'sd1':20,'sd2':40,'lf_hf':np.nan,'rr_mean':np.nan,'rr_std':np.nan}
                prob, label, used = classify_features(feats)
                decision, thresh = adaptive_decision(prob)
                st.metric("Stress probability (demo)", f"{prob:.3f}")
                st.write(f"Demo label: **{'Stress' if decision else 'No Stress'}**")
                if enable_shap:
                    dfsh = shap_table_for_features(feats)
                    if dfsh is not None:
                        st.subheader("SHAP (demo)")
                        st.table(dfsh)
                if openai_key:
                    out = call_openai_api(openai_key, {'prob':prob,'label':label,'top_features':None,'feats':feats})
                    st.subheader("AI Explanation")
                    st.write(out)
            try: os.unlink(fname)
            except: pass

elif input_method == "Webcam Record":
    st.subheader("Record a short video from your webcam (local). Keep face steady.")
    if st.button("Start Recording"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fname = tmp.name; tmp.close()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam. Make sure you're running locally and camera is available.")
        else:
            fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            ret, frame = cap.read()
            h,w = frame.shape[:2]
            out = cv2.VideoWriter(fname, fourcc, fps, (w,h))
            t0 = time.time()
            progress = st.progress(0)
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
            st.success("Recording saved, processing...")
            try:
                sig, fps_used = extract_signal_from_video(fname, max_seconds=record_seconds)
                hr_mean, hr_series, rr = estimate_hr_rr_from_signal(sig, fps_used)
                if hr_mean is None:
                    st.error("Could not detect peaks. Try again with better lighting and minimal motion.")
                else:
                    feats = compute_hrv_features_from_rr(rr, hr_series)
                    st.subheader("HRV Features")
                    st.json(feats)
                    prob, label, used = classify_features(feats)
                    decision, thresh = adaptive_decision(prob)
                    st.metric("Stress probability", f"{prob:.3f}")
                    st.write(f"Adaptive label (threshold={thresh:.2f}): **{'Stress' if decision else 'No Stress'}**")
                    if enable_shap:
                        dfsh = shap_table_for_features(feats)
                        if dfsh is not None:
                            st.subheader("SHAP contributions")
                            st.table(dfsh)
                    if openai_key:
                        with st.spinner("Calling OpenAI..."):
                            out = call_openai_api(openai_key, {'prob':prob,'label':label,'top_features':None,'feats':feats})
                            st.subheader("AI Explanation")
                            st.write(out)
            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                try: os.unlink(fname)
                except: pass

# Footer
st.write("---")
st.markdown("**Notes & limitations:** rPPG here is a simplified method (center ROI mean-green). For production-level rPPG use POS/CHROM + motion compensation + skin detection. This demo is for research/prototyping and not medical advice.")
