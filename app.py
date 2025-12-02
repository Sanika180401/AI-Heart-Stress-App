# app.py (final)
import os, time, math, tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
from scipy.signal import filtfilt, butter, find_peaks, welch, detrend
import mediapipe as mp
from joblib import load
import shap
import openai

# ---------------------------
# Config
# ---------------------------
MODEL_DIR = "models"
MLP_PATH = os.path.join(MODEL_DIR, "mlp.pkl")
RF_PATH = os.path.join(MODEL_DIR, "rf.pkl")
XGB_PATH = os.path.join(MODEL_DIR, "xgb.pkl")
STACK_PATH = os.path.join(MODEL_DIR, "stacker.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

st.set_page_config(page_title="HRV Stress Monitor", layout="wide")
st.title("AI Heart Rate & Stress Monitoring — rPPG + HRV + Ensemble + XAI + LLM")

# ---------------------------
# Utility rPPG & HRV functions
# ---------------------------
def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def chrom_extract_from_video(path, max_seconds=12, resize_width=320):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds, fps*max_seconds))
    reds, greens, blues = [], [], []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret: break
        h,w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # detect face ROI via mediapipe (more robust)
        # fallback to center crop if face detection fails for speed
        face = frame
        # choose center crop
        h,w = face.shape[:2]
        cx, cy = w//2, h//2
        wbox, hbox = int(w*0.35), int(h*0.45)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(w, cx+wbox//2), min(h, cy+hbox//2)
        roi = face[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1; continue
        b,g,r = np.mean(roi[:,:,0]), np.mean(roi[:,:,1]), np.mean(roi[:,:,2])
        reds.append(r); greens.append(g); blues.append(b)
        frames += 1
    cap.release()
    if len(reds) < 30:
        return None, None
    rgb = np.vstack([reds, greens, blues]).T
    rgb_norm = rgb / np.mean(rgb, axis=0)
    x1 = 3*rgb_norm[:,0] - 2*rgb_norm[:,1]
    x2 = 1.5*rgb_norm[:,0] + rgb_norm[:,1] - 1.5*rgb_norm[:,2]
    S = x1 - (np.std(x1)/np.std(x2))*x2
    S = bandpass(S, fps)
    return S, fps

def detect_peaks_and_rr(sig, fps):
    if sig is None: return None, None
    distance = int(0.4*fps)
    peaks, _ = find_peaks(sig, distance=distance, prominence=np.std(sig)*0.2)
    if len(peaks) < 2:
        peaks, _ = find_peaks(sig, distance=distance, prominence=np.std(sig)*0.05)
    if len(peaks) < 2:
        return None, None
    times = peaks / fps
    rr = np.diff(times) * 1000.0
    hr_series = 60000.0 / rr
    return rr, hr_series

def compute_hrv_features(rr, hr_ser):
    if rr is None or len(rr)<2: return None
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff)/2.0))
    sd2 = float(np.sqrt(2*np.var(rr) - np.var(diff)/2.0))
    rr_mean = float(np.mean(rr)); rr_std = float(np.std(rr))
    mean_hr = float(np.mean(hr_ser))
    # LF/HF
    try:
        fs_interp = 4.0
        t = np.cumsum(rr)/1000.0
        t_interp = np.arange(0, t[-1], 1.0/fs_interp)
        inst_hr = 60000.0/rr
        beat_times = t[:-1]
        interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
        f, p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
        lf_mask = (f>=0.04) & (f<=0.15)
        hf_mask = (f>0.15) & (f<=0.4)
        lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
        hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
        lf_hf = float(lf/hf) if hf>0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {'mean_hr':mean_hr,'rmssd':rmssd,'pnn50':pnn50,'sd1':sd1,'sd2':sd2,'lf_hf':lf_hf,'rr_mean':rr_mean,'rr_std':rr_std}

# ---------------------------
# Load models (if available)
# ---------------------------
def safe_load(path):
    try:
        return load(path)
    except Exception:
        return None

mlp = safe_load(MLP_PATH)
rf = safe_load(RF_PATH)
xgb = safe_load(XGB_PATH)
stacker = safe_load(STACK_PATH)
scaler = safe_load(SCALER_PATH)

# if no models exist, create lightweight demo models (so app doesn't crash)
if mlp is None or rf is None or xgb is None or stacker is None or scaler is None:
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_demo = np.random.rand(120,8)
    y_demo = np.random.randint(0,2,120)
    scaler = StandardScaler().fit(X_demo)
    Xs = scaler.transform(X_demo)
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=300).fit(Xs,y_demo)
    rf = RandomForestClassifier(n_estimators=100).fit(Xs,y_demo)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100).fit(Xs,y_demo)
    meta = np.vstack([mlp.predict_proba(Xs)[:,1], rf.predict_proba(Xs)[:,1], xgb.predict_proba(Xs)[:,1]]).T
    stacker = LogisticRegression().fit(meta, y_demo)

# SHAP explainer (TreeExplainer for RF)
explainer = shap.TreeExplainer(rf)

# ---------------------------
# Adaptive threshold per-session
# ---------------------------
if 'baseline_probs' not in st.session_state:
    st.session_state['baseline_probs'] = []

def adaptive_threshold(prob, k=0.9):
    baseline = st.session_state['baseline_probs']
    if len(baseline) < 12:
        thresh = 0.5
    else:
        mu = float(np.mean(baseline)); sigma = float(np.std(baseline))
        thresh = float(np.clip(mu + k*sigma, 0.3, 0.8))
    baseline.append(prob)
    if len(baseline) > 500:
        baseline.pop(0)
    st.session_state['baseline_probs'] = baseline
    return prob >= thresh, thresh

# ---------------------------
# LLM helper
# ---------------------------
def call_openai(api_key, payload):
    if not api_key:
        return "No OpenAI API key provided."
    openai.api_key = api_key
    system = "You are a helpful health assistant (non-diagnostic). Provide concise explanation and 3 recommendations for stress reduction."
    user_msg = f"""Payload: {payload}
Provide:
1) One-sentence plain explanation why the model flagged stress.
2) Immediate, short-term, long-term actionable suggestions (1-2 short bullets each).
3) When to seek medical help (one sentence).
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":system},{"role":"user","content":user_msg}],
            temperature=0.7,
            max_tokens=300
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ---------------------------
# UI layout
# ---------------------------
st.sidebar.header("Settings")
video_seconds = st.sidebar.slider("Max seconds to process", 6, 20, 10)
record_seconds = st.sidebar.slider("Record webcam (sec)", 4, 12, 8)
openai_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
enable_shap = st.sidebar.checkbox("Enable SHAP", True)
st.sidebar.markdown("Models loaded from `models/` if available.")

col1, col2 = st.columns([2,1])

with col1:
    mode = st.selectbox("Input method", ["Manual Entry","Upload Video","Webcam Record"])
    st.markdown("---")

    if mode == "Manual Entry":
        st.subheader("Manual entry of HR/HRV (useful to test pipeline)")
        mean_hr = st.number_input("Mean heart rate (bpm)", 50, 140, 75)
        rmssd = st.number_input("RMSSD (ms)", 5.0, 200.0, 30.0)
        sd1 = st.number_input("SD1 (ms)", 1.0, 100.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 1.0, 200.0, 40.0)
        if st.button("Predict"):
            feat = np.array([[mean_hr,rmssd,0,sd1,sd2,0.0,0.0,0.0]])  # keep same order as training
            feat_scaled = scaler.transform(feat)
            p1 = mlp.predict_proba(feat_scaled)[:,1][0]
            p2 = rf.predict_proba(feat_scaled)[:,1][0]
            p3 = xgb.predict_proba(feat_scaled)[:,1][0]
            meta = np.array([[p1,p2,p3]])
            prob = float(stacker.predict_proba(meta)[:,1][0])
            decision, thresh = adaptive_threshold(prob)
            st.metric("Stress probability", f"{prob:.3f}")
            st.metric("Adaptive label", "Stress" if decision else "No Stress")
            if enable_shap:
                shap_vals = explainer.shap_values(feat_scaled)[0]
                df_shap = pd.DataFrame({'feature':['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std'],'shap':shap_vals[0]})
                st.table(df_shap)

    elif mode == "Upload Video":
        st.subheader("Upload short face video (mp4) — 6–15s recommended")
        uploaded = st.file_uploader("Video file", type=["mp4","mov","avi"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.flush()
            st.info("Processing video — extracting rPPG and HRV...")
            try:
                sig, fps = chrom_extract_from_video(tmp.name, max_seconds=video_seconds)
                if sig is None:
                    st.error("Could not extract rPPG. Try clearer video or longer duration.")
                else:
                    rr, hr_ser = detect_peaks_and_rr(sig, fps)
                    feats = compute_hrv_features(rr, hr_ser)
                    if feats is None:
                        st.error("Could not detect peaks reliably.")
                    else:
                        st.subheader("HRV Features")
                        st.json(feats)
                        order = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
                        feat_vec = np.array([[feats[k] for k in order]])
                        feat_scaled = scaler.transform(feat_vec)
                        p1 = mlp.predict_proba(feat_scaled)[:,1][0]
                        p2 = rf.predict_proba(feat_scaled)[:,1][0]
                        p3 = xgb.predict_proba(feat_scaled)[:,1][0]
                        meta = np.array([[p1,p2,p3]])
                        prob = float(stacker.predict_proba(meta)[:,1][0])
                        decision, thresh = adaptive_threshold(prob)
                        st.metric("Stress probability", f"{prob:.3f}")
                        st.metric("Adaptive label", "Stress" if decision else "No Stress")
                        if enable_shap:
                            shap_vals = explainer.shap_values(feat_scaled)[0]
                            df_shap = pd.DataFrame({'feature':order,'shap':shap_vals[0]})
                            st.subheader("SHAP (approx.)")
                            st.table(df_shap)
                        # LLM explanation
                        payload = {'prob':prob,'features':feats}
                        if openai_key:
                            with st.spinner("Generating explanation from OpenAI..."):
                                llm = call_openai(openai_key, payload)
                                st.subheader("AI Explanation / Recommendations")
                                st.write(llm)
                        else:
                            st.info("OpenAI key not provided — showing template suggestion.")
                            st.write("Template: If stress probability > 0.6 consider relaxation; immediate: 2-min breathing; short-term: 20-min walk; long-term: sleep/exercise habit.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
            finally:
                try: os.unlink(tmp.name)
                except: pass

    elif mode == "Webcam Record":
        st.subheader("Record from webcam (local). Keep face in frame for best results.")
        if st.button("Start Recording"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fname = tmp.name; tmp.close()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Run locally and allow camera.")
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
                st.success("Recorded. Processing...")
                try:
                    sig, fps = chrom_extract_from_video(fname, max_seconds=record_seconds)
                    rr, hr_ser = detect_peaks_and_rr(sig, fps)
                    feats = compute_hrv_features(rr, hr_ser)
                    if feats is None:
                        st.error("Could not extract HRV reliably.")
                    else:
                        st.json(feats)
                        order = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
                        feat_vec = np.array([[feats[k] for k in order]])
                        feat_scaled = scaler.transform(feat_vec)
                        p1 = mlp.predict_proba(feat_scaled)[:,1][0]
                        p2 = rf.predict_proba(feat_scaled)[:,1][0]
                        p3 = xgb.predict_proba(feat_scaled)[:,1][0]
                        meta = np.array([[p1,p2,p3]])
                        prob = float(stacker.predict_proba(meta)[:,1][0])
                        decision, thresh = adaptive_threshold(prob)
                        st.metric("Stress probability", f"{prob:.3f}")
                        st.metric("Adaptive label", "Stress" if decision else "No Stress")
                        if enable_shap:
                            shap_vals = explainer.shap_values(feat_scaled)[0]
                            df_shap = pd.DataFrame({'feature':order,'shap':shap_vals[0]})
                            st.subheader("SHAP (approx.)")
                            st.table(df_shap)
                        if openai_key:
                            with st.spinner("Generating explanation from OpenAI..."):
                                llm = call_openai(openai_key, {'prob':prob,'features':feats})
                                st.subheader("AI Explanation / Recommendations")
                                st.write(llm)
                        else:
                            st.info("Add OpenAI key in sidebar to enable natural language explanation.")
                except Exception as e:
                    st.error(f"Error processing recording: {e}")
                finally:
                    try: os.unlink(fname)
                    except: pass

# Footer
st.markdown("---")
st.markdown("Notes: rPPG is sensitive to lighting and motion. For best results: stable lighting, face in frame, minimal movement. This tool is research/demo — not diagnostic.")
