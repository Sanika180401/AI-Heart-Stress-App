import streamlit as st
st.set_page_config(page_title="HR & Stress Monitor", layout="wide")

import os
import time
import tempfile
import math
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io
import pickle
import base64

# Signal processing & ML
import cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Try to import optional libs, handle if not installed
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, Dropout
except Exception:
    tf = None

try:
    import shap
except Exception:
    shap = None

try:
    import openai
except Exception:
    openai = None

# ---------------------------
# Utilities: rPPG extraction
# ---------------------------

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

def get_face_roi_bbox(frame):
    """Return bounding box (x1,y1,x2,y2) of the most confident face detection or None."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(image_rgb)
    h, w = frame.shape[:2]
    if results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(0, bbox.xmin * w))
        y1 = int(max(0, bbox.ymin * h))
        x2 = int(min(w - 1, (bbox.xmin + bbox.width) * w))
        y2 = int(min(h - 1, (bbox.ymin + bbox.height) * h))
        # Expand box slightly
        pad_w = int(0.05 * w)
        pad_h = int(0.05 * h)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w - 1, x2 + pad_w)
        y2 = min(h - 1, y2 + pad_h)
        return (x1, y1, x2, y2)
    return None

def extract_raw_signal_from_video(path, max_seconds=20, resize_width=320):
    """
    Read video frames, detect face ROI and return a 1D raw PPG-like signal using mean green channel in ROI.
    Limits to max_seconds for speed.
    Returns: signal (np.array), fps (float)
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open video file / webcam stream.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(total_frames, fps * max_seconds)) if total_frames>0 else int(fps * max_seconds)

    raw = []
    last_roi = None
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        # optional resize for speed
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        roi = get_face_roi_bbox(frame)
        if roi is None:
            # if no face found, use previous ROI if any
            if last_roi is not None:
                x1,y1,x2,y2 = last_roi
            else:
                frames += 1
                continue
        else:
            x1,y1,x2,y2 = roi
            last_roi = roi
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            frames += 1
            continue
        # mean green channel as primitive rPPG surrogate
        mean_g = np.mean(face[:, :, 1])
        raw.append(mean_g)
        frames += 1

    cap.release()
    raw = np.array(raw)
    if raw.size == 0:
        raise RuntimeError("No valid frames / face ROI detected in video.")
    return raw, fps

def bandpass_filter(signal, fps, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fps
    lown = low / nyq
    highn = high / nyq
    b, a = butter(order, [lown, highn], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered

def get_heart_rate_and_rr(filtered_signal, fps):
    """
    Estimate instantaneous heart rate & RR intervals (ms) from filtered signal.
    Using peak detection on filtered signal.
    Returns:
        hr_mean (bpm),
        hr_series (np.array),
        rr_intervals_ms (np.array)
    """
    # find peaks
    distance = int(0.4 * fps)  # at least 0.4s between peaks (150 bpm upper bound)
    peaks, _ = find_peaks(filtered_signal, distance=distance, prominence=np.std(filtered_signal)*0.3)
    if len(peaks) < 2:
        # fallback: try lower prominence
        peaks, _ = find_peaks(filtered_signal, distance=distance, prominence=np.std(filtered_signal)*0.1)
    if len(peaks) < 2:
        return None, None, None
    # convert to times
    times = peaks / fps
    rr = np.diff(times) * 1000.0  # ms
    hr_series = 60000.0 / rr  # bpm per interval
    hr_mean = np.mean(hr_series)
    return hr_mean, hr_series, rr

# ---------------------------
# HRV feature calculations
# ---------------------------

def sd1_sd2(rr_ms):
    """Compute Poincare SD1 and SD2 (rr_ms is array of RR intervals in ms)"""
    if rr_ms is None or len(rr_ms) < 2:
        return np.nan, np.nan
    diff = np.diff(rr_ms)
    sd1 = math.sqrt(np.var(diff) / 2.0)
    sd2 = math.sqrt(2 * np.var(rr_ms) - np.var(diff) / 2.0)
    return float(sd1), float(sd2)

def rmssd(rr_ms):
    if rr_ms is None or len(rr_ms) < 2:
        return np.nan
    diff = np.diff(rr_ms)
    return float(np.sqrt(np.mean(diff**2)))

def pnn50(rr_ms):
    if rr_ms is None or len(rr_ms) < 2:
        return np.nan
    diff = np.abs(np.diff(rr_ms))
    return float(np.sum(diff > 50) / len(diff) * 100.0)

def lf_hf_ratio(rr_ms, fs=4.0):
    """
    Estimate LF/HF via Welch on interpolated RR tachogram.
    Interpolate RR (ms) to evenly spaced signal at 'fs' Hz.
    """
    if rr_ms is None or len(rr_ms) < 4:
        return np.nan
    # construct cumulative time of beats
    times = np.cumsum(rr_ms) / 1000.0  # seconds, relative
    # create evenly spaced time series
    t_interp = np.arange(0, times[-1], 1.0/fs)
    # linear interp for RR tachogram (instantaneous HR)
    inst_hr = 60000.0 / rr_ms
    # map inst_hr samples at beat times (times[:-1]) to t_interp — use first len(inst_hr) times
    beat_times = times[:-1]  # alignment
    if len(beat_times) < 4:
        return np.nan
    try:
        interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
    except Exception:
        return np.nan
    f, p = welch(detrend(interp), fs=fs, nperseg=min(256, len(interp)))
    # LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz
    lf_mask = (f >= 0.04) & (f <= 0.15)
    hf_mask = (f > 0.15) & (f <= 0.4)
    lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
    hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
    if hf <= 0:
        return np.nan
    return float(lf / hf)

def compute_hrv_features(rr_ms, hr_series=None):
    features = {}
    features['mean_hr'] = float(np.nanmean(hr_series)) if hr_series is not None else np.nan
    features['rmssd'] = rmssd(rr_ms)
    features['pnn50'] = pnn50(rr_ms)
    sd1, sd2 = sd1_sd2(rr_ms)
    features['sd1'] = sd1
    features['sd2'] = sd2
    features['lf_hf'] = lf_hf_ratio(rr_ms)
    # additional simple statistics
    if rr_ms is not None and len(rr_ms) > 0:
        features['rr_mean'] = float(np.nanmean(rr_ms))
        features['rr_std'] = float(np.nanstd(rr_ms))
    else:
        features['rr_mean'] = np.nan
        features['rr_std'] = np.nan
    return features

# ---------------------------
# Classification / Ensemble
# ---------------------------

MODEL_PATH = "models/ensemble_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_models_if_present():
    ensemble = None
    scaler = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                ensemble = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load ensemble_model.pkl: {e}")
            ensemble = None
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load scaler.pkl: {e}")
            scaler = None
    return ensemble, scaler

def heuristic_stress_probability(features):
    """
    A robust heuristic fallback: combine some HRV metrics into a stress 'probability'.
    This is not a trained model — it's a reasonable rule-of-thumb to let the app run.
    High LF/HF, low RMSSD / low SD1 -> higher stress.
    """
    # collect values with safe defaults
    rm = features.get('rmssd', np.nan)
    sd1 = features.get('sd1', np.nan)
    lf_hf = features.get('lf_hf', np.nan)
    mean_hr = features.get('mean_hr', np.nan)

    # Normalize by plausible ranges (simple)
    score = 0.0
    if not math.isnan(lf_hf):
        # LF/HF typical range 0.5-4, higher means sympathetic dominance
        score += (np.tanh((lf_hf - 1.0) / 2.0) + 1) * 0.4  # scaled contribution
    if not math.isnan(rm):
        # RMSSD lower -> stress (typical restful RMSSD 30-50 ms)
        score += (1.0 - np.tanh((rm - 30.0) / 30.0)) * 0.35
    if not math.isnan(sd1):
        score += (1.0 - np.tanh((sd1 - 20.0) / 20.0)) * 0.15
    if not math.isnan(mean_hr):
        # if mean_hr is high, small contribution
        score += (np.tanh((mean_hr - 70.0) / 20.0) + 1) * 0.05
    # map to 0-1
    prob = 1.0 / (1.0 + np.exp(-score + 0.5))  # sigmoid mapping
    return float(np.clip(prob, 0.0, 1.0))

def classify(features, ensemble=None, scaler=None):
    """
    Return (probability, label, used_method)
    If ensemble is present (sklearn pipeline), use it; else use heuristic.
    """
    feat_names = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
    X = np.array([features.get(k, np.nan) for k in feat_names]).reshape(1, -1)
    used = "heuristic"
    prob = heuristic_stress_probability(features)
    label = "Stress" if prob >= 0.5 else "No Stress"
    if ensemble is not None:
        try:
            # handle scaler
            if scaler is not None:
                Xs = scaler.transform(X)
            else:
                Xs = X
            if hasattr(ensemble, "predict_proba"):
                prob = float(ensemble.predict_proba(Xs)[:,1][0])
            else:
                prob = float(ensemble.predict(Xs))
            label = "Stress" if prob >= 0.5 else "No Stress"
            used = "ensemble_model"
        except Exception as e:
            st.warning(f"Ensemble model prediction failed, using heuristic. ({e})")
            prob = heuristic_stress_probability(features)
            label = "Stress" if prob >= 0.5 else "No Stress"
            used = "heuristic"
    return prob, label, used

# ---------------------------
# SHAP explainability
# ---------------------------

def try_shap_explain(ensemble, scaler, features_df):
    """
    Returns a matplotlib figure with SHAP summary / force-plot-like textual fallback.
    If shap or model not compatible, returns text explanation.
    """
    if shap is None:
        return "SHAP not installed; install shap to view detailed explanations."
    if ensemble is None:
        return "No trained model found; SHAP requires a trained model."
    try:
        # If ensemble is tree-based or sklearn pipeline with tree, use TreeExplainer
        explainer = None
        try:
            explainer = shap.Explainer(ensemble, masker=shap.maskers.Independent(features_df))
        except Exception:
            try:
                explainer = shap.KernelExplainer(lambda x: ensemble.predict_proba(x)[:,1], shap.sample(features_df, 50))
            except Exception:
                explainer = None
        if explainer is None:
            return "Could not init SHAP explainer for this model."
        shap_vals = explainer(features_df)
        # simple summary plot
        fig = plt.figure(figsize=(6,4))
        shap.plots.bar(shap_vals, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"SHAP explanation error: {e}"

# ---------------------------
# Adaptive thresholding
# ---------------------------

def adaptive_decision(prob, k=1.0, min_thresh=0.35, max_thresh=0.75):
    """
    Maintain a per-session baseline of probs in session_state['baseline_probs'].
    Compute threshold = mean + k*std clipped to [min_thresh, max_thresh].
    Return (decision_bool, threshold_used).
    """
    if 'baseline_probs' not in st.session_state:
        st.session_state['baseline_probs'] = []
    baseline = st.session_state['baseline_probs']
    if len(baseline) < 10:
        thresh = 0.5
    else:
        mu = float(np.mean(baseline))
        sigma = float(np.std(baseline))
        thresh = float(np.clip(mu + k * sigma, min_thresh, max_thresh))
    decision = prob >= thresh
    # append prob to baseline with forgetting factor to keep recent baseline
    baseline.append(prob)
    if len(baseline) > 500:
        baseline.pop(0)
    st.session_state['baseline_probs'] = baseline
    return decision, float(thresh)

# ---------------------------
# LLM (OpenAI) explanation
# ---------------------------

def gpt_explain(api_key, payload):
    """
    Send payload (dict) to OpenAI chat completion (gpt-4-like or gpt-3.5) and return text.
    If openai not installed or key not provided, return template only.
    """
    if openai is None:
        return "OpenAI library not installed. To enable GPT explanations install 'openai' and provide API key."
    if not api_key:
        return "No OpenAI API key provided. Enter your key in the 'OpenAI API Key' field to enable GPT explanations."

    openai.api_key = api_key
    system = (
        "You are a helpful, non-diagnostic health assistant. Provide short, practical, and safe stress management advice. "
        "Use the provided model outputs and SHAP top features. Do NOT give medical diagnoses; advise to consult a professional when appropriate."
    )
    user_msg = f"""Model prediction payload:
Probability: {payload.get('prob'):.3f}
Label: {payload.get('label')}
Top positive feature contributions: {payload.get('top_pos')}
Top negative feature contributions: {payload.get('top_neg')}
Latest HR: {payload.get('mean_hr')}
Recent HRV: RMSSD={payload.get('rmssd')}, SD1={payload.get('sd1')}, SD2={payload.get('sd2')}, LF/HF={payload.get('lf_hf')}

Please:
1) Give a concise 2-sentence explanation (why model flagged stress).
2) Give 3 short actionable recommendations: immediate (1-2 min), short-term (day), long-term (weeks).
3) Provide a sentence indicating when to seek medical help.
"""
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4" in openai.models.list() else "gpt-3.5-turbo",
            messages=[{"role":"system","content":system},{"role":"user","content":user_msg}],
            temperature=0.7,
            max_tokens=300
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"OpenAI API call failed: {e}"

# ---------------------------
# Visualization helpers
# ---------------------------

def plot_signal_and_peaks(signal, fps, peaks=None, title="rPPG signal"):
    fig, ax = plt.subplots(figsize=(8,2.4))
    t = np.arange(len(signal)) / float(fps)
    ax.plot(t, signal, label='signal')
    if peaks is not None and len(peaks)>0:
        ax.plot(peaks / float(fps), signal[peaks], 'ro', label='peaks')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc='upper right')
    plt.tight_layout()
    return fig

def display_features_table(features):
    df = pd.DataFrame([features])
    return df

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("AI-Based Real-Time Heart Rate & Stress Monitoring (rPPG + HRV + Ensemble + XAI + LLM)")
st.markdown("**Three input options:** Manual Entry, Upload (image or short video), Webcam Capture (local).")

# Load models if present
ensemble_model, scaler_model = load_models_if_present()
if ensemble_model is not None:
    st.success("Found trained ensemble model in models/ensemble_model.pkl — will use for prediction if compatible.")
else:
    st.info("No trained ensemble model found in models/. The app will use a heuristic fallback classifier. To use trained model, place a pickled sklearn pipeline at models/ensemble_model.pkl and optional scaler at models/scaler.pkl.")

col1, col2 = st.columns([2,1])

with col2:
    st.subheader("Settings")
    video_seconds = st.number_input("Max video seconds to process (local/upload)", min_value=5, max_value=30, value=12, step=1)
    recording_seconds = st.number_input("Webcam record seconds (local)", min_value=4, max_value=20, value=10, step=1)
    enable_shap = st.checkbox("Enable SHAP explainability (needs shap & model)", value=True)
    openai_key_input = st.text_input("OpenAI API Key (optional)", type="password")
    show_debug = st.checkbox("Show debug logs", value=False)
    st.markdown("---")
    st.markdown("**Note:** Single still images cannot yield HRV. Upload a short video or use webcam capture for temporal PPG signal.")

with col1:
    option = st.selectbox("Select Input Method", ("Manual Entry", "Upload Image/Video", "Webcam Capture (local)"))

# Session placeholders
if 'last_features' not in st.session_state:
    st.session_state['last_features'] = None
if 'last_prob' not in st.session_state:
    st.session_state['last_prob'] = None

# ---------------------------
# Manual Entry
# ---------------------------
if option == "Manual Entry":
    st.header("Manual Entry")
    hr = st.number_input("Enter Heart Rate (BPM)", min_value=30, max_value=200, value=75, step=1)
    rmssd_val = st.number_input("Enter RMSSD (ms) if known (optional)", value=30.0, step=0.1)
    if st.button("Compute & Predict"):
        features = {
            'mean_hr': float(hr),
            'rmssd': float(rmssd_val),
            'pnn50': np.nan,
            'sd1': np.nan,
            'sd2': np.nan,
            'lf_hf': np.nan,
            'rr_mean': np.nan,
            'rr_std': np.nan
        }
        prob, label, used = classify(features, ensemble_model, scaler_model)
        decision, thresh = adaptive_decision(prob)
        st.write(f"Prediction method: **{used}**")
        st.metric("Stress Probability", f"{prob:.3f}")
        st.metric("Stress Label (adaptive)", f"{'Stress' if decision else 'No Stress'} (threshold={thresh:.2f})")
        st.table(display_features_table(features))

# ---------------------------
# Upload Image/Video
# ---------------------------
elif option == "Upload Image/Video":
    st.header("Upload a face image *or* a short face video (recommended)")
    uploaded = st.file_uploader("Upload file (jpg/png/mp4/mov)", type=["jpg","jpeg","png","mp4","mov"], accept_multiple_files=False)
    if uploaded is not None:
        # save to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tfile.write(uploaded.getvalue())
        tfile.flush()
        fname = tfile.name
        ext = os.path.splitext(uploaded.name)[1].lower()
        try:
            if ext in ['.mp4','.mov']:
                st.info("Processing uploaded video — detecting face ROI and extracting rPPG...")
                raw, fps = extract_raw_signal_from_video(fname, max_seconds=video_seconds)
                filtered = bandpass_filter(raw, fps)
                hr_mean, hr_series, rr = get_heart_rate_and_rr(filtered, fps)
                if hr_mean is None:
                    st.error("Could not detect reliable pulse peaks. Try a longer/clearer video with visible face and minimal motion.")
                else:
                    features = compute_hrv_features(rr, hr_series)
                    st.session_state['last_features'] = features
                    prob, label, used = classify(features, ensemble_model, scaler_model)
                    decision, thresh = adaptive_decision(prob)
                    st.subheader("Results")
                    st.metric("Estimated Mean HR (bpm)", f"{features.get('mean_hr'):.1f}")
                    st.metric("Stress Probability", f"{prob:.3f}")
                    st.metric("Adaptive Stress Label", f"{'Stress' if decision else 'No Stress'} (threshold={thresh:.2f})")
                    # show rPPG plot
                    peaks, _ = find_peaks(filtered, distance=int(0.4*fps), prominence=np.std(filtered)*0.15)
                    fig_sig = plot_signal_and_peaks(filtered, fps, peaks, title="Extracted rPPG-like signal (mean green in face ROI)")
                    st.pyplot(fig_sig)
                    st.subheader("HRV Features")
                    st.table(display_features_table(features))

                    # SHAP
                    if enable_shap and ensemble_model is not None:
                        st.subheader("SHAP Explainability")
                        feat_names = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
                        feat_df = pd.DataFrame([ [features.get(k, np.nan) for k in feat_names] ], columns=feat_names)
                        shap_out = try_shap_explain(ensemble_model, scaler_model, feat_df)
                        if isinstance(shap_out, plt.Figure):
                            st.pyplot(shap_out)
                        else:
                            st.write(shap_out)

                    # LLM
                    st.subheader("LLM Explanation & Recommendations")
                    top_pos = []
                    top_neg = []
                    # try to craft simple top contributions from features (fallback to heuristics)
                    if not math.isnan(features.get('lf_hf', np.nan)) and features.get('lf_hf')>1.2:
                        top_pos.append(("LF/HF", round(features.get('lf_hf'),2)))
                    if not math.isnan(features.get('rmssd', np.nan)) and features.get('rmssd')<30:
                        top_pos.append(("RMSSD low", round(features.get('rmssd'),2)))
                    if not math.isnan(features.get('sd1', np.nan)) and features.get('sd1')<20:
                        top_pos.append(("SD1 low", round(features.get('sd1'),2)))
                    payload = {
                        'prob': prob,
                        'label': label,
                        'top_pos': top_pos,
                        'top_neg': top_neg,
                        'mean_hr': features.get('mean_hr'),
                        'rmssd': features.get('rmssd'),
                        'sd1': features.get('sd1'),
                        'sd2': features.get('sd2'),
                        'lf_hf': features.get('lf_hf')
                    }
                    if openai_key_input:
                        with st.spinner("Calling OpenAI to generate explanation..."):
                            llm_text = gpt_explain(openai_key_input, payload)
                            st.write(llm_text)
                    else:
                        st.info("OpenAI key not provided — showing template explanation.")
                        st.write("Template explanation:")
                        st.write(f"The model estimated stress probability {prob:.2f}. Top contributors include: {top_pos if top_pos else 'none strongly observed'}. Immediate recommendation: try 1-2 minutes of paced breathing. If symptoms persist, consult a medical professional.")

            else:
                # image file: show the image and explain temporal need
                img = Image.open(fname)
                st.image(img, caption="Uploaded Image")
                st.warning("A single image cannot produce HRV or rPPG. Upload a short video (8-20s) or use 'Webcam Capture (local)' to record a short video for HRV extraction.")
        except Exception as e:
            st.error(f"Processing error: {e}")
        finally:
            try:
                os.unlink(tfile.name)
            except Exception:
                pass

# ---------------------------
# Webcam Capture (local)
# ---------------------------
elif option == "Webcam Capture (local)":
    st.header("Webcam Capture (Local only — uses OpenCV to record short video)")
    st.write("This option records a short video from your local webcam (works when running locally / not in remote Streamlit hosting).")
    if st.button("Record from Webcam"):
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fname = tmp_video.name
        tmp_video.close()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access local webcam. Ensure you run this app locally and your webcam is available.")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to read from webcam.")
            else:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(fname, fourcc, fps, (w, h))
                start = time.time()
                st.info(f"Recording {recording_seconds} seconds. Keep face steady and in frame.")
                progress = st.progress(0)
                frames_recorded = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frames_recorded += 1
                    elapsed = time.time() - start
                    progress.progress(min(100, int((elapsed/recording_seconds)*100)))
                    if elapsed >= recording_seconds:
                        break
                out.release()
                cap.release()
                st.success(f"Saved recording ({frames_recorded} frames). Processing...")
                try:
                    raw, fps_used = extract_raw_signal_from_video(fname, max_seconds=video_seconds)
                    filtered = bandpass_filter(raw, fps_used)
                    hr_mean, hr_series, rr = get_heart_rate_and_rr(filtered, fps_used)
                    if hr_mean is None:
                        st.error("Could not detect reliable pulse peaks. Try again with better lighting and minimal motion.")
                    else:
                        features = compute_hrv_features(rr, hr_series)
                        st.session_state['last_features'] = features
                        prob, label, used = classify(features, ensemble_model, scaler_model)
                        decision, thresh = adaptive_decision(prob)
                        st.subheader("Results")
                        st.metric("Estimated Mean HR (bpm)", f"{features.get('mean_hr'):.1f}")
                        st.metric("Stress Probability", f"{prob:.3f}")
                        st.metric("Adaptive Stress Label", f"{'Stress' if decision else 'No Stress'} (threshold={thresh:.2f})")
                        peaks, _ = find_peaks(filtered, distance=int(0.4*fps_used), prominence=np.std(filtered)*0.15)
                        fig_sig = plot_signal_and_peaks(filtered, fps_used, peaks, title="Extracted rPPG-like signal (mean green in face ROI)")
                        st.pyplot(fig_sig)
                        st.subheader("HRV Features")
                        st.table(display_features_table(features))
                        # SHAP section
                        if enable_shap and ensemble_model is not None:
                            st.subheader("SHAP Explainability")
                            feat_names = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
                            feat_df = pd.DataFrame([ [features.get(k, np.nan) for k in feat_names] ], columns=feat_names)
                            shap_out = try_shap_explain(ensemble_model, scaler_model, feat_df)
                            if isinstance(shap_out, plt.Figure):
                                st.pyplot(shap_out)
                            else:
                                st.write(shap_out)
                        # LLM
                        st.subheader("LLM Explanation & Recommendations")
                        payload = {
                            'prob': prob,
                            'label': label,
                            'top_pos': [],
                            'top_neg': [],
                            'mean_hr': features.get('mean_hr'),
                            'rmssd': features.get('rmssd'),
                            'sd1': features.get('sd1'),
                            'sd2': features.get('sd2'),
                            'lf_hf': features.get('lf_hf')
                        }
                        if openai_key_input:
                            with st.spinner("Calling OpenAI to generate explanation..."):
                                llm_text = gpt_explain(openai_key_input, payload)
                                st.write(llm_text)
                        else:
                            st.info("OpenAI key not provided — showing template explanation.")
                            st.write("Template explanation: The model estimated stress probability ... (see Upload option for example).")
                except Exception as e:
                    st.error(f"Error during processing: {e}")
                finally:
                    try:
                        os.unlink(fname)
                    except Exception:
                        pass

# ---------------------------
# Footer / Help
# ---------------------------

st.markdown("---")
st.markdown("### How this works (short)")
st.markdown(
    """
1. We extract mean green-channel values from a face ROI across frames (primitive rPPG).
2. Band-pass filter the signal for heart-rate band (0.7–4 Hz). Detect peaks to estimate inter-beat intervals (RR).
3. Compute HRV features including RMSSD, pNN50, SD1, SD2, LF/HF.
4. Use an ensemble model (if provided) or a heuristic function to estimate stress probability.
5. Explain predictions with SHAP (if model present) and optionally use GPT to generate human-friendly explanations and personalized advice.
"""
)

st.markdown("### Files / Models")
st.write("If you have trained models, place them in `models/` as `ensemble_model.pkl` (sklearn pipeline or model supporting predict_proba) and `scaler.pkl` for feature scaling if used.")

st.markdown("### Limitations")
st.write(
    """
- This rPPG method is a simplified approach — production-grade rPPG pipelines include advanced detrending, chrominance-based extraction, face tracking, and motion artifact removal.
- Results depend on camera quality, lighting, motion, and skin tone. Use still posture, stable lighting and a clear face view for best results.
- The heuristic classifier is a fallback and not a medical diagnostic. For a production-grade model, train the ensemble on labelled HRV/stress datasets (WESAD, etc.) and save models into `models/`.
- OpenAI GPT usage requires your API key and may incur costs.
"""
)
