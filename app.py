# app.py -- AI Heart & Stress Monitoring (MediaPipe face features, HRV, Ensemble)
import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- must be first Streamlit command
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")

# --- ML / DL imports
from tensorflow.keras.models import load_model
from sklearn.exceptions import NotFittedError
import shap

# --- MediaPipe face mesh
import cv2
import mediapipe as mp

# -------------- Paths & safe loader ----------------
BASE_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_PATH, "models")

def safe_path(fname):
    p = os.path.join(MODELS_PATH, fname)
    if not os.path.exists(p):
        st.error(f"Missing file: {p}")
        raise FileNotFoundError(p)
    return p

@st.cache_resource
def load_all_models():
    """Load preprocess and models if present. Return tuple (pre, rf, xgb, mlp, thr)."""
    pre = None; rf = None; xgb = None; mlp = None; thr = None
    # preprocess
    try:
        pre = joblib.load(safe_path("preprocess.joblib"))
    except Exception as e:
        st.warning(f"preprocess.joblib not loaded: {e}")
    # RF
    try:
        rf = joblib.load(safe_path("rf_hrv.joblib"))
    except Exception as e:
        st.warning(f"rf_hrv.joblib not loaded: {e}")
    # XGB
    try:
        xgb = joblib.load(safe_path("xgb_hrv.joblib"))
    except Exception as e:
        st.info("xgboost model not found or not loaded.")
        xgb = None
    # MLP (keras model)
    try:
        # try both .h5/.keras/.keras format names commonly used
        for candidate in ["mlp_hrv.h5","mlp_hrv_v2.h5","mlp_hrv_clean.h5","mlp_hrv_clean_fixed.keras","mlp_hrv_clean.keras","mlp_hrv_fixed.keras","mlp_hrv_fixed.h5"]:
            try:
                p = os.path.join(MODELS_PATH, candidate)
                if os.path.exists(p):
                    mlp = load_model(p, compile=False)
                    break
            except Exception:
                continue
    except Exception as e:
        st.warning(f"Could not load MLP model: {e}")
        mlp = None
    # thresholds
    try:
        thr = np.load(safe_path("adaptive_thresholds.npy"), allow_pickle=True).item()
    except Exception:
        thr = {"low":0.4,"high":0.7}
    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all_models()

# ---------- Utility: face feature extraction using MediaPipe Face Mesh ----------
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                          max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5)

def _euclid(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def extract_face_features_from_bgr(img_bgr):
    """Return (mouth_ar, eye_ar, brow_gap) or None if no face."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = mp_face.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark

    h, w = img_bgr.shape[:2]
    # helper to convert landmark idx to (x,y)
    def xy(i):
        return (int(lm[i].x * w), int(lm[i].y * h))

    # Mouth landmarks (approx): upper lip 13, lower lip 14; corners 61 and 291 (face mesh mapping)
    # NOTE: Mediapipe indexes: use commonly-known indices; these are stable for face_mesh.
    try:
        top_lip = xy(13)
        bottom_lip = xy(14)
        left_corner = xy(61)
        right_corner = xy(291)
        mouth_vert = _euclid(top_lip, bottom_lip)
        mouth_horiz = _euclid(left_corner, right_corner) + 1e-6
        mouth_ar = mouth_vert / mouth_horiz
    except Exception:
        mouth_ar = 0.0

    # Eyes: use right eye (33, 159 top/bottom) / left eye (362, 386); compute average EAR-like
    try:
        # right eye
        r_top = xy(159); r_bottom = xy(145); r_left = xy(33); r_right = xy(133)
        r_vert = (_euclid(r_top, r_bottom) + 1e-6)
        r_horiz = _euclid(r_left, r_right) + 1e-6
        r_ear = r_vert / r_horiz
        # left eye
        l_top = xy(386); l_bottom = xy(374); l_left = xy(362); l_right = xy(263)
        l_vert = (_euclid(l_top, l_bottom) + 1e-6)
        l_horiz = _euclid(l_left, l_right) + 1e-6
        l_ear = l_vert / l_horiz
        eye_ar = (r_ear + l_ear) / 2.0
    except Exception:
        eye_ar = 0.0

    # Brow-eye gap (proxy for tension): distance from mid-brow to eye top (use indices)
    try:
        brow = xy(10)   # forehead/brow area
        eye_top = xy(159)
        brow_gap = _euclid(brow, eye_top)
        brow_gap = brow_gap / (h+1e-6)  # normalize by height
    except Exception:
        brow_gap = 0.0

    return float(mouth_ar), float(eye_ar), float(brow_gap)

# --------------- Prediction / input handling ----------------
def build_input_row(hrv_values, face_agg, scaler):
    """
    hrv_values: dict for HRV/physio (ordered keys assumed)
    face_agg: dict with keys mouth_ar_mean, eye_ar_mean, brow_gap_mean (may be None)
    scaler: pre['scaler'] object or None
    Returns numpy array shaped (1, n_features) matching scaler.n_features_in_ if available
    """
    # default ordering used in training earlier in this project:
    base_order = ['mean_hr','sdnn','rmssd','pnn50','lf','hf','sd1','sd2','temp','eda']
    face_order = ['mouth_ar_mean','eye_ar_mean','brow_gap_mean']
    row = []
    for k in base_order:
        row.append(float(hrv_values.get(k, 0.0)))
    # append face features if available
    if face_agg:
        for k in face_order:
            row.append(float(face_agg.get(k, 0.0)))
    else:
        # pad zeros for face features
        row.extend([0.0,0.0,0.0])

    arr = np.array([row], dtype=float)

    # If scaler present, adapt to scaler expected number of features
    if scaler is not None:
        expected = getattr(scaler, "n_features_in_", None)
        if expected is not None:
            cur = arr.shape[1]
            if cur < expected:
                # pad with zeros
                pad = np.zeros((1, expected - cur))
                arr = np.concatenate([arr, pad], axis=1)
            elif cur > expected:
                # truncate rightmost features
                arr = arr[:, :expected]
    return arr

def ensemble_predict(input_array):
    """Return (avg_prob, level, probs_dict). Handles missing models gracefully."""
    # scale input
    if pre is None or 'scaler' not in pre:
        raise NotFittedError("Preprocessor (scaler) not available.")
    input_scaled = pre['scaler'].transform(input_array)

    probs = []
    probs_dict = {}
    if rf is not None:
        p_rf = rf.predict_proba(input_scaled)[:,1][0]
        probs.append(p_rf); probs_dict['rf'] = float(p_rf)
    if xgb is not None:
        try:
            p_xgb = xgb.predict_proba(input_scaled)[:,1][0]
            probs.append(p_xgb); probs_dict['xgb'] = float(p_xgb)
        except Exception:
            probs_dict['xgb'] = None
    if mlp is not None:
        try:
            p_mlp = float(np.array(mlp.predict(input_scaled)).flatten()[0])
            probs.append(p_mlp); probs_dict['mlp'] = float(p_mlp)
        except Exception:
            probs_dict['mlp'] = None

    if len(probs) == 0:
        raise RuntimeError("No models available for prediction.")
    avg = float(np.mean(probs))
    low, high = thr.get('low',0.4), thr.get('high',0.7)
    if avg < low:
        level = "Low Stress"
    elif avg < high:
        level = "Moderate Stress"
    else:
        level = "High Stress"
    return avg, level, probs_dict

# --------------- GPT (optional) ----------------
def gpt_explain_stub(level, prob):
    return f"(GPT disabled) Predicted {level} with probability {prob*100:.2f}%."

# If user placed OPENAI key in .streamlit/secrets.toml, use it
openai_available = False
try:
    import openai
    key = st.secrets.get("OPENAI_API_KEY", None)
    if key:
        openai.api_key = key
        openai_available = True
except Exception:
    openai_available = False

def gpt_explain(level, prob):
    if not openai_available:
        return gpt_explain_stub(level, prob)
    prompt = f"Stress probability: {prob*100:.2f}%. Category: {level}. Provide a short medically-safe explanation and 3 lifestyle tips."
    try:
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                           messages=[{"role":"system","content":"You are a helpful medical assistant."},
                                                     {"role":"user","content":prompt}],
                                           max_tokens=200)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT error: {e})"

# --------------- UI ----------------
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("**Modes:** Manual / CSV upload / Camera capture / Image upload")

mode = st.selectbox("Select mode", ["Manual Entry","Upload CSV","Camera Capture (face)","Upload Image (face)"])

# Common HRV manual inputs (left side)
with st.expander("HRV / Physiological inputs (used in Manual mode and combined with face features):"):
    c1, c2 = st.columns(2)
    with c1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 30, 200, 75.0)
        sdnn = st.number_input("SDNN (ms)", 1, 500, 50.0)
        rmssd = st.number_input("RMSSD (ms)", 1, 500, 30.0)
        pnn50 = st.number_input("pNN50 (%)", 0.0, 100.0, 20.0)
        lf = st.number_input("LF power (ms²)", 0.0, 10000.0, 800.0)
    with c2:
        hf = st.number_input("HF power (ms²)", 0.0, 10000.0, 600.0)
        sd1 = st.number_input("SD1 (ms)", 0.0, 200.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 0.0, 300.0, 40.0)
        temp = st.number_input("Skin Temp (°C)", 30.0, 40.0, 36.5)
        eda = st.number_input("EDA (µS)", 0.0, 50.0, 2.0)

hrv_manual = {'mean_hr':mean_hr,'sdnn':sdnn,'rmssd':rmssd,'pnn50':pnn50,'lf':lf,'hf':hf,'sd1':sd1,'sd2':sd2,'temp':temp,'eda':eda}

if mode == "Manual Entry":
    st.write("Click Predict to use manual HRV + default/no face features.")
    if st.button("Predict (Manual)"):
        face_agg=None
        inp = build_input_row(hrv_manual, face_agg, pre['scaler'] if pre else None)
        try:
            avg, level, probs = ensemble_predict(inp)
            st.metric("Stress Probability", f"{avg*100:.2f}%")
            st.write("Category:", level)
            st.write("Model probs:", probs)
            st.info(gpt_explain(level, avg))
        except Exception as e:
            st.error(f"Prediction error: {e}")

elif mode == "Upload CSV":
    st.write("Upload CSV with columns matching the scaler / model expectation (or with base HRV columns).")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict CSV"):
            try:
                # Build/pad to scaler features if necessary then predict for each row
                scaler = pre['scaler'] if pre else None
                # If df already matches scaler.n_features_in_, good. Else try to pick base columns and pad face zeros.
                rows = []
                for _, r in df.iterrows():
                    # r could contain full feature set already
                    rdict = {c: r.get(c, 0.0) for c in ['mean_hr','sdnn','rmssd','pnn50','lf','hf','sd1','sd2','temp','eda']}
                    inp = build_input_row(rdict, None, scaler)
                    rows.append(inp[0])
                X_all = np.vstack(rows)
                if scaler is not None:
                    Xs = scaler.transform(X_all)
                else:
                    Xs = X_all
                # ensemble predictions
                preds = []
                if rf: preds.append(rf.predict_proba(Xs)[:,1])
                if xgb:
                    try: preds.append(xgb.predict_proba(Xs)[:,1])
                    except: preds.append(np.zeros(Xs.shape[0]))
                if mlp:
                    try: preds.append(mlp.predict(Xs).flatten())
                    except: preds.append(np.zeros(Xs.shape[0]))
                avg_preds = np.mean(np.vstack(preds), axis=0)
                df['stress_prob'] = avg_preds
                df['stress_level'] = pd.cut(avg_preds, bins=[0, thr.get('low',0.4), thr.get('high',0.7), 1.0], labels=['Low','Moderate','High'])
                st.dataframe(df)
            except Exception as e:
                st.error(f"CSV Prediction error: {e}")

elif mode == "Camera Capture (face)":
    st.write("Camera capture: click 'Capture Image' multiple times (3-6) while looking at camera to collect frames, then 'Predict from captured frames'.")
    cap = st.camera_input("Capture face image (allow camera permission). Take 3-6 captures sequentially.")

    # We store recent captures in session_state list
    if 'captures' not in st.session_state:
        st.session_state['captures'] = []

    if cap is not None:
        # convert uploaded file (in-memory) to opencv BGR
        bytes_data = cap.getvalue()
        arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.session_state['captures'].append(img)
        st.success(f"Captured frames: {len(st.session_state['captures'])}")

    st.write("You can clear captures if you want.")
    if st.button("Clear Captures"):
        st.session_state['captures'] = []
        st.info("Cleared.")

    if st.session_state.get('captures') and st.button("Predict from captured frames"):
        frames = st.session_state['captures']
        feats = []
        for f in frames:
            ff = extract_face_features_from_bgr(f)
            if ff:
                feats.append({'mouth_ar':ff[0],'eye_ar':ff[1],'brow_gap':ff[2]})
        if len(feats)==0:
            st.error("No faces found in captured frames. Try again (face centered, good light).")
        else:
            df_feats = pd.DataFrame(feats)
            agg = {f"{c}_mean": df_feats[c].mean() for c in df_feats.columns}
            st.write("Aggregated face features:", agg)
            # build input row combining manual HRV and face features
            input_row = build_input_row(hrv_manual, agg, pre['scaler'] if pre else None)
            try:
                avg, level, probs = ensemble_predict(input_row)
                st.metric("Stress Probability", f"{avg*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probs:", probs)
                st.info(gpt_explain(level, avg))
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif mode == "Upload Image (face)":
    img_file = st.file_uploader("Upload a clear face image (jpg/png)", type=["jpg","jpeg","png"])
    if img_file:
        bytes_data = img_file.read()
        arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)
        ff = extract_face_features_from_bgr(img)
        if ff is None:
            st.error("No face detected. Upload a clearer image.")
        else:
            agg = {'mouth_ar_mean':ff[0],'eye_ar_mean':ff[1],'brow_gap_mean':ff[2]}
            st.write("Extracted face features:", agg)
            input_row = build_input_row(hrv_manual, agg, pre['scaler'] if pre else None)
            try:
                avg, level, probs = ensemble_predict(input_row)
                st.metric("Stress Probability", f"{avg*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probs:", probs)
                st.info(gpt_explain(level, avg))
            except Exception as e:
                st.error(f"Prediction error: {e}")

# SHAP visualization (optional)
st.markdown("---")
if st.button("Show SHAP (RF) summary"):
    if rf is None:
        st.warning("RandomForest model not available for SHAP.")
    else:
        try:
            # build random sample consistent with scaler shape
            if pre and 'scaler' in pre:
                n_features = getattr(pre['scaler'], 'n_features_in_', 10)
            else:
                n_features = 10
            Xs = np.random.rand(50, n_features)
            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(Xs)
            shap.summary_plot(sv[1] if isinstance(sv, list) else sv, Xs, show=False)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"SHAP error: {e}")

st.markdown("### Notes")
st.markdown("""
- Put all model files (preprocess.joblib, rf_hrv.joblib, xgb_hrv.joblib, mlp_hrv.* , adaptive_thresholds.npy) inside `models/` folder.
- Camera capture collects **frames** via `st.camera_input`. Take 3–6 captures for more robust face aggregation.
- If models are missing, you'll get friendly messages; app still runs.
- To enable GPT explanations, add your key to `.streamlit/secrets.toml`:
