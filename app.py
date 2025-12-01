# app.py -- Final clean, complete application
# AI-Based Heart & Stress Monitoring
# - Manual entry, CSV upload, Camera capture (MediaPipe face features), Image upload
# - Ensemble: RandomForest, XGBoost, MLP
# - Optional SHAP (if installed), optional GPT via OpenAI key in .streamlit/secrets.toml
# - Robust: handles missing models, missing optional libs, mismatched feature-lengths

import os
import io
import joblib
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# MUST be first Streamlit config call
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")

# ML libs
from sklearn.exceptions import NotFittedError

# Try TensorFlow (MLP) but do not fail if not installed / model missing
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional SHAP (import guarded)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# MediaPipe + OpenCV for face features (guarded)
try:
    import cv2
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
except Exception:
    MP_AVAILABLE = False
    mp_face_mesh = None

# OpenAI (GPT) optional via streamlit secrets
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ----------------- Paths -----------------
BASE_PATH = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def model_path(fname):
    return os.path.join(MODELS_DIR, fname)

def exists(fname):
    return os.path.exists(model_path(fname))

# ----------------- Load models -----------------
@st.cache_resource
def load_all():
    """Load preprocess + models (if present). Return dict."""
    loaded = {"pre": None, "rf": None, "xgb": None, "mlp": None, "thr": {"low":0.4, "high":0.7}}
    # Preprocessor (scaler, imputer) expected in preprocess.joblib
    try:
        if exists("preprocess.joblib"):
            loaded["pre"] = joblib.load(model_path("preprocess.joblib"))
        else:
            st.info("preprocess.joblib not found — app will still run but scaling may be skipped.")
    except Exception as e:
        st.warning(f"Failed to load preprocess.joblib: {e}")

    # Random forest
    try:
        if exists("rf_hrv.joblib"):
            loaded["rf"] = joblib.load(model_path("rf_hrv.joblib"))
    except Exception as e:
        st.warning(f"Failed to load rf_hrv.joblib: {e}")

    # XGBoost
    try:
        if exists("xgb_hrv.joblib"):
            loaded["xgb"] = joblib.load(model_path("xgb_hrv.joblib"))
    except Exception as e:
        st.info(f"xgb model not loaded: {e}")

    # MLP/TF model (optional)
    if TF_AVAILABLE:
        mlp_candidates = ["mlp_hrv.h5","mlp_hrv_v2.h5","mlp_hrv_clean.h5","mlp_hrv_fixed.keras","mlp_hrv_video.h5"]
        for c in mlp_candidates:
            try:
                if exists(c):
                    loaded["mlp"] = load_model(model_path(c), compile=False)
                    break
            except Exception:
                loaded["mlp"] = None
    else:
        # no tf installed
        loaded["mlp"] = None

    # thresholds
    try:
        if exists("adaptive_thresholds.npy"):
            thr = np.load(model_path("adaptive_thresholds.npy"), allow_pickle=True).item()
            if isinstance(thr, dict) and "low" in thr and "high" in thr:
                loaded["thr"] = thr
    except Exception:
        pass

    return loaded

models = load_all()
pre = models["pre"]
rf = models["rf"]
xgb = models["xgb"]
mlp = models["mlp"]
thr = models["thr"]

# ----------------- Utility: face feature extraction via MediaPipe -----------------
def _euclid(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def extract_face_features(img_bgr):
    """Return mouth_ar, eye_ar, brow_gap (floats). Return None if face not found or MP not available."""
    if not MP_AVAILABLE or mp_face_mesh is None:
        return None
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        h,w = img_bgr.shape[:2]
        def xy(i): return (int(lm[i].x*w), int(lm[i].y*h))
        # mouth vertical (13,14), corners (61,291)
        try:
            top_lip = xy(13); bottom_lip = xy(14); left_corner = xy(61); right_corner = xy(291)
            mouth_vert = _euclid(top_lip,bottom_lip); mouth_horiz = _euclid(left_corner,right_corner)+1e-6
            mouth_ar = mouth_vert / mouth_horiz
        except Exception:
            mouth_ar = 0.0
        # eyes (sample indices)
        try:
            r_top = xy(159); r_bottom = xy(145); r_left = xy(33); r_right = xy(133)
            l_top = xy(386); l_bottom = xy(374); l_left = xy(362); l_right = xy(263)
            r_ear = (_euclid(r_top,r_bottom)+1e-6) / (_euclid(r_left,r_right)+1e-6)
            l_ear = (_euclid(l_top,l_bottom)+1e-6) / (_euclid(l_left,l_right)+1e-6)
            eye_ar = (r_ear + l_ear)/2.0
        except Exception:
            eye_ar = 0.0
        # brow gap
        try:
            brow = xy(10); eye_top = xy(159)
            brow_gap = _euclid(brow, eye_top) / (h+1e-6)
        except Exception:
            brow_gap = 0.0
        return float(mouth_ar), float(eye_ar), float(brow_gap)
    except Exception:
        return None

# ----------------- Build input vector (order must match training scaler) -----------------
def build_input_row(hrv_dict, face_agg=None, scaler_obj=None):
    """
    hrv_dict: dict of base HRV features (mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda)
    face_agg: dict with keys mouth_ar_mean, eye_ar_mean, brow_gap_mean OR None
    scaler_obj: pre['scaler'] if available
    returns np.array shape (1, n_features) appropriate for scaler
    """
    base_order = ['mean_hr','sdnn','rmssd','pnn50','lf','hf','sd1','sd2','temp','eda']
    face_order = ['mouth_ar_mean','eye_ar_mean','brow_gap_mean']
    row = []
    for k in base_order:
        row.append(float(hrv_dict.get(k, 0.0)))
    if face_agg:
        for k in face_order:
            row.append(float(face_agg.get(k, 0.0)))
    else:
        row.extend([0.0,0.0,0.0])
    arr = np.array([row], dtype=float)
    # adjust to scaler expected size
    if scaler_obj is not None:
        expected = getattr(scaler_obj, "n_features_in_", None)
        if expected is not None:
            cur = arr.shape[1]
            if cur < expected:
                pad = np.zeros((1, expected-cur), dtype=float)
                arr = np.concatenate([arr, pad], axis=1)
            elif cur > expected:
                arr = arr[:, :expected]
    return arr

# ----------------- Ensemble predict -----------------
def ensemble_predict(input_array):
    """Return (avg_prob_float, level_str, dict_of_model_probs)."""
    # scale
    if pre and "scaler" in pre and pre["scaler"] is not None:
        Xs = pre["scaler"].transform(input_array)
    else:
        Xs = input_array
    probs = []
    p_dict = {}
    if rf is not None:
        try:
            p_rf = float(rf.predict_proba(Xs)[:,1][0]); probs.append(p_rf); p_dict['rf']=p_rf
        except Exception:
            p_dict['rf']=None
    if xgb is not None:
        try:
            p_x = float(xgb.predict_proba(Xs)[:,1][0]); probs.append(p_x); p_dict['xgb']=p_x
        except Exception:
            p_dict['xgb']=None
    if mlp is not None:
        try:
            p_m = float(np.array(mlp.predict(Xs)).flatten()[0]); probs.append(p_m); p_dict['mlp']=p_m
        except Exception:
            p_dict['mlp']=None
    if len(probs)==0:
        raise RuntimeError("No models available (rf/xgb/mlp not loaded). Put model files into models/ folder.")
    avg = float(np.mean(probs))
    low = thr.get('low', 0.4); high = thr.get('high', 0.7)
    if avg < low:
        level = "Low Stress"
    elif avg < high:
        level = "Moderate Stress"
    else:
        level = "High Stress"
    return avg, level, p_dict

# ----------------- GPT (optional) -----------------
def gpt_explain(level, prob):
    if not OPENAI_AVAILABLE:
        return "(GPT unavailable - openai package not installed or key not provided)"
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
        if not key:
            return "(GPT unavailable - set OPENAI_API_KEY in .streamlit/secrets.toml)"
        openai.api_key = key
        prompt = f"Predicted stress probability: {prob*100:.2f}%. Category: {level}. Provide a short doctor-style explanation (no medical diagnosis) and 3 simple, safe lifestyle tips."
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are a calm helpful medical assistant."},{"role":"user","content":prompt}],
            max_tokens=200
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT error: {e})"

# ----------------- UI -----------------
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("Modes: **Manual Entry** | **Upload CSV** | **Camera Capture** | **Upload Image**")

mode = st.selectbox("Choose mode", ["Manual Entry","Upload CSV","Camera Capture (face)","Upload Image (face)"])

# common manual HRV inputs
with st.expander("HRV / Physiological Inputs (manual) — used in Manual/Camera/Image modes"):
    c1, c2 = st.columns(2)
    with c1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 30.0, 200.0, 75.0)
        sdnn = st.number_input("SDNN (ms)", 1.0, 500.0, 50.0)
        rmssd = st.number_input("RMSSD (ms)", 1.0, 500.0, 30.0)
        pnn50 = st.number_input("pNN50 (%)", 0.0, 100.0, 20.0)
        lf = st.number_input("LF power (ms²)", 0.0, 20000.0, 800.0)
    with c2:
        hf = st.number_input("HF power (ms²)", 0.0, 20000.0, 600.0)
        sd1 = st.number_input("SD1 (ms)", 0.0, 500.0, 20.0)
        sd2 = st.number_input("SD2 (ms)", 0.0, 500.0, 40.0)
        temp = st.number_input("Skin Temperature (°C)", 30.0, 40.0, 36.5)
        eda = st.number_input("EDA (µS)", 0.0, 100.0, 2.0)

hrv_manual = {'mean_hr':mean_hr,'sdnn':sdnn,'rmssd':rmssd,'pnn50':pnn50,'lf':lf,'hf':hf,'sd1':sd1,'sd2':sd2,'temp':temp,'eda':eda}

# ---------------- Manual Entry ----------------
if mode == "Manual Entry":
    st.write("Manual input: uses the HRV inputs above (no face features).")
    if st.button("Predict (Manual)"):
        try:
            inp = build_input_row(hrv_manual, None, pre['scaler'] if pre else None)
            avg, level, probs = ensemble_predict(inp)
            st.metric("Stress Probability", f"{avg*100:.2f}%")
            st.write("Stress Category:", level)
            st.write("Model probabilities:", probs)
            st.info(gpt_explain(level, avg) if OPENAI_AVAILABLE else "(GPT not enabled)")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------- CSV Upload ----------------
elif mode == "Upload CSV":
    st.write("Upload CSV. Expected columns: mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda (face columns optional).")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict CSV"):
            try:
                rows = []
                scaler_obj = pre['scaler'] if pre else None
                for _, r in df.iterrows():
                    base = {c: r.get(c, 0.0) for c in ['mean_hr','sdnn','rmssd','pnn50','lf','hf','sd1','sd2','temp','eda']}
                    # face features in csv expected as mouth_ar_mean, eye_ar_mean, brow_gap_mean (optional)
                    face_agg = {k: r.get(k, 0.0) for k in ['mouth_ar_mean','eye_ar_mean','brow_gap_mean']}
                    row = build_input_row(base, face_agg, scaler_obj)[0]
                    rows.append(row)
                X_all = np.vstack(rows)
                # predictions
                preds_list = []
                if rf: preds_list.append(rf.predict_proba(X_all)[:,1])
                if xgb:
                    try: preds_list.append(xgb.predict_proba(X_all)[:,1])
                    except: preds_list.append(np.zeros(X_all.shape[0]))
                if mlp:
                    try: preds_list.append(np.array(mlp.predict(X_all)).flatten())
                    except: preds_list.append(np.zeros(X_all.shape[0]))
                avg_preds = np.mean(np.vstack(preds_list), axis=0)
                df['stress_prob'] = avg_preds
                df['stress_level'] = pd.cut(avg_preds, bins=[0, thr.get('low',0.4), thr.get('high',0.7), 1.0], labels=['Low','Moderate','High'])
                st.dataframe(df)
                csv_out = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results CSV", csv_out, "stress_results.csv", "text/csv")
            except Exception as e:
                st.error(f"CSV prediction failed: {e}")

# ---------------- Camera Capture ----------------
elif mode == "Camera Capture (face)":
    st.write("Camera capture: use the camera widget below. Take 3–6 captures and then 'Predict from Captures'.")
    # camera_input returns UploadedFile-like object
    img_file = st.camera_input("Capture face (allow camera permission). Take multiple captures sequentially.")
    # session_state to store captures
    if 'captures' not in st.session_state:
        st.session_state['captures'] = []
    if img_file is not None:
        # convert to cv2 image BGR
        bytes_data = img_file.getvalue()
        arr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        st.session_state['captures'].append(img)
        st.success(f"Captured frames: {len(st.session_state['captures'])}")
    if st.button("Clear Captures"):
        st.session_state['captures'] = []
        st.info("Cleared captures.")
    if st.session_state.get('captures') and st.button("Predict from Captures"):
        frames = st.session_state['captures']
        feats = []
        for f in frames:
            ff = extract_face_features(f)
            if ff:
                feats.append({'mouth_ar':ff[0],'eye_ar':ff[1],'brow_gap':ff[2]})
        if len(feats) == 0:
            st.error("No faces detected in captures. Try better lighting/face centered.")
        else:
            df_feats = pd.DataFrame(feats)
            agg = {f"{c}_mean": df_feats[c].mean() for c in df_feats.columns}
            st.write("Aggregated face features:", agg)
            try:
                inp = build_input_row(hrv_manual, agg, pre['scaler'] if pre else None)
                avg, level, probs = ensemble_predict(inp)
                st.metric("Stress Probability", f"{avg*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probs:", probs)
                st.info(gpt_explain(level, avg) if OPENAI_AVAILABLE else "(GPT not enabled)")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------- Image Upload ----------------
elif mode == "Upload Image (face)":
    st.write("Upload a face image (jpg/png). The app will extract face features and run prediction combined with manual HRV inputs.")
    img_u = st.file_uploader("Upload face image", type=["png","jpg","jpeg"])
    if img_u is not None:
        file_bytes = np.frombuffer(img_u.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)
        ff = extract_face_features(img)
        if ff is None:
            st.error("No face detected (or MediaPipe not available). Make sure face is visible and lighting is good.")
        else:
            agg = {'mouth_ar_mean':ff[0],'eye_ar_mean':ff[1],'brow_gap_mean':ff[2]}
            st.write("Extracted face features:", agg)
            try:
                inp = build_input_row(hrv_manual, agg, pre['scaler'] if pre else None)
                avg, level, probs = ensemble_predict(inp)
                st.metric("Stress Probability", f"{avg*100:.2f}%")
                st.write("Category:", level)
                st.write("Model probs:", probs)
                st.info(gpt_explain(level, avg) if OPENAI_AVAILABLE else "(GPT not enabled)")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------- SHAP (optional) ----------------
st.markdown("---")
if st.button("Show SHAP summary (RF)"):
    if rf is None:
        st.warning("RandomForest not loaded; SHAP summary unavailable.")
    elif not SHAP_AVAILABLE:
        st.warning("SHAP not installed. Install shap to see feature importance.")
    else:
        try:
            n_features = getattr(pre['scaler'], "n_features_in_", 10) if pre and 'scaler' in pre else 10
            Xs = np.random.rand(50, n_features)
            explainer = shap.TreeExplainer(rf)
            sv = explainer.shap_values(Xs)
            shap.summary_plot(sv[1] if isinstance(sv, list) else sv, Xs, show=False)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"SHAP plotting failed: {e}")

st.markdown("### Notes & Troubleshooting")
st.markdown("""
- Put your model files inside the `models/` folder (same folder as this app):
  `preprocess.joblib`, `rf_hrv.joblib`, `xgb_hrv.joblib`, optional `mlp_hrv*.h5`, `adaptive_thresholds.npy`.
- If you want GPT explanations, create `.streamlit/secrets.toml` with:
    OPENAI_API_KEY = \"your_openai_key_here\"
- If MediaPipe or OpenCV fails to install on Windows 3.13, try Python 3.10/3.11 or follow the troubleshooting notes in the README.
- If models were trained on different feature sets, ensure the ordering used here matches your scaler/model training order.
""")
