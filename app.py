# app.py — Updated for Python 3.13 (no SHAP / no SciPy)
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# TensorFlow / Keras
from tensorflow.keras.models import load_model

# Webcam / DeepFace
from deepface import DeepFace
import cv2, time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ===============================================================
# STREAMLIT CONFIG
# ===============================================================
st.set_page_config(page_title="AI Heart & Stress Monitoring", layout="wide")
st.title("AI-Based Real-Time Heart Rate & Stress Monitoring")
st.markdown("### Early Heart Attack Risk Prediction using Non-Wearable Sensors")

# ===============================================================
# PATH SETUP (models folder)
# ===============================================================
BASE_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(BASE_PATH, "models")
os.makedirs(MODELS_PATH, exist_ok=True)

def safe_load(filename):
    path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        raise FileNotFoundError(path)
    return path

# ===============================================================
# LOAD MODELS (cached)
# ===============================================================
@st.cache_resource
def load_all():
    # Expect a preprocess object that contains scaler and optionally feature names
    pre = None
    try:
        pre = joblib.load(safe_load("preprocess.joblib"))
    except Exception as e:
        st.warning(f"preprocess.joblib not found or failed to load: {e}")

    rf = None
    try:
        rf = joblib.load(safe_load("rf_hrv.joblib"))
    except Exception:
        st.warning("rf_hrv.joblib not found or failed to load (RF disabled).")

    xgb = None
    try:
        xgb = joblib.load(safe_load("xgb_hrv.joblib"))
    except Exception:
        st.warning("xgb_hrv.joblib not found or failed to load (XGB disabled).")

    mlp = None
    try:
        mlp_path = safe_load("mlp_hrv_video.h5")
        mlp = load_model(mlp_path, compile=False)
    except Exception:
        st.warning("mlp_hrv_video.h5 not found or failed to load (MLP disabled).")

    # thresholds for low/moderate/high
    thr = {"low": 0.35, "high": 0.70}
    try:
        thr_path = os.path.join(MODELS_PATH, "adaptive_thresholds.npy")
        if os.path.exists(thr_path):
            thr = np.load(thr_path, allow_pickle=True).item()
    except Exception:
        st.warning("adaptive_thresholds.npy not found; using defaults.")

    return pre, rf, xgb, mlp, thr

pre, rf, xgb, mlp, thr = load_all()

# ===============================================================
# GPT OPTIONAL EXPLANATION
# ===============================================================
try:
    import openai
    openai.api_key = st.secrets.get("OPENAI_API_KEY", None)
except Exception:
    openai = None

def gpt_explain(stress_level, prob):
    if not openai or not openai.api_key:
        return "(GPT explanation unavailable — no API key configured.)"
    prompt = f"Stress probability {prob*100:.1f}% — level: {stress_level}. Provide short explanation and 3 tips."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are a calm medical assistant."},
                {"role":"user","content":prompt}
            ],
            max_tokens=200
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(GPT error: {e})"

# ===============================================================
# PREDICTION helper
# ===============================================================
def predict_stress_from_array(input_array):
    """input_array shape: (n_samples, n_features) in the order the scaler expects"""
    # If we have a preprocessor with scaler
    if pre and isinstance(pre, dict) and "scaler" in pre:
        scaler = pre["scaler"]
        Xs = scaler.transform(input_array)
    else:
        # fallback: standardize with mean/std of input (not ideal)
        Xs = (input_array - np.mean(input_array, axis=0)) / (np.std(input_array, axis=0) + 1e-8)

    preds = []
    if rf:
        try:
            preds.append(rf.predict_proba(Xs)[:,1])
        except Exception:
            preds.append(rf.predict(Xs))
    if xgb:
        try:
            preds.append(xgb.predict_proba(Xs)[:,1])
        except Exception:
            preds.append(xgb.predict(Xs))
    if mlp:
        try:
            preds.append(mlp.predict(Xs).flatten())
        except Exception:
            pass

    if len(preds) == 0:
        raise RuntimeError("No models available to predict.")
    avg = np.mean(preds, axis=0)
    return avg

# ===============================================================
# Feature importance fallback (no SHAP)
# ===============================================================
def plot_feature_importance():
    st.markdown("### Model Feature Importance (approximate)")
    feature_names = None
    if pre and isinstance(pre, dict) and pre.get("feature_names_in_", None):
        feature_names = pre["feature_names_in_"]
    elif pre and hasattr(pre, "feature_names_in_"):
        feature_names = pre.feature_names_in_
    else:
        # fallback generic names
        feature_names = [f"f{i+1}" for i in range(10)]

    fig, ax = plt.subplots(figsize=(8,4))
    importance = None
    labels = feature_names
    if rf is not None:
        try:
            importance = rf.feature_importances_
            ax.barh(labels[:len(importance)], importance)
            ax.set_title("RandomForest feature_importances_")
            st.pyplot(fig)
            return
        except Exception:
            pass
    if xgb is not None:
        try:
            importance = xgb.feature_importances_
            ax.barh(labels[:len(importance)], importance)
            ax.set_title("XGBoost feature_importances_")
            st.pyplot(fig)
            return
        except Exception:
            pass

    st.info("No tree-based model found to show feature importances. Showing generic placeholder.")
    ax.barh(labels[:len(labels)], np.ones(len(labels)))
    st.pyplot(fig)

# ===============================================================
# UI: Input mode
# ===============================================================
option = st.radio("Select Input Method:", ("Manual Entry", "Upload CSV", "Webcam Emotion Analysis", "Upload Image"))

# --- Manual Entry ---
if option == "Manual Entry":
    st.subheader("Enter HRV & Physiological Features (manual)")
    col1, col2 = st.columns(2)
    with col1:
        mean_hr = st.number_input("Mean Heart Rate (bpm)", 30,300,75)
        sdnn = st.number_input("SDNN (ms)", 0,1000,50)
        rmssd = st.number_input("RMSSD (ms)", 0,1000,30)
        sd1 = st.number_input("SD1 (ms)", 0.0,100.0,20.0)
        temp = st.number_input("Skin Temperature (°C)", 30.0,40.0,36.5)
    with col2:
        pnn50 = st.number_input("pNN50 (%)", 0.0,100.0,20.0)
        lf = st.number_input("LF Power (ms²)", 0.0,10000.0,800.0)
        hf = st.number_input("HF Power (ms²)", 0.0,10000.0,600.0)
        sd2 = st.number_input("SD2 (ms)", 0.0,200.0,40.0)
        eda = st.number_input("Electrodermal Activity (µS)", 0.0,50.0,2.0)

    if st.button("Predict"):
        try:
            input_row = np.array([[mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda]])
            avg = predict_stress_from_array(input_row)[0]
            level = "Low Stress" if avg < thr["low"] else ("Moderate Stress" if avg < thr["high"] else "High Stress")
            st.metric("Stress Probability", f"{avg*100:.2f}%")
            if level=="Low Stress":
                st.success(level)
            elif level=="Moderate Stress":
                st.warning(level)
            else:
                st.error(level)
            st.markdown("#### Explanation (GPT):")
            st.info(gpt_explain(level, avg))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Upload CSV ---
elif option == "Upload CSV":
    st.subheader("Upload CSV with columns: mean_hr, sdnn, rmssd, pnn50, lf, hf, sd1, sd2, temp, eda")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            X = df.values
            preds = predict_stress_from_array(X)
            df["Stress_Probability"] = preds
            df["Stress_Level"] = pd.cut(preds, bins=[0, thr["low"], thr["high"], 1],
                                        labels=["Low","Moderate","High"])
            st.dataframe(df)
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results", csv_out, "stress_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")

# --- Webcam Emotion Analysis ---
elif option == "Webcam Emotion Analysis":
    st.subheader("Live Webcam Emotion Analysis")
    st.info("Click START to begin webcam. Look at camera for 8–15 seconds. Then click Capture & Predict.")

    class EmotionTransformer(VideoTransformerBase):
        def __init__(self):
            self.buffer = []
            self.last_ts = time.time()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                emo = res.get("emotion", None)
                if emo:
                    # append a numeric mapping (dictionary) for each frame
                    self.buffer.append(emo)
                    if len(self.buffer) > 400:
                        # keep buffer limited
                        self.buffer.pop(0)
            except Exception:
                pass

            # draw dominant emotion overlay
            if len(self.buffer) > 0:
                last = self.buffer[-1]
                dom = max(last, key=last.get)
                cv2.putText(img, f"Emotion: {dom}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            return img

    webrtc_ctx = webrtc_streamer(
        key="emotion_cam",
        video_transformer_factory=EmotionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Capture & Predict"):
            if webrtc_ctx and webrtc_ctx.video_transformer:
                buffer = webrtc_ctx.video_transformer.buffer
                if not buffer or len(buffer) == 0:
                    st.error("No emotion data captured. Start webcam and look at camera for 8-15 seconds.")
                else:
                    emo_df = pd.DataFrame(buffer)
                    st.write("Aggregated emotion stats (first rows):")
                    st.dataframe(emo_df.describe().T)
                    # aggregate emotion means and std
                    agg = {}
                    for col in emo_df.columns:
                        agg[f"{col}_mean"] = emo_df[col].mean()
                        agg[f"{col}_std"] = emo_df[col].std()
                    # Now combine with default HRV placeholders (or you can add UI to input manual HRV fields beforehand)
                    hrv_defaults = [75, 50, 30, 20, 800, 600, 20, 40, 36.5, 2.0]
                    final_features = np.array([hrv_defaults + list(agg.values())])
                    try:
                        avg = predict_stress_from_array(final_features)[0]
                        level = "Low Stress" if avg < thr["low"] else ("Moderate Stress" if avg < thr["high"] else "High Stress")
                        st.metric("Stress Probability", f"{avg*100:.2f}%")
                        if level == "Low Stress":
                            st.success(level)
                        elif level == "Moderate Stress":
                            st.warning(level)
                        else:
                            st.error(level)
                        st.markdown("#### GPT Explanation (optional)")
                        st.info(gpt_explain(level, avg))
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
            else:
                st.error("Webcam not active — please START camera above.")

    with col2:
        st.write("Webcam stream is above. After starting, allow browser camera access.")

# --- Upload Image (single) ---
elif option == "Upload Image":
    st.subheader("Upload a face image (jpg/png) to analyze emotion + predict (uses default HRV values)")
    image_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if image_file:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_column_width=True)
        if st.button("Analyze + Predict"):
            try:
                res = DeepFace.analyze(display_img, actions=['emotion'], enforce_detection=False)
                emo = res.get("emotion", {})
                st.json(emo)
                # make features from emotion dict (means are just the raw values here)
                agg = {}
                for k,v in emo.items():
                    agg[f"{k}_mean"] = float(v)
                    agg[f"{k}_std"] = 0.0
                hrv_defaults = [75, 50, 30, 20, 800, 600, 20, 40, 36.5, 2.0]
                final_features = np.array([hrv_defaults + list(agg.values())])
                avg = predict_stress_from_array(final_features)[0]
                level = "Low Stress" if avg < thr["low"] else ("Moderate Stress" if avg < thr["high"] else "High Stress")
                st.metric("Stress Probability", f"{avg*100:.2f}%")
                if level == "Low Stress":
                    st.success(level)
                elif level == "Moderate Stress":
                    st.warning(level)
                else:
                    st.error(level)
            except Exception as e:
                st.error(f"DeepFace analysis failed: {e}")

# ===============================================================
# Feature importance visualization button (no SHAP)
# ===============================================================
st.markdown("---")
if st.button("Show Feature Importance (approximate)"):
    try:
        plot_feature_importance()
    except Exception as e:
        st.error(f"Feature importance failed: {e}")

# ===============================================================
# Show small note about models
# ===============================================================
st.markdown("""
**Notes:**
- Place models and preprocess objects in `models/` (see README).  
  Expected names: `preprocess.joblib`, `rf_hrv.joblib`, `xgb_hrv.joblib`, `mlp_hrv_video.h5`, `adaptive_thresholds.npy`.
- GPT explanations require `OPENAI_API_KEY` in Streamlit secrets or environment.
""")
