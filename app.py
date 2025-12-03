import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from joblib import load
from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

# Optional libs
try:
    import plotly.graph_objects as go
except:
    go = None

try:
    import shap
except:
    shap = None

# Try to import google-genai - if absent, we'll fall back safely
try:
    from google import genai
except Exception:
    genai = None

# ---------------------------
# Streamlit layout + theme
# ---------------------------
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer", layout="wide")

st.markdown(
    """
<style>
.stApp { background: #f7f9fb; color:#0b1721; }
.logo-box { width:70px; height:70px; border-radius:15px;
  background:linear-gradient(135deg,#ff4d6d,#ff7a59);
  display:flex; align-items:center; justify-content:center;
  color:white; font-size:28px; font-weight:700; box-shadow:0 6px 18px rgba(0,0,0,0.12); }
.card { background:white; border-radius:14px; padding:18px; margin-bottom:14px;
  box-shadow:0 6px 20px rgba(15,23,42,0.07); border:1px solid rgba(0,0,0,0.04); }
.stButton > button { background:linear-gradient(90deg,#ff4d6d,#ff7a59); color:white; border:none; border-radius:10px; padding:10px 20px; font-weight:600; }
.preview-img { width:300px; height:180px; object-fit:cover; border-radius:10px; border:1px solid #ddd; }
.stress-outer { background:#eef2f7; padding:5px; border-radius:12px; }
.stress-inner { height:26px; border-radius:10px; background:linear-gradient(90deg,#ff7a59,#ff4d6d); color:white; font-weight:700; text-align:right; padding-right:8px; transition:width 1.0s ease; }
.muted { color:#64748b; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------
# Paths and models
# ---------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

def load_model_safe(name):
    p = os.path.join(MODELS_DIR, name)
    if os.path.exists(p):
        try:
            return load(p)
        except Exception:
            return None
    return None

scaler = load_model_safe("scaler.pkl")
stacker = load_model_safe("stacker.pkl")
mlp = load_model_safe("mlp.pkl")
rf = load_model_safe("rf.pkl")
xgb = load_model_safe("xgb.pkl")

# ---------------------------
# Gemini / AI helper (safe)
# ---------------------------

def get_gemini_client_from_file():
    """
    Try to configure google-genai (if installed) using gemini_key.txt.
    Return a client object or None.
    """
    keyfile = os.path.join(ROOT, "gemini_key.txt")
    if genai is None or not os.path.exists(keyfile):
        return None
    try:
        key = open(keyfile, "r", encoding="utf-8").read().strip()
        # some genai versions use genai.configure(api_key=...) others different; try configure
        try:
            genai.configure(api_key=key)
            # keep genai module as client
            return genai
        except Exception:
            # if module provides Client class, attempt to instantiate
            try:
                client = genai.Client(api_key=key)
                return client
            except Exception:
                return None
    except Exception:
        return None

gemini = get_gemini_client_from_file()

# A robust fallback local explainer if Gemini fails or model unsupported
def local_explain(feats, prob, risk_pct, risk_cat, lang="English"):
    # Create a short one-liner and three recommendations heuristically
    hr = int(round(feats.get("mean_hr", 0)))
    stress_cat = "Low" if prob < 0.35 else ("Moderate" if prob < 0.65 else "High")
    if lang == "Hindi":
        one = f"अनुमानित HR ≈ {hr} bpm, तनाव: {stress_cat},  हार्ट-रिस्क ~ {int(risk_pct)}%."
        rec1 = "गहरी श्वास लें (2 मिनट)।"
        rec2 = "आज हल्की फिजिकल एक्टिविटी करें।"
        rec3 = "लंबे समय में नींद व आहार सुधारें।"
        care = "यदि साँस फूलना या वेज दर्द हो तो डॉक्टर से मिलें।"
    elif lang == "Marathi":
        one = f"अनुमानित HR ≈ {hr} bpm, ताण: {stress_cat}, हार्ट-रिस्क ~ {int(risk_pct)}%."
        rec1 = "सखोल श्वास घ्या (2 मिनिटे)."
        rec2 = "आज हलकी व्यायाम करा."
        rec3 = "दीर्घकालीन: झोप व आहार सांभाळा."
        care = "श्वास फु सर्व किंवा छातीत वेदना तर वैद्याकरांची मदत घ्या."
    else:
        one = f"Predicted HR ≈ {hr} bpm, Stress: {stress_cat}, Heart-risk ≈ {int(risk_pct)}%."
        rec1 = "Do slow breathing for 2 minutes (4s in, 4s out)."
        rec2 = "Take a short walk and hydrate today."
        rec3 = "Improve sleep and reduce caffeine long-term."
        care = "Seek medical help for chest pain, fainting, or shortness of breath."

    out = f"{one}\n\nRecommendations:\n1. {rec1}\n2. {rec2}\n3. {rec3}\n\nWhen to seek care: {care}"
    return out

def safe_gemini_generate(feats, prob, risk_pct, risk_cat, lang="English"):
    """
    Try to call Gemini safely. If any error / unsupported model, return None so we fallback locally.
    We DO NOT raise exceptions to Streamlit; we swallow and return fallback text.
    """
    if gemini is None:
        return None  # caller will fallback

    prompt_lang = {"English":"English","Hindi":"Hindi","Marathi":"Marathi"}.get(lang, "English")
    prompt = (
        f"You are a concise medical assistant. Reply in {prompt_lang}.\n\n"
        f"HRV features: {feats}\nStress probability: {prob:.2f}\n"
        f"Heart attack estimate: {risk_pct:.0f}% ({risk_cat})\n\n"
        "Provide: 1) One-sentence explanation. 2) Three short recommendations (immediate, today, long-term). "
        "3) One-sentence when to seek medical care."
    )

    # Try a few model names (some regions / SDKs use different names); if any works, return text.
    candidate_models = ["gemini-1.5-realtime", "gemini-1.5", "gemini-1.5-flash", "gemini-1.0"]
    # If module is configured like genai (with generate_text or models.generate_content), attempt robustly
    try:
        # preferred approach: if module has models.generate_content use it (newer SDK)
        if hasattr(gemini, "models") and hasattr(gemini.models, "generate_content"):
            for m in candidate_models:
                try:
                    resp = gemini.models.generate_content(model=m, prompt=prompt)  # some SDK variants accept 'prompt'
                    # resp may be object or dict-like; try to extract .text or .output[0]
                    if hasattr(resp, "content"):
                        txt = getattr(resp, "content")
                        return txt
                    # check common shapes
                    if isinstance(resp, dict):
                        # new API may return {'candidates':[{'content':'...'}]}
                        if "candidates" in resp and len(resp["candidates"])>0 and "content" in resp["candidates"][0]:
                            return resp["candidates"][0]["content"]
                        if "output" in resp:
                            try:
                                return json.dumps(resp["output"])[:2000]
                            except:
                                return str(resp)
                    # fallback
                    try:
                        return str(resp)
                    except:
                        pass
                except Exception:
                    continue

        # older SDK: gemini.generate_text or genai.generate_text style
        if hasattr(gemini, "generate_text"):
            for m in candidate_models:
                try:
                    resp = gemini.generate_text(model=m, prompt=prompt)
                    # try common accessors
                    if hasattr(resp, "text"):
                        return resp.text
                    if isinstance(resp, dict) and "candidates" in resp and len(resp["candidates"])>0:
                        return resp["candidates"][0].get("content", str(resp))
                    return str(resp)
                except Exception:
                    continue

        # some SDKs expose Client with .models or .generate; try generic
        # If nothing worked, return None to fallback.
        return None
    except Exception:
        return None

def generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang="English"):
    # Try Gemini first (safe); if fails, use local fallback.
    gem = safe_gemini_generate(feats, prob, risk_pct, risk_cat, lang)
    if gem is not None:
        return gem
    # fallback local explanation (no external call, guaranteed)
    return local_explain(feats, prob, risk_pct, risk_cat, lang)

# ---------------------------
# Signal processing helpers
# ---------------------------
def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,sig)

def extract_mean_green_signal_from_video_file(video_path, max_seconds=10, resize_width=360):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds)
    max_frames = min(total, int(fps*max_seconds))
    sig = []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret: break
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

# Ensemble predict
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

def categorize_stress(p, lang="English"):
    if p < 0.35: return {"English":"Low","Hindi":"कम","Marathi":"कमी"}[lang]
    if p < 0.65: return {"English":"Moderate","Hindi":"मध्यम","Marathi":"मध्यम"}[lang]
    return {"English":"High","Hindi":"उच्च","Marathi":"जास्त"}[lang]

def categorize_hr(hr, lang="English"):
    if hr < 60: return {"English":"Low","Hindi":"कम","Marathi":"कमी"}[lang]
    if hr <= 100: return {"English":"Normal","Hindi":"सामान्य","Marathi":"सामान्य"}[lang]
    return {"English":"High","Hindi":"उच्च","Marathi":"उच्च"}[lang]

def heart_attack_risk_heuristic(prob, mean_hr):
    hr_score = max(0.0, (mean_hr - 60.0) / 60.0)
    risk_raw = 0.6 * prob + 0.4 * min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20: cat="Low"
    elif risk_pct < 50: cat="Moderate"
    else: cat="High"
    return risk_pct, cat

# ---------------------------
# UI header / sidebar
# ---------------------------
col_logo, col_title = st.columns([0.8, 6])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("<h1>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Input")
    method = st.radio("Choose method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max video seconds", 6, 20, 10)
    rec_seconds = st.slider("Webcam recording seconds", 4, 12, 8)
    show_shap = st.checkbox("Show SHAP Explainability", True)
    lang = st.selectbox("Language", ["English","Hindi","Marathi"])
    st.markdown("---")
    st.markdown(f"Gemini AI: {'Available' if gemini is not None else 'Missing'}")

# Main layout placeholders
left, right = st.columns([1, 1.4])
features = None

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input")
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
            st.image(uploaded, width=300, caption="Preview")
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
                tmp = "tmp_uploaded_video.mp4"
                open(tmp,"wb").write(vid.read())
                sig, fps = extract_mean_green_signal_from_video_file(tmp, max_seconds=max_seconds)
                if sig is None:
                    st.error("Couldn't extract reliable signal — use clearer/longer video.")
                else:
                    mean_hr, hr_series = get_hr_from_signal(sig, fps)
                    if hr_series is None:
                        st.error("Pulse peaks not reliable.")
                    else:
                        rr_ms = (60000.0/np.array(hr_series))
                        feats = compute_hrv_from_rr(rr_ms, hr_series)
                        feats["mean_hr"] = mean_hr
                        features = feats
                try: os.remove(tmp)
                except: pass

    elif method == "Webcam Image":
        cam = st.camera_input("Capture image")
        if cam:
            st.image(cam, width=300)
            if st.button("Predict"):
                img = Image.open(cam).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":np.nan, "sd1":18.0, "sd2":36.0, "lf_hf":np.nan, "rr_mean":np.nan, "rr_std":np.nan}

    elif method == "Webcam Video":
        st.write("Click to record webcam video (local only).")
        if st.button("Start Webcam Recording"):
            tmpf = "tmp_webcam_record.mp4"
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam.")
            else:
                fps = 20.0
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read webcam.")
                else:
                    h,w = frame.shape[:2]
                    out = cv2.VideoWriter(tmpf, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
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
                        sig, fps = extract_mean_green_signal_from_video_file(tmpf, max_seconds=rec_seconds)
                        if sig is None:
                            st.error("Couldn't extract reliable signal from webcam video.")
                        else:
                            mean_hr, hr_series = get_hr_from_signal(sig, fps)
                            if hr_series is None:
                                st.error("Pulse not detected.")
                            else:
                                rr_ms = (60000.0/np.array(hr_series))
                                feats = compute_hrv_from_rr(rr_ms, hr_series)
                                feats["mean_hr"] = mean_hr
                                features = feats
                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                    finally:
                        try: os.remove(tmpf)
                        except: pass

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Live Dashboard")
    gauge_placeholder = st.empty()
    metrics_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# After prediction: results
# ---------------------------
if features is not None:
    # Ensure features contains all keys
    feats = {k: float(features.get(k, np.nan)) for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob, lang)
    hr_label = categorize_hr(feats.get("mean_hr", 0.0), lang)
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get("mean_hr", 0.0))
    sentence = f"HR ≈ {feats.get('mean_hr', np.nan):.0f} bpm ({hr_label}). Stress: {stress_label} ({prob:.2f}). Heart-attack estimate: {risk_pct:.0f}% ({risk_cat})."

    # gauge
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

    # neon card
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

# ---------------------------
# SHAP explainability (improved)
# ---------------------------
if show_shap and shap is not None and rf is not None:
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("SHAP contributions")

    try:
        # 1) Replace NaN with safe median defaults
        safe_feats = {k: (feats[k] if not np.isnan(feats[k]) else 0.0) for k in FEATURE_ORDER}

        # 2) Create proper dataframe
        feature_columns = FEATURE_ORDER.copy()
        input_df = pd.DataFrame([[safe_feats[c] for c in feature_columns]], columns=feature_columns)

        # 3) KernelExplainer (works on ANY classifier)
        background = np.zeros((1, len(feature_columns)))
        explainer = shap.KernelExplainer(lambda X: rf.predict_proba(X)[:,1], background)

        shap_values = explainer.shap_values(input_df)

        # 4) SHAP vector
        vals = shap_values.reshape(-1)

        # 5) Display table
        sh_df = pd.DataFrame({
            "Feature": feature_columns,
            "SHAP": vals.round(4)
        })

        st.table(sh_df)

    except Exception as e:
        st.info(f"SHAP not available: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # AI Explanation (Gemini safe + fallback)
    # ---------------------------
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("AI Explanation")
    ai_text = generate_ai_explanation(feats, prob, risk_pct, risk_cat, lang)
    st.write(ai_text)
    st.markdown("</div>", unsafe_allow_html=True)

# small spacer
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
