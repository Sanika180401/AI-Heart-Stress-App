import os
import time
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image
import cv2

# Optional libs
try:
    import shap
except Exception:
    shap = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

# ---------------------------
# Small helpers & config
# ---------------------------
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def load_model_safe(filename):
    p = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p):
        try:
            return load(p)
        except Exception as e:
            st.warning(f"Failed to load {filename}: {e}")
            return None
    return None

# load models (optional)
scaler = load_model_safe("scaler.pkl")
mlp = load_model_safe("mlp.pkl")
rf = load_model_safe("rf.pkl")
xgb = load_model_safe("xgb.pkl")
stacker = load_model_safe("stacker.pkl")

# ---------------------------
# Styling (light premium)
# ---------------------------
st.markdown("""
<style>
.stApp { background: #f7fafc; color: #0b1721; }
.logo-box { width:72px; height:72px; border-radius:12px; background:linear-gradient(135deg,#ff4d6d,#ff7a59); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:28px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
.card { background: white; border-radius:12px; padding:18px; box-shadow: 0 6px 20px rgba(15,23,42,0.04); border:1px solid rgba(2,6,23,0.03); }
.preview-img { width:360px; height:200px; object-fit:cover; border-radius:10px; }
.stButton > button { background: linear-gradient(90deg,#ff4d6d,#ff7a59); color:white; border-radius:10px; padding:8px 14px; font-weight:600; border:none; }
.stButton > button:hover { transform: translateY(-2px); }
.stSlider > div { color:#c41b23; }
.stress-outer { width:100%; background:#eef2f7; border-radius:10px; padding:6px; }
.stress-inner { height:26px; border-radius:8px; width:0%; background:linear-gradient(90deg,#ff7a59,#ff4d6d); color:white; display:flex; align-items:center; justify-content:flex-end; padding-right:8px; font-weight:700; transition: width 1s ease; }
.muted { color:#64748b; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# rPPG extraction (simplified CHROM-ish)
# ---------------------------
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
        fs_interp = 4.0
        times = np.cumsum(rr_ms)/1000.0
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

def vectorize_features(feats):
    # return 2D row
    return np.array([float(feats.get(k, 0.0) if feats.get(k) is not None else 0.0) for k in FEATURE_ORDER]).reshape(1,-1)

# ---------------------------
# Ensemble prediction & heuristics
# ---------------------------
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
        # fallback heuristic using mean_hr & rmssd:
        hr = feats.get("mean_hr", 75)
        rmssd = feats.get("rmssd", 30)
        # simple mapping: high HR + low RMSSD -> high stress
        score = np.clip((hr-60)/60 * 0.6 + (40 - min(rmssd,40))/40 * 0.4, 0, 1)
        return float(score), []
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
    try:
        hr = float(hr)
    except Exception:
        hr = 0.0
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

# ---------------------------
# AI-style explanation generator (local) — multi-language and level-specific
# ---------------------------
def generate_explanation_text(feats, stress_label, hr_label, risk_pct, risk_cat, lang="English"):
    # Templates per stress level and language
    templates = {
        "English": {
            "Low": {
                "one": "You're showing low stress and a healthy heart rate. Keep up good habits!",
                "rec": [
                    "Immediate: Take a short stretch or 30-second calm breath.",
                    "Today: Keep hydrated and take short breaks from screens.",
                    "Long-term: Maintain regular exercise and sleep schedule."
                ],
                "when": "Seek care if you notice chest pain, dizziness, or breathlessness."
            },
            "Moderate": {
                "one": "Your stress markers are moderate and heart rate is slightly elevated — minor changes can help.",
                "rec": [
                    "Immediate: Try 2 minutes of paced breathing (4s in, 4s out).",
                    "Today: Reduce caffeine, go for a 10-minute walk.",
                    "Long-term: Practice relaxation (yoga, mindfulness) regularly."
                ],
                "when": "Consult a doctor if symptoms persist or worsen."
            },
            "High": {
                "one": "Stress indicators are high and heart-rate is concerning — take action now.",
                "rec": [
                    "Immediate: Stop activity, sit/lie down, practice slow breathing for 5 minutes.",
                    "Today: Avoid stimulants, rest, and contact a healthcare professional if needed.",
                    "Long-term: Seek structured stress management and medical advice."
                ],
                "when": "If you experience chest pain, fainting, or severe breathlessness, seek emergency care."
            }
        },
        "Hindi": {
            "Low": {
                "one": "आपका तनाव कम है और हृदय गति स्वस्थ है। अच्छा बनाए रखें!",
                "rec": [
                    "तुरंत: थोड़ी स्ट्रेचिंग या 30 सेकंड की गहरी साँस लें।",
                    "आज: पानी भरपूर पिएँ और स्क्रीन से ब्रेक लें।",
                    "दीर्घकालिक: नियमित व्यायाम और अच्छी नींद बनाए रखें।"
                ],
                "when": "यदि सीने में दर्द, चक्कर या सांस लेने में कठिनाई हो तो डॉक्टर से संपर्क करें।"
            },
            "Moderate": {
                "one": "तनाव मध्यम है और हृदय गति थोड़ी ऊँची है — कुछ बदलाव मदद करेंगे।",
                "rec": [
                    "तुरंत: 2 मिनट की नियंत्रित श्वास (4 सेकंड अंदर, 4 बाहर)।",
                    "आज: कैफीन कम करें, 10 मिनट की सैर करें।",
                    "दीर्घकालिक: नियमित माइंडफुलनेस या योग करें।"
                ],
                "when": "यदि लक्षण बने रहें या बढ़ें तो डॉक्टर से सलाह लें।"
            },
            "High": {
                "one": "तनाव के संकेत उच्च हैं — तुरंत सावधानी बरतें।",
                "rec": [
                    "तुरंत: शांति से बैठकर 5 मिनट धीमी साँस लें।",
                    "आज: उत्तेजक चीजें टालें और आराम करें; आवश्यकता हो तो हेल्थकेयर से संपर्क करें।",
                    "दीर्घकालिक: स्ट्रेस मैनेजमेंट और मेडिकल सलाह लें।"
                ],
                "when": "यदि सीने में तेज दर्द या बेहोशी हो, तो आपातकालीन सहायता लें।"
            }
        },
        "Marathi": {
            "Low": {
                "one": "तणाव कमी आहे आणि हृदय गती सामान्य आहे. हे असेच ठेवा!",
                "rec": [
                    "तत्काळ: थोडी स्ट्रेचिंग किंवा 30 सेकंद श्वास घ्या.",
                    "आज: पुरेसं पाणी प्या आणि छोट्या ब्रेक घ्या.",
                    "दीर्घकालीन: नियमित व्यायाम व योग्य झोप ठेवा."
                ],
                "when": "छातीमध्ये वेदना, चक्कर किंवा श्वसनात त्रास असल्यास डॉक्टरांकडे जा."
            },
            "Moderate": {
                "one": "तणाव मध्यम आहे आणि हृदय गती थोडी वाढलेली आहे — काही बदल उपयोगी ठरतील.",
                "rec": [
                    "तत्काळ: 2 मिनिटांची नियंत्रित श्वास (4s इन, 4s आउट).",
                    "आज: कॅफीन कमी करा, 10 मिनिटांची चाल करा.",
                    "दीर्घकालीन: ध्यान / योगाचा समावेश करा."
                ],
                "when": "लक्षणे कायम राहिली तर डॉक्टरांचा सल्ला घ्या."
            },
            "High": {
                "one": "तणावाचे निदर्शने जास्त आहेत — ताबडतोब खबरदारी घ्या.",
                "rec": [
                    "तत्काळ: आराम करा आणि 5 मिनिटे हळूहळू श्वास घ्या.",
                    "आज: उत्तेजक पदार्थ टाळा आणि विश्रांती घ्या; गरज भासल्यास डॉक्टर संपर्क करा.",
                    "दीर्घकालीन: तणाव व्यवस्थापन व वैद्यकीय सल्ला घ्या."
                ],
                "when": "छातीतील वेदना, बेहोशी किंवा श्वासात गंभीर त्रास असल्यास आपत्कालीन मदत घ्या."
            }
        }
    }

    # Select template
    lang = lang if lang in templates else "English"
    tset = templates[lang].get(stress_label, templates["English"]["Moderate"])
    one = tset["one"]
    recs = tset["rec"]
    when = tset["when"]

    # Variation by HR label: small suffix
    hr_suffix = {
        "Low": {"English":" Your heart rate is a bit low; check for fatigue or medications." ,
                "Hindi":" आपकी हृदय गति थोड़ी कम है; थकान या दवाइयों को देखें।",
                "Marathi":"हृदय गती थोडी कमी आहे; थकवा किंवा औषधे तपासा."},
        "Normal": {"English":" Heart rate is in normal range.","Hindi":" हृदय गति सामान्य है।","Marathi":"हृदय गती सामान्य श्रेणीत आहे."},
        "High": {"English":" Heart rate is high — avoid exertion and monitor closely.","Hindi":" हृदय गति ऊँची है — व्यायाम से बचें और नजर रखें।","Marathi":"हृदय गती जास्त आहे — कष्ट टाळा आणि लक्ष ठेवा."}
    }
    suffix = hr_suffix.get(hr_label, hr_suffix["Normal"]).get(lang, "")

    # Build explanation text
    lines = []
    lines.append(one + " " + suffix)
    lines.append("")
    lines.append({"English":"Recommendations:","Hindi":"सिफारिशें:","Marathi":"शिफारसी:"}[lang])
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")
    lines.append({"English":"When to seek care:","Hindi":"कब चिकित्सक से मिलें:","Marathi":"केव्हा डॉक्टरांचा सल्ला घ्यावा:"}[lang])
    lines.append(when)
    return "\n".join(lines)

# ---------------------------
# UI layout
# ---------------------------
col_logo, col_title = st.columns([0.8, 6])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=72)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='color:#c41b23; margin:0'>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>", unsafe_allow_html=True)

st.write("")

with st.sidebar:
    st.header("Input")
    method = st.radio("Choose method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max video seconds", 6, 20, 10)
    rec_seconds = st.slider("Webcam record sec", 4, 12, 8)
    show_shap = st.checkbox("Show SHAP explainability", True)
    language = st.selectbox("Language", ["English","Hindi","Marathi"])
    st.markdown("---")
    

# main card inputs
left, right = st.columns([1,1.4])
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input")
    features = None

    if method == "Manual Entry":
        vals = {}
        cols = st.columns(2)
        for i, f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f=='mean_hr' else 20.0
                vals[f] = st.number_input(f, value=float(default))
        if st.button("Predict"):
            features = vals

    elif method == "Upload Image":
        uploaded = st.file_uploader("Image", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, width=360, caption="Preview")
            if st.button("Predict"):
                img = Image.open(uploaded).convert("RGB")
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                # surrogate features for single image
                feats = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":5.0, "sd1":12.0, "sd2":24.0, "lf_hf":0.8, "rr_mean":800.0, "rr_std":40.0}
                features = feats

    elif method == "Upload Video":
        vid = st.file_uploader("Video (mp4,mov,avi)", type=["mp4","mov","avi"])
        if vid:
            st.video(vid, start_time=0)
            if st.button("Predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(vid.read()); tmp.flush(); tmp.close()
                try:
                    sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                    if sig is None:
                        st.error("Couldn't extract reliable signal — use a clearer/longer video.")
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error("Pulse peaks not reliable — try again.")
                        else:
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
                feats = {"mean_hr":72 + (128-mean_g)/18.0, "rmssd":28.0, "pnn50":5.0, "sd1":12.0, "sd2":24.0, "lf_hf":0.8, "rr_mean":800.0, "rr_std":40.0}
                features = feats

    elif method == "Webcam Video":
        st.write("Click 'Start Webcam Recording' (works locally).")
        if st.button("Start Webcam Recording"):
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmpf.close()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Ensure local run and camera permission.")
            else:
                fps = 20.0
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read webcam frame.")
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
# After prediction: results
# ---------------------------
if 'features' in locals() and features is not None:
    feats = {k: float(features.get(k, 0.0) if features.get(k) is not None else 0.0) for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob)
    hr_label = categorize_hr(feats.get("mean_hr", 0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get("mean_hr", 0.0))
    summary_sentence = f"Predicted HR ≈ {feats.get('mean_hr',0):.0f} bpm | Stress: {stress_label} | Attack risk ≈ {risk_pct:.0f}% ({risk_cat})"

    # Gauge
    if go is not None:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=feats.get("mean_hr",60),
                                    gauge={'axis':{'range':[30,180]}, 'bar':{'color':'#ff4d6d'}},
                                    title={'text': "<b>Heart Rate (bpm)</b>"}))
        fig.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="white", font_color="#0b1721")
        gauge_placeholder.plotly_chart(fig, use_container_width=True)
    else:
        gauge_placeholder.info(f"HR: {feats.get('mean_hr',0):.0f} bpm")

    # result card with stress bar
    st.markdown(f"""
    <div class="card" style="margin-top:12px">
      <h3 style="color:#c41b23; margin-bottom:6px">Result</h3>
      <div class="muted">{summary_sentence}</div>
      <div style="height:10px"></div>
      <div class="stress-outer"><div class="stress-inner" style="width:{int(prob*100)}%;">{int(prob*100)}%</div></div>
      <div style="height:12px"></div>
      <div style="display:flex; gap:10px;">
        <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#c41b23">{stress_label}</b><div class="muted">Stress</div></div>
        <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#0b1721">{int(feats.get('mean_hr',0))} bpm</b><div class="muted">Heart Rate</div></div>
        <div style="flex:1; padding:10px; border-radius:10px; background:#fff;"><b style="color:#f97316">{int(risk_pct)}%</b><div class="muted">Attack Risk</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------  
# SHAP Explainability (Final Working Version)
# ---------------------------  
if show_shap and shap is not None and (rf is not None or xgb is not None or mlp is not None):
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("SHAP contributions")

    try:
        # pick best model for SHAP
        model_for_shap = rf or xgb or mlp or stacker

        # build small background data
        try:
            if scaler is not None and hasattr(scaler, "mean_"):
                means = scaler.mean_
                X_bg = np.tile(means.reshape(1, -1), (20, 1))
            else:
                # fallback: noisy version of current features
                base = vectorize_features(feats).reshape(-1)
                X_bg = np.vstack([base + np.random.normal(scale=0.01, size=base.shape) for _ in range(20)])
        except:
            base = vectorize_features(feats).reshape(-1)
            X_bg = np.vstack([base + np.random.normal(scale=0.01, size=base.shape) for _ in range(20)])

        # scale if scaler exists
        try:
            X_for_shap = scaler.transform(vectorize_features(feats)) if scaler is not None else vectorize_features(feats)
            X_bg_for_shap = scaler.transform(X_bg) if scaler is not None else X_bg
        except:
            X_for_shap = vectorize_features(feats)
            X_bg_for_shap = X_bg

        # choose explainer
        if hasattr(shap, "TreeExplainer") and (rf is not None or xgb is not None):
            explainer = shap.TreeExplainer(model_for_shap, data=X_bg_for_shap, feature_perturbation="interventional")

            try:
                shap_values = explainer.shap_values(X_for_shap)
            except:
                shap_values = explainer(X_for_shap).values
        else:
            explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(X_bg_for_shap))
            out = explainer(X_for_shap)
            shap_values = out.values

        # extract class-1 SHAP values properly
        if isinstance(shap_values, list):  
            svals = np.array(shap_values[1]).reshape(-1)[:len(FEATURE_ORDER)]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:  
                svals = arr[1].reshape(-1)[:len(FEATURE_ORDER)]
            else:
                svals = arr.reshape(-1)[:len(FEATURE_ORDER)]

        # show table
        sh_df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "SHAP": np.round(svals.astype(float), 4)
        })
        st.table(sh_df)

    except Exception as e:
        st.info(f"SHAP not available: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # AI Explanation (local templates)
    st.markdown("<div class='card' style='margin-top:14px'>", unsafe_allow_html=True)
    st.subheader("AI Explanation")
    ai_text = generate_explanation_text(feats, stress_label, hr_label, risk_pct, risk_cat, lang=language)
    st.markdown(f"```text\n{ai_text}\n```")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
