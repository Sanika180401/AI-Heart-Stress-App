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
from google import genai

# Optional libraries
try:
    import plotly.graph_objects as go
except:
    go = None

try:
    import shap
except:
    shap = None


# ============================================================
#   STREAMLIT CONFIG + LIGHT THEME
# ============================================================
st.set_page_config(page_title="AI Heart Rate & Stress Analyzer", layout="wide")

st.markdown(
    """
<style>
/* Background */
.stApp { background: #f7f9fb; color:#0b1721; }

/* HEADER */
.header { display:flex; gap:16px; align-items:center; }

/* LOGO BOX */
.logo-box {
    width:70px; height:70px; border-radius:15px;
    background:linear-gradient(135deg,#ff4d6d,#ff7a59);
    display:flex; justify-content:center; align-items:center;
    color:white; font-size:28px; font-weight:700;
    box-shadow:0 6px 18px rgba(0,0,0,0.15);
}

/* CARD */
.card {
    background:white; border-radius:14px;
    padding:18px; margin-bottom:14px;
    box-shadow:0 6px 20px rgba(15,23,42,0.07);
    border:1px solid rgba(0,0,0,0.04);
}

/* BUTTONS */
.stButton > button {
    background:linear-gradient(90deg,#ff4d6d,#ff7a59);
    color:white; border:none; border-radius:10px;
    padding:10px 20px; font-weight:600;
}
.stButton > button:hover {
    transform:translateY(-2px);
}

/* SECONDARY BUTTON */
.stButton.secondary > button {
    background:linear-gradient(90deg,#06b6d4,#3b82f6);
}

/* Compact image preview */
.preview-img {
    width:300px; height:180px; object-fit:cover;
    border-radius:10px; border:1px solid #ddd;
}

/* Stress bar */
.stress-outer { background:#eef2f7; padding:5px; border-radius:12px; }
.stress-inner {
    height:26px; border-radius:10px;
    background:linear-gradient(90deg,#ff7a59,#ff4d6d);
    color:white; font-weight:700; text-align:right;
    padding-right:8px;
    transition:width 1.0s ease;
}

/* Muted text */
.muted { color:#64748b; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#    DIRECTORIES & MODELS
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
ASSETS_DIR = os.path.join(ROOT, "assets")

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

def load_model(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        try:
            return load(path)
        except:
            return None
    return None

scaler = load_model("scaler.pkl")
stacker = load_model("stacker.pkl")
mlp = load_model("mlp.pkl")
rf = load_model("rf.pkl")
xgb = load_model("xgb.pkl")


# ============================================================
#   GEMINI (FREE) INITIALIZATION
# ============================================================
def load_gemini_client():
    keyfile = os.path.join(ROOT, "gemini_key.txt")
    if os.path.exists(keyfile):
        try:
            key = open(keyfile, "r").read().strip()
            return genai.Client(api_key=key)
        except:
            return None
    return None

gemini_client = load_gemini_client()


# ============================================================
#   LANGUAGE SYSTEM (EN / HI / MR)
# ============================================================
LANG = st.sidebar.selectbox("Language", ["English", "Hindi", "Marathi"])

def T(en, hi, mr):
    if LANG == "English": return en
    if LANG == "Hindi": return hi
    return mr


# ============================================================
#   SIGNAL PROCESSING (rPPG)
# ============================================================
def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = fs/2
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,sig)

def extract_green(video_path, max_seconds=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), fps*max_seconds))

    values=[]
    for _ in range(frames):
        ret, frame = cap.read()
        if not ret: break
        g = np.mean(frame[:,:,1])
        values.append(g)

    cap.release()
    if len(values)<20: return None, fps
    return np.array(values), fps

def heart_rate_from_signal(sig, fs):
    sig = sig - np.mean(sig)
    try:
        sig = bandpass(sig, fs)
    except:
        pass

    peaks,_ = find_peaks(sig, distance=int(0.4*fs))
    if len(peaks)<2: return None, None

    times = peaks/fs
    rr = np.diff(times)*1000
    hr = 60000/rr
    return float(np.mean(hr)), hr


def compute_hrv(rr_ms):
    if rr_ms is None or len(rr_ms)<2: return None
    diff = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff**2))
    pnn50 = np.sum(np.abs(diff)>50)/len(diff)*100
    sd1 = np.sqrt(np.var(diff)/2)
    sd2 = np.sqrt(max(0,2*np.var(rr_ms)-np.var(diff)/2))
    return {
        "rmssd":float(rmssd),
        "pnn50":float(pnn50),
        "sd1":float(sd1),
        "sd2":float(sd2),
        "rr_mean":float(np.mean(rr_ms)),
        "rr_std":float(np.std(rr_ms)),
    }


# ============================================================
#   PREDICTORS
# ============================================================
FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def vectorize(f):
    return np.array([f.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1,-1)

def predict_ensemble(feats):
    X = vectorize(feats)
    try:
        Xs = scaler.transform(X)
    except:
        Xs = X

    probs=[]
    for m in [mlp, rf, xgb]:
        if m is None: continue
        try: p = m.predict_proba(Xs)[0,1]
        except:
            try: p = m.predict(Xs)[0]
            except: p=0.5
        probs.append(float(p))

    if len(probs)==0:
        return 0.5, []

    meta = np.array(probs).reshape(1,-1)
    try:
        final = stacker.predict_proba(meta)[0,1]
    except:
        final = np.mean(probs)

    return float(final), probs


def stress_label(p):
    if p<0.35: return T("Low","कम","कमी")
    if p<0.65: return T("Moderate","मध्यम","मध्यम")
    return T("High","उच्च","जास्त")


def heart_attack_risk(prob, hr):
    hr_score = max(0,(hr-60)/60)
    raw = 0.6*prob + 0.4*min(1,hr_score)
    pct = float(np.clip(raw*100,0,100))
    if pct<20: cat=T("Low","कम","कमी")
    elif pct<50: cat=T("Moderate","मध्यम","मध्यम")
    else: cat=T("High","उच्च","जास्त")
    return pct, cat


# ============================================================
#   AI EXPLAINER — GEMINI (FREE)
# ============================================================
def gemini_explain(feats, prob, risk_pct, risk_cat):
    if gemini_client is None:
        return T(
            "AI assistant unavailable (missing gemini_key.txt).",
            "AI सहायक उपलब्ध नहीं (gemini_key.txt गायब)।",
            "AI सहाय्य उपलब्ध नाही (gemini_key.txt नाही)."
        )

    prompt = f"""
You are a friendly medical helper. Patient:

HRV Features: {feats}
Stress Probability: {prob:.2f}
Heart Attack Estimate: {risk_pct:.0f}% ({risk_cat})

Give very short:
1. One line explanation
2. Three small recommendations
3. One line: when to seek medical care
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini error: {e}"


# ============================================================
#   HEADER UI
# ============================================================
col1,col2 = st.columns([1,6])
with col1:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=70)
    else:
        st.markdown('<div class="logo-box">AI</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<h1 style='color:#c41b23; margin-bottom:0;'>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>{T('Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction','हृदयाघात के जोखिम की शीघ्र भविष्यवाणी के लिए हृदय गति और तनाव निगरानी प्रणाली','हृदयविकाराच्या सुरुवातीच्या जोखीम अंदाजासाठी हृदय गती आणि ताण देखरेख प्रणाली')}</div>", unsafe_allow_html=True)


# ============================================================
#   SIDEBAR INPUT
# ============================================================
with st.sidebar:
    st.header(T("Input","इनपुट","इनपुट"))
    method = st.radio(T("Select Method","विधि चुनें","पद्धत निवडा"),
                      ["Manual Entry","Upload Image","Upload Video","Webcam Image","Webcam Video"])

    max_seconds = st.slider("Video Duration", 6, 20, 10)
    rec_seconds = st.slider("Webcam Record Sec", 4, 12, 8)

    show_shap = st.checkbox("Show SHAP", True)

    st.markdown("---")
    st.markdown(f"**Gemini AI:** {'Available' if gemini_client else 'Missing Key'}")


# ============================================================
#   MAIN LAYOUT
# ============================================================
left, right = st.columns([1,1.4])

features = None

# ============================================================
# LEFT PANEL
# ============================================================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(T("Input Data","इनपुट डेटा","डेटा इनपुट"))

    # ------------------------
    # Manual Entry
    # ------------------------
    if method == "Manual Entry":
        vals={}
        cols = st.columns(2)
        for i,f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f=="mean_hr" else 1.0
                vals[f]=st.number_input(f, value=float(default))
        if st.button(T("Predict","अनुमान लगाएं","भाकीत करा")):
            features = vals

    # ------------------------
    # Upload Image
    # ------------------------
    elif method == "Upload Image":
        up = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
        if up:
            st.image(up, width=300)
            if st.button(T("Predict","अनुमान लगाएं","प्रेडिक्ट")):
                img = Image.open(up)
                arr = np.array(img)
                g = np.mean(arr[:,:,1])
                features = {
                    "mean_hr":72+(128-g)/18,
                    "rmssd":28, "pnn50":np.nan,
                    "sd1":18, "sd2":36,
                    "lf_hf":np.nan,
                    "rr_mean":np.nan, "rr_std":np.nan
                }

    # ------------------------
    # Upload Video
    # ------------------------
    elif method == "Upload Video":
        vid = st.file_uploader("Upload Video", type=["mp4","mov","avi"])
        if vid:
            st.video(vid)
            if st.button(T("Predict","अनुमान लगाएं","प्रेडिक्ट")):
                tmp = "temp_vid.mp4"
                open(tmp,"wb").write(vid.read())
                sig, fs = extract_green(tmp, max_seconds)
                if sig is None:
                    st.error(T("Could not extract signal.","सिग्नल नहीं निकला।","सिग्नल मिळाला नाही."))
                else:
                    hr, series = heart_rate_from_signal(sig, fs)
                    rr = 60000/series
                    hrv = compute_hrv(rr)
                    features = {"mean_hr":hr, "lf_hf":0.5, **hrv}
                os.remove(tmp)

    # ------------------------
    # Webcam Image
    # ------------------------
    elif method == "Webcam Image":
        cam = st.camera_input("Capture")
        if cam:
            st.image(cam, width=300)
            if st.button(T("Predict","अनुमान लगाएं","प्रेडिक्ट")):
                img = Image.open(cam)
                arr = np.array(img)
                g = np.mean(arr[:,:,1])
                features = {
                    "mean_hr":72+(128-g)/18,
                    "rmssd":28, "pnn50":np.nan,
                    "sd1":18, "sd2":36,
                    "lf_hf":np.nan,
                    "rr_mean":np.nan, "rr_std":np.nan
                }

    # ------------------------
    # Webcam Video
    # ------------------------
    elif method == "Webcam Video":
        if st.button("Start Recording"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Camera not available.")
            else:
                tmp = "webcam_temp.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                ret, frm = cap.read()
                h,w = frm.shape[:2]
                out = cv2.VideoWriter(tmp, fourcc, 20, (w,h))
                t0=time.time()
                pg = st.progress(0)

                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    out.write(frame)
                    elapsed=time.time()-t0
                    pg.progress(min(100,int((elapsed/rec_seconds)*100)))
                    if elapsed>=rec_seconds: break

                cap.release()
                out.release()

                st.success("Recorded. Processing...")
                sig, fs = extract_green(tmp, rec_seconds)
                if sig is not None:
                    hr, series = heart_rate_from_signal(sig, fs)
                    rr = 60000/series
                    hrv = compute_hrv(rr)
                    features = {"mean_hr":hr, "lf_hf":0.5, **hrv}
                os.remove(tmp)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# RIGHT PANEL
# ============================================================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(T("Dashboard","डैशबोर्ड","डॅशबोर्ड"))
    gauge_area = st.empty()
    metric_area = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================
if features:
    feats = {k:float(features.get(k,0)) for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    risk_pct, risk_cat = heart_attack_risk(prob, feats["mean_hr"])

    # -------------------------
    # HEART RATE GAUGE
    # -------------------------
    if go is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=feats["mean_hr"],
            title={"text":T("Heart Rate","हार्ट रेट","हार्ट रेट")},
            gauge={"axis":{"range":[30,180]}, "bar":{"color":"#ff4d6d"}}
        ))
        fig.update_layout(height=260, margin=dict(t=20,b=0))
        gauge_area.plotly_chart(fig, use_container_width=True)
    else:
        gauge_area.info(f"HR: {feats['mean_hr']:.1f}")

    # -------------------------
    # RESULT CARD
    # -------------------------
    stress_txt = stress_label(prob)
    summary = f"{T('Heart Rate','हार्ट रेट','हृदय गती')}: {feats['mean_hr']:.0f} bpm | " \
              f"{T('Stress','तनाव','ताण')}: {stress_txt} | " \
              f"{T('Attack Risk','हार्ट अटैक जोखिम','हार्ट अटॅक रिस्क')}: {risk_pct:.0f}%"

    st.markdown(
        f"""
        <div class="card">
            <h3 style="color:#c41b23;">{T('Result','परिणाम','निकाल')}</h3>
            <div class='muted'>{summary}</div>
            <br>
            <div class='stress-outer'>
                <div class='stress-inner' style='width:{int(prob*100)}%;'>
                    {int(prob*100)}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # SHAP
    # -------------------------
    if show_shap and shap is not None and rf is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("SHAP")

        try:
            expl = shap.Explainer(rf)
            sv = expl(vectorize(feats))
            vals = np.array(sv.values).reshape(-1)

            df = pd.DataFrame({"Feature":FEATURE_ORDER, "SHAP":vals})
            st.table(df)

        except Exception as e:
            st.info(f"SHAP not available: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # AI Explanation (Gemini)
    # -------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(T("AI Explanation","AI स्पष्टीकरण","AI स्पष्टीकरण"))

    ai_text = gemini_explain(feats, prob, risk_pct, risk_cat)
    st.write(ai_text)

    st.markdown("</div>", unsafe_allow_html=True)
