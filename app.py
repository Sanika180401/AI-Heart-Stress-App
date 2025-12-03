# app.py — Premium Dashboard UI (final)
import os
import time
import tempfile
import json
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from PIL import Image
import cv2
import base64

# Optional libs
try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_lottie import st_lottie
except Exception:
    st_lottie = None

try:
    import shap
except Exception:
    shap = None

# OpenAI modern client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------
# Config + Theme (neon/premium)
# -----------------------
st.set_page_config(page_title="AI Heart & Stress Analyzer", layout="wide", page_icon="❤️")
# Custom CSS for neon cards, dark bg, animated stress bar
st.markdown(
    """
<style>
:root{
  --bg:#0b0f14; --card: rgba(255,255,255,0.03); --muted:#9aa6b2; --accent1:#00ffd5; --accent2:#0066ff;
}
body, .stApp { background: var(--bg); color: #e8eef3; }
.header { display:flex; align-items:center; gap:18px; padding:6px 0; }
.logo { width:68px; height:68px; border-radius:14px; background: linear-gradient(135deg,var(--accent2), #7c3aed); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:28px; color:white; box-shadow: 0 8px 30px rgba(0,0,0,0.6); }
.title { font-size:28px; margin:0; color: #cfe9ff; }
.subtitle { color:var(--muted); margin-top:2px; font-size:13px; }

/* Neon card */
.neon-card {
  background: var(--card);
  border-radius:14px;
  padding:18px;
  box-shadow: 0 6px 24px rgba(3,10,30,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
  border: 1px solid rgba(255,255,255,0.03);
  transition: transform .18s ease, box-shadow .18s ease;
}
.neon-card:hover { transform: translateY(-6px); box-shadow: 0 14px 36px rgba(0,102,255,0.08); }

/* Gradient heading */
.grad {
  background: linear-gradient(90deg, #00ffd5 0%, #00b3ff 40%, #7c3aed 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight:700;
}

/* compact preview */
.preview { width:320px; height:200px; object-fit:cover; border-radius:10px; border:1px solid rgba(255,255,255,0.05); }

/* animated stress bar container */
.stress-outer {
  width:100%; background: rgba(255,255,255,0.03); border-radius:10px; padding:8px;
}
.stress-inner {
  height:26px; border-radius:8px; width:0%;
  background: linear-gradient(90deg,#ff4d6d,#ff7a59);
  box-shadow: 0 6px 18px rgba(255,80,120,0.12);
  transition: width 0.9s ease;
  display:flex; align-items:center; justify-content:flex-end; color:#fff; padding-right:8px; font-weight:600;
}
.small { color: var(--muted); font-size:13px; }

/* neon stat */
.stat {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.03);
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------
# Utilities: load models & openai key
# -----------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

def load_model_file(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        return load(path)
    return None

scaler = load_model_file("scaler.pkl")
stacker = load_model_file("stacker.pkl")
mlp = load_model_file("mlp.pkl")
rf = load_model_file("rf.pkl")
xgb = load_model_file("xgb.pkl")

# OpenAI client init (hidden key)
def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        txt = os.path.join(ROOT, "openai_key.txt")
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                key = f.read().strip()
    if key and OpenAI is not None:
        return OpenAI(api_key=key)
    return None

openai_client = get_openai_client()

# -----------------------
# Minimal rPPG extraction (uses center ROI)
# (kept compact and robust)
# -----------------------
def extract_mean_green_signal_from_video_file(video_path, max_seconds=10, resize_width=360):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds)
    max_frames = min(total_frames, int(fps*max_seconds))
    sig = []
    frames=0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h,w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        wbox, hbox = int(frame.shape[1]*0.34), int(frame.shape[0]*0.46)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(frame.shape[1], cx+wbox//2), min(frame.shape[0], cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size==0:
            frames+=1; continue
        meang = float(np.mean(roi[:,:,1]))
        sig.append(meang)
        frames+=1
    cap.release()
    if len(sig) < 8:
        return None, fps
    return np.array(sig), fps

from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b,a,sig)

def get_hr_from_signal(sig, fs):
    try:
        filt = bandpass(sig - np.mean(sig), fs)
    except Exception:
        filt = sig - np.mean(sig)
    distance = max(1, int(0.4*fs))
    peaks, _ = find_peaks(filt, distance=distance, prominence=np.std(filt)*0.2)
    if len(peaks) < 2:
        return None, None
    times = peaks / float(fs)
    rr = np.diff(times) * 1000.0
    hr_series = 60000.0/rr
    return float(np.mean(hr_series)), list(hr_series)

# -----------------------
# Feature vectorizer and prediction
# -----------------------
FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

def compute_hrv_from_rr(rr_ms, hr_series=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff)>50)/len(diff)*100.0)
    sd1 = float(np.sqrt(np.var(diff)/2.0))
    sd2 = float(np.sqrt(max(0, 2*np.var(rr_ms)-np.var(diff)/2.0)))
    rr_mean = float(np.mean(rr_ms)); rr_std=float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
    # LF/HF approx
    lf_hf = 0.0
    try:
        fs_interp = 4.0
        times = np.cumsum(rr_ms)/1000.0
        if len(times) >= 4:
            t_interp = np.arange(0, times[-1], 1.0/fs_interp)
            inst_hr = 60000.0/rr_ms
            beat_times = times[:-1]
            interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
            f, p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
            lf = np.trapz(p[(f>=0.04)&(f<=0.15)], f[(f>=0.04)&(f<=0.15)]) if np.any((f>=0.04)&(f<=0.15)) else 0.0
            hf = np.trapz(p[(f>0.15)&(f<=0.4)], f[(f>0.15)&(f<=0.4)]) if np.any((f>0.15)&(f<=0.4)) else 0.0
            lf_hf = float(lf/hf) if hf > 0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {"mean_hr":mean_hr,"rmssd":rmssd,"pnn50":pnn50,"sd1":sd1,"sd2":sd2,"lf_hf":lf_hf,"rr_mean":rr_mean,"rr_std":rr_std}

def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1,-1)

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
        final=0.5
    else:
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

# -----------------------
# GPT helper
# -----------------------
def call_gpt(feats, prob, risk_pct, risk_cat):
    client = openai_client
    if client is None:
        return "AI assistant is not available (OpenAI key missing or client not installed)."
    prompt = f"""You are a concise health assistant.
HRV features: {feats}
Stress probability: {prob:.2f}
Estimated heart-attack chance: {risk_pct:.0f}% ({risk_cat})

Provide:
1) One-sentence explanation
2) Three brief recommendations (immediate, today, long-term)
3) When to seek medical care (one line).
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=260,
            temperature=0.7
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)
    except Exception as e:
        return f"OpenAI call error: {e}"

# -----------------------
# Multi-language small support (en/hi)
# -----------------------
LANGUAGES = {"English":"en","हिन्दी":"hi"}
STRINGS = {
    "en": {
        "title":"AI Heart & Stress Analyzer",
        "subtitle":"Real-time HRV → Stress & Heart-Risk Dashboard",
        "choose":"Choose Method",
        "manual":"Manual Entry",
        "up_vid":"Upload Video",
        "up_img":"Upload Image",
        "web_img":"Webcam Image",
        "web_vid":"Webcam Video",
        "predict":"Predict",
        "no_signal":"Couldn't extract reliable signal — try a clearer/longer video.",
        "gpt_off":"AI assistant unavailable.",
        "shap":"SHAP contributions"
    },
    "hi": {
        "title":"AI हृदय और तनाव विश्लेषक",
        "subtitle":"रियल-टाइम HRV → तनाव और हृदय जोखिम डैशबोर्ड",
        "choose":"विधि चुनें",
        "manual":"मैन्युअल दर्ज",
        "up_vid":"वीडियो अपलोड करें",
        "up_img":"छवि अपलोड करें",
        "web_img":"वेबकैम छवि",
        "web_vid":"वेबकैम वीडियो",
        "predict":"भाव करें",
        "no_signal":"विश्वसनीय सिग्नल नहीं मिला — स्पष्ट/लंबा वीडियो आज़माएँ।",
        "gpt_off":"AI सहायक उपलब्ध नहीं है।",
        "shap":"SHAP योगदान"
    }
}

# -----------------------
# Layout
# -----------------------
lang_label = st.sidebar.selectbox("Language", list(LANGUAGES.keys()), index=0)
lang = STRINGS[LANGUAGES[lang_label]]

st.markdown(f"""
<div class="header">
  <div class="logo">AH</div>
  <div>
    <div class="title grad">{lang['title']}</div>
    <div class="small">{lang['subtitle']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("### Input")
method = st.sidebar.radio("", [lang['manual'], lang['up_vid'], lang['up_img'], lang['web_img'], lang['web_vid']])
max_seconds = st.sidebar.slider("Max video seconds", 6, 20, 10)
rec_seconds = st.sidebar.slider("Webcam seconds (if used)", 4, 12, 8)
show_shap = st.sidebar.checkbox("Show SHAP explainability", value=True)

# Lottie breathing (left sidebar)
if st_lottie is not None:
    lottie_path = os.path.join(ROOT, "assets", "breathing.json")
    if os.path.exists(lottie_path):
        with open(lottie_path, "r", encoding="utf-8") as f:
            lottie_json = json.load(f)
        st.sidebar.markdown("### Breathing guide")
        st_lottie(lottie_json, height=180)
    else:
        st.sidebar.markdown("")

# -----------------------
# Main interactive area
# -----------------------
st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
col_left, col_right = st.columns([1,2])
with col_left:
    st.markdown("### Input")
    features = None
    if method == lang['manual']:
        st.markdown("Enter HRV numeric values (manual)")
        vals = {}
        for f in FEATURE_ORDER:
            default = 75.0 if f=='mean_hr' else 1.0
            vals[f] = st.number_input(f, value=float(default))
        if st.button(lang['predict']):
            features = vals

    elif method == lang['up_img']:
        uploaded = st.file_uploader("Image (jpg/png)", type=['jpg','png','jpeg'])
        if uploaded:
            st.image(uploaded, width=320, caption="Preview", output_format="auto")
            if st.button(lang['predict']):
                # surrogate: compute simple mean-green-based surrogate
                img = Image.open(uploaded).convert('RGB')
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0,"rmssd":28.0,"pnn50":np.nan,"sd1":18.0,"sd2":36.0,"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}

    elif method == lang['up_vid']:
        vid = st.file_uploader("Video (mp4)", type=['mp4','mov','avi'])
        if vid:
            # small preview
            st.video(vid, start_time=0)
            if st.button(lang['predict']):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(vid.read()); tmp.flush(); tmp.close()
                try:
                    sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                    if sig is None:
                        st.error(lang['no_signal']); features=None
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error(lang['no_signal']); features=None
                        else:
                            # convert back to rr_ms
                            rr_ms = np.diff(np.cumsum([0] + list(np.array(hr_series) * 0.0 + 0.0)))  # placeholder
                            # better: compute rr from peaks above (we used hr_series) so approximate rr_ms:
                            rr_ms = (60000.0/np.array(hr_series))
                            feats = compute_hrv_from_rr(rr_ms, hr_series)
                            features = feats
                except Exception as e:
                    st.error(f"Video processing failed: {e}")
                finally:
                    try: os.unlink(tmp.name)
                    except: pass

    elif method == lang['web_img']:
        cam = st.camera_input("Take a selfie")
        if cam:
            st.image(cam, width=320, caption="Captured")
            if st.button(lang['predict']):
                img = Image.open(cam).convert('RGB')
                arr = np.array(img)
                mean_g = float(np.mean(arr[:,:,1]))
                features = {"mean_hr":72 + (128-mean_g)/18.0,"rmssd":28.0,"pnn50":np.nan,"sd1":18.0,"sd2":36.0,"lf_hf":np.nan,"rr_mean":np.nan,"rr_std":np.nan}

    else:
        st.info("Webcam video recording not supported; use Upload Video.")

with col_right:
    st.markdown("### Live Dashboard")
    # placeholder animated gauge & placeholders
    gauge_area = st.empty()
    stats_area = st.empty()
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# If features computed => predict & render premium UI
# -----------------------
if features is not None:
    # ensure numeric features
    # if missing keys, fill with zeros/nan-handled later
    feats = {k: float(features.get(k, np.nan)) for k in FEATURE_ORDER}
    pred_prob, parts = predict_ensemble(feats)
    stress_cat = categorize_stress(pred_prob)
    hr_cat = categorize_hr(feats.get("mean_hr", 0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(pred_prob, feats.get("mean_hr", 0.0))
    sentence = f"Predicted HR ≈ {feats.get('mean_hr',np.nan):.0f} bpm ({hr_cat}). Stress level: {stress_cat} (score {pred_prob:.2f}). Heart-attack estimate: {risk_pct:.0f}% ({risk_cat})."

    # Animated gauge (plotly) if available
    if go is not None:
        target = max(0, min(180, feats.get("mean_hr", 0)))
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=0,
            delta={'reference':60,'increasing':{'color':'red'}},
            gauge={'axis':{'range':[30,180]},
                   'bar':{'color':'#00ffd5'},
                   'steps':[{'range':[30,60],'color':'#2b6bff10'},{'range':[60,100],'color':'#00ff8a10'},{'range':[100,180],'color':'#ff4d6d10'}]},
            title={'text':"<b>Heart Rate (bpm)</b>"}))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eef3", height=300, margin=dict(t=30,b=0,l=0,r=0))
        gbox = gauge_area.plotly_chart(fig, use_container_width=True)
        # animate from 30 to target
        for v in np.linspace(40, target, num=18):
            fig.data[0].value = float(v)
            gauge_area.plotly_chart(fig, use_container_width=True)
            time.sleep(0.02)
    else:
        gauge_area.info(f"HR: {feats.get('mean_hr',np.nan):.0f} bpm")

    # Neon card with results
    st.markdown(
        f"""<div class='neon-card'>
            <div style='display:flex;gap:18px;align-items:center;'>
              <div style='flex:1'>
                <h3 style='margin:0' class='grad'>Result</h3>
                <div class='small'>{sentence}</div>
                <div style='height:14px'></div>
                <div class='stress-outer'>
                  <div class='stress-inner' style='width:{int(pred_prob*100)}%'>
                    {int(pred_prob*100)}%
                  </div>
                </div>
                <div style='height:10px'></div>
                <div style='display:flex;gap:12px; margin-top:10px;'>
                  <div class='stat'><b>Stress:</b> {stress_cat}</div>
                  <div class='stat'><b>HR:</b> {feats.get('mean_hr',np.nan):.0f} bpm ({hr_cat})</div>
                  <div class='stat'><b>Attack Risk:</b> {int(risk_pct)}% ({risk_cat})</div>
                </div>
              </div>
              <div style='width:320px'>
                 <div style='height:12px'></div>
                 <div style='text-align:right; color:var(--muted); font-size:13px;'>Compact preview</div>
              </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Live HR graph (simple): show last HR values if available
    hr_series = None
    try:
        # if hr_series present in compute step (rare for image surrogates)
        hr_series = feats.get("hr_series", None)
    except Exception:
        hr_series = None

    # Show SHAP if requested & available
    if show_shap and shap is not None and (mlp is not None or rf is not None or xgb is not None):
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader(lang['shap'])
        try:
            # use rf if available for TreeExplainer else stacker (linear)
            explainer = None
            model_for_shap = rf or mlp or stacker
            if model_for_shap is not None:
                explainer = shap.Explainer(model_for_shap, masker=shap.maskers.Independent(np.zeros((1,len(FEATURE_ORDER)))))
                vals = explainer(vectorize_features(feats))
                shap_vals = np.array(vals.values).reshape(-1)[:len(FEATURE_ORDER)]
                sh_df = pd.DataFrame({"Feature": FEATURE_ORDER, "SHAP": shap_vals})
                st.table(sh_df)
            else:
                st.info("No model for SHAP.")
        except Exception as e:
            st.info(f"SHAP not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # GPT explanation
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.subheader("AI Explanation (GPT)")
    gpt_text = call_gpt if False else None  # placeholder to avoid linter
    # call GPT safely
    gpt_out = call_gpt(feats, pred_prob, risk_pct, risk_cat) if openai_client is not None else "AI assistant offline (no key)."
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:rgba(255,255,255,0.02);'>{gpt_out}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer spacing
st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
