import os, time, json, tempfile, math
from pathlib import Path
import numpy as np, pandas as pd
import streamlit as st
from PIL import Image
import cv2
from joblib import load
from scipy.signal import find_peaks, butter, filtfilt, welch, detrend

# optional libs
try:
    import shap
except Exception:
    shap = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    go = None

# ------------------ config ------------------
st.set_page_config(page_title='AI Heart Rate & Stress Analyzer', layout='wide')
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / 'models'
ASSETS_DIR = ROOT / 'assets'
USERS_FILE = ROOT / 'users.json'
HISTORY_FILE = ROOT / 'history.json'  # persist trend history between runs (simple)

FEATURE_ORDER = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']

# ------------------ helper functions ------------------
def load_model_safe(name):
    p = MODELS_DIR / name
    if p.exists():
        try:
            return load(p)
        except Exception as e:
            st.warning(f'Failed to load {name}: {e}')
    return None

scaler = load_model_safe('scaler.pkl')
mlp = load_model_safe('mlp.pkl')
rf = load_model_safe('rf.pkl')
xgb = load_model_safe('xgb.pkl')
stacker = load_model_safe('stacker.pkl')

def bandpass(sig, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5*fs
    b,a = butter(order, [max(0.001, low/nyq), min(0.999, high/nyq)], btype='band')
    return filtfilt(b, a, sig)

def extract_mean_green_signal_from_video_file(video_path, max_seconds=10, resize_width=360):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError('Cannot open video file.')
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
        cx, cy = frame.shape[1]//2, frame.shape[0]//2
        wbox, hbox = int(frame.shape[1]*0.35), int(frame.shape[0]*0.45)
        x1,y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2,y2 = min(frame.shape[1], cx+wbox//2), min(frame.shape[0], cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1; continue
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
    if rr_ms is None or len(rr_ms) < 2: return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff)/2.0))
    sd2 = float(np.sqrt(max(0.0, 2*np.var(rr_ms) - np.var(diff)/2.0)))
    rr_mean = float(np.mean(rr_ms)); rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_series)) if hr_series is not None else float(60000.0/np.mean(rr_ms))
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
            lf = np.trapz(p[(f>=0.04)&(f<=0.15)], f[(f>=0.04)&(f<=0.15)]) if np.any((f>=0.04)&(f<=0.15)) else 0.0
            hf = np.trapz(p[(f>0.15)&(f<=0.4)], f[(f>0.15)&(f<=0.4)]) if np.any((f>0.15)&(f<=0.4)) else 0.0
            lf_hf = float(lf/hf) if hf>0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {'mean_hr':mean_hr,'rmssd':rmssd,'pnn50':pnn50,'sd1':sd1,'sd2':sd2,'lf_hf':lf_hf,'rr_mean':rr_mean,'rr_std':rr_std}

def vectorize_features(feats):
    return np.array([feats.get(k, np.nan) for k in FEATURE_ORDER]).reshape(1,-1)

# ensemble predict (same as before)
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
        # simple heuristic
        hr = feats.get('mean_hr',75)
        rmssd = feats.get('rmssd',30)
        score = np.clip((hr-60)/60*0.6 + (40 - min(rmssd,40))/40*0.4, 0, 1)
        return float(score), []
    meta = np.array(probs).reshape(1,-1)
    try:
        final = float(stacker.predict_proba(meta)[:,1][0]) if stacker is not None else float(np.mean(probs))
    except Exception:
        final = float(np.mean(probs))
    return final, probs

def categorize_stress(p):
    if p < 0.35: return 'Low'
    if p < 0.65: return 'Moderate'
    return 'High'
def categorize_hr(hr):
    if hr < 60: return 'Low'
    if hr <= 100: return 'Normal'
    return 'High'
def heart_attack_risk_heuristic(prob, mean_hr):
    hr_score = max(0.0, (mean_hr - 60.0) / 60.0)
    risk_raw = 0.6 * prob + 0.4 * min(1.0, hr_score)
    risk_pct = float(np.clip(risk_raw * 100.0, 0, 100))
    if risk_pct < 20: cat='Low'
    elif risk_pct < 50: cat='Moderate'
    else: cat='High'
    return risk_pct, cat

# local AI explanation (simple)
def generate_ai_explanation(feats, prob, risk_pct, risk_cat, level, lang='English'):
    # short templates by stress level
    texts = {
        'English':{
            'Low':("You're at low stress — good job.",["Keep moving, stay hydrated.","Take short breaks.","Maintain regular sleep."],"See doctor if severe symptoms."),
            'Moderate':("Stress is moderate — small steps help.",["Try 2-min breathing.","Reduce caffeine.","Take a short walk."],"See doctor if persistent."),
            'High':("High stress detected — act now.",["Stop activity, breathe.","Avoid stimulants.","Seek medical advice if needed."],"If chest pain or fainting, seek emergency care.")
        },
        'Hindi':{
            'Low':("आपका तनाव कम है।",["हल्का व्यायाम, पानी पियें","छोटे ब्रेक लें","नियमित नींद रखें"],"गंभीर लक्षण होने पर डॉक्टर से संपर्क करें।"),
            'Moderate':("तनाव मध्यम है — कुछ कदम मदद करेंगे।",["2 मिनट की श्वास लें","कैफीन कम करें","थोड़ी सैर करें"],"लक्षण बने रहें तो डॉक्टर से मिलें।"),
            'High':("उच्च तनाव का संकेत — तुरंत ध्यान दें।",["आराम व धीमी साँस लें","उत्तेजक पदार्थ न लें","चिकित्सकीय सलाह लें"],"यदि सीने में दर्द हो तो तुरंत मदद लें।")
        },
        'Marathi':{
            'Low':("तणाव कमी आहे — छान काम.",["हलके व्यायाम करा, पाणी घ्या","लहान ब्रेक घ्या","नियमित झोप ठेवा"],"गंभीर लक्षण असल्यास डॉक्टरांकडे जा."),
            'Moderate':("तणाव मध्यम आहे — थोडे बदल फायदेशीर.",["2 मिनिटे श्वास घेणे","कॅफीन कमी करा","थोडी चाल करा"],"लक्षण कायम राहिल्यास डॉक्टरांचा सल्ला घ्या."),
            'High':("उच्च तणाव दिसतो — कृती करा.",["आराम करा व श्वास घ्या","उत्तेजक पदार्थ टाळा","वैद्यकीय सल्ला घ्या"],"छातीत वेदना असल्यास तातडीने मदत घ्या.")
        }
    }
    langset = texts.get(lang, texts['English'])
    one, recs, when = langset.get(level, langset['Moderate'])
    out = one + "\n\nRecommendations:\n- " + "\n- ".join(recs) + "\n\nWhen to seek care: " + when
    return out

# ------------------ Multi-user login (simple) ------------------
def ensure_users_file():
    if not USERS_FILE.exists():
        users = {'admin':'admin'}
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
ensure_users_file()

def check_login(username, password):
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    except Exception:
        users = {}
    return users.get(username) == password

def register_user(username, password):
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    except Exception:
        users = {}
    users[username] = password
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# ------------------ HISTORY persistence ------------------
def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []
def save_history(hist):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(hist, f)
    except Exception:
        pass

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = load_history()

# ------------------ UI CSS for butterfly and animated cards ------------------
st.markdown("""
<style>
.logo-box { width:72px; height:72px; border-radius:12px; background:linear-gradient(135deg,#ff4d6d,#ff7a59); display:flex; align-items:center; justify-content:center; color:white; font-weight:700; font-size:28px; }
.butterfly { width:120px; height:80px; display:inline-block; }
.heart-beat { animation: beat 1s infinite; transform-origin: center; }
@keyframes beat { 0%{ transform:scale(1);} 50%{ transform:scale(1.08);} 100%{ transform:scale(1);} }
.card-animated { border-radius:12px; padding:12px; background: linear-gradient(135deg,#fff,#fff); box-shadow: 0 8px 30px rgba(200,30,50,0.06); }
</style>
""", unsafe_allow_html=True)

# ------------------ Header & Login ------------------
col1, col2 = st.columns([0.8,4])
with col1:
    if (ASSETS_DIR/'logo.png').exists():
        st.image(str(ASSETS_DIR/'logo.png'), width=72)
    else:
        st.markdown('<div class="logo-box">AH</div>', unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color:#c41b23;margin:0'>AI Heart Rate & Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#64748b'>Heart Rate and Stress Monitoring System for Early Heart Attack Risk Prediction</div>", unsafe_allow_html=True)

# Sidebar login or main controls
with st.sidebar:
    st.header("Account")
    if not st.session_state['logged_in']:
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if check_login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['user'] = username
                st.success("Logged in")
            else:
                st.error("Invalid credentials")
        st.write("Or register:")
        newu = st.text_input("New username")
        newp = st.text_input("New password", type='password')
        if st.button("Register"):
            if newu and newp:
                register_user(newu, newp)
                st.success("Registered — you can now login")
            else:
                st.error("Provide username and password")
        st.stop()
    else:
        st.markdown(f"**User:** {st.session_state['user']}")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['user'] = None
            st.experimental_rerun()

    st.markdown("---")
    st.header("Input & Settings")
    method = st.radio("Method", ["Manual Entry","Upload Video","Upload Image","Webcam Image","Webcam Video"])
    max_seconds = st.slider("Max seconds for video", 6, 20, 10)
    rec_seconds = st.slider("Webcam record sec", 4, 12, 8)
    show_shap = st.checkbox("Show SHAP explainability", value=True)
    language = st.selectbox("Language", ["English","Hindi","Marathi"])
    st.markdown("---")
    st.write("Note: This demo stores credentials locally (not secure). For production, use secure auth.")

# ------------------ Main UI ------------------
left, right = st.columns([1,1.6])
with left:
    st.markdown('<div class="card-animated">', unsafe_allow_html=True)
    st.subheader("Input")
    features = None
    uploaded_temp = None

    if method == 'Manual Entry':
        vals = {}
        cols = st.columns(2)
        for i,f in enumerate(FEATURE_ORDER):
            with cols[i%2]:
                default = 75.0 if f=='mean_hr' else 20.0
                vals[f] = st.number_input(f, value=float(default))
        if st.button("Predict"):
            features = vals

    elif method == 'Upload Image':
        up = st.file_uploader("Image", type=['jpg','jpeg','png'])
        if up:
            st.image(up, width=320)
            if st.button("Predict"):
                img = Image.open(up).convert('RGB')
                arr = np.array(img); mean_g = float(np.mean(arr[:,:,1]))
                features = {'mean_hr':72 + (128-mean_g)/18.0, 'rmssd':28.0, 'pnn50':5.0, 'sd1':12.0, 'sd2':24.0, 'lf_hf':0.8, 'rr_mean':800.0, 'rr_std':40.0}

    elif method == 'Upload Video':
        vid = st.file_uploader("Video (mp4,mov,avi)", type=['mp4','mov','avi'])
        if vid:
            st.video(vid)
            if st.button("Predict"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4'); tmp.write(vid.read()); tmp.flush(); tmp.close()
                uploaded_temp = tmp.name
                try:
                    sig, fps = extract_mean_green_signal_from_video_file(tmp.name, max_seconds=max_seconds)
                    if sig is None:
                        st.error('No reliable signal')
                    else:
                        mean_hr, hr_series = get_hr_from_signal(sig, fps)
                        if hr_series is None:
                            st.error('No peaks detected')
                        else:
                            rr_ms = (60000.0/np.array(hr_series))
                            feats = compute_hrv_from_rr(rr_ms, hr_series)
                            features = feats
                            # store waveform for plotting
                            st.session_state['last_waveform'] = {'sig': sig.tolist(), 'fs': float(fps)}
                except Exception as e:
                    st.error(f'Processing failed: {e}')
                finally:
                    try: os.unlink(tmp.name)
                    except: pass

    elif method == 'Webcam Image':
        cam = st.camera_input("Capture image")
        if cam:
            st.image(cam, width=320)
            if st.button("Predict"):
                img = Image.open(cam).convert('RGB'); arr = np.array(img); mean_g = float(np.mean(arr[:,:,1]))
                features = {'mean_hr':72 + (128-mean_g)/18.0, 'rmssd':28.0, 'pnn50':5.0, 'sd1':12.0, 'sd2':24.0, 'lf_hf':0.8, 'rr_mean':800.0, 'rr_std':40.0}

    elif method == 'Webcam Video':
        st.write('Click Start Webcam Recording (local only)')
        if st.button('Start Webcam Recording'):
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4'); tmpf.close()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error('Cannot access webcam')
            else:
                fps = 20.0; ret, frame = cap.read()
                if not ret: st.error('Cannot read webcam')
                else:
                    h,w = frame.shape[:2]
                    out = cv2.VideoWriter(tmpf.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                    t0 = time.time(); progress = st.progress(0)
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        out.write(frame)
                        elapsed = time.time()-t0
                        progress.progress(min(100, int((elapsed/rec_seconds)*100)))
                        if elapsed >= rec_seconds: break
                    out.release(); cap.release()
                    st.success('Recording finished')
                    try:
                        sig, fps = extract_mean_green_signal_from_video_file(tmpf.name, max_seconds=rec_seconds)
                        if sig is None: st.error('No reliable signal')
                        else:
                            mean_hr, hr_series = get_hr_from_signal(sig, fps)
                            if hr_series is None: st.error('No peaks detected')
                            else:
                                rr_ms = (60000.0/np.array(hr_series))
                                feats = compute_hrv_from_rr(rr_ms, hr_series)
                                features = feats
                                st.session_state['last_waveform'] = {'sig': sig.tolist(), 'fs': float(fps)}
                    except Exception as e:
                        st.error(f'Processing failed: {e}')
                    finally:
                        try: os.unlink(tmpf.name)
                        except: pass

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card-animated">', unsafe_allow_html=True)
    st.subheader('Live Dashboard')

    # Butterfly-style indicator (SVG + CSS animation)
    st.markdown('''
    <div style="display:flex; gap:16px; align-items:center;">
      <div style="text-align:center;">
        <svg class="butterfly" viewBox="0 0 120 80">
          <g class="heart-beat">
            <ellipse cx="30" cy="40" rx="16" ry="22" fill="#ff7a59" opacity="0.9"/>
            <ellipse cx="90" cy="40" rx="16" ry="22" fill="#ff4d6d" opacity="0.9"/>
            <circle cx="60" cy="40" r="10" fill="#fff" opacity="0.06"/>
          </g>
        </svg>
        <div style="font-weight:700;color:#c41b23">Live Indicator</div>
      </div>
      <div style="flex:1">
        <div id="wave-placeholder">Waveform and metrics below</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    gauge_ph = st.empty()
    trend_ph = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

 # Colorful primary buttons
.stButton > button {
    background: linear-gradient(90deg, #ff4d6d, #ff7a59);
    color: white !important;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight: 600;
    border: none;
    transition: 0.2s;
}
.stButton > button:hover {
    transform: scale(1.05);
}

/* Secondary blue-green buttons */
.stButton.secondary > button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white !important;
}

# ------------------ After prediction show results ------------------
if features is not None:
    feats = {k: float(features.get(k, math.nan)) if features.get(k, None) is not None else float('nan') for k in FEATURE_ORDER}
    prob, parts = predict_ensemble(feats)
    stress_label = categorize_stress(prob)
    hr_label = categorize_hr(feats.get('mean_hr',0.0))
    risk_pct, risk_cat = heart_attack_risk_heuristic(prob, feats.get('mean_hr',0.0))
    sentence = f"HR ≈ {feats.get('mean_hr',0):.0f} bpm ({hr_label}). Stress: {stress_label} ({prob:.2f}). Heart-attack estimate: {risk_pct:.0f}% ({risk_cat})."

    # gauge (plotly) if available
    if go is not None:
        fig = go.Figure(go.Indicator(mode='gauge+number', value=feats.get('mean_hr',60),
                                    gauge={'axis':{'range':[30,180]}, 'bar':{'color':'#ff4d6d'}},
                                    title={'text':'<b>Heart Rate (bpm)</b>'}))
        fig.update_layout(height=260, margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor='white', font_color='#0b1721')
        gauge_ph.plotly_chart(fig, use_container_width=True)
    else:
        gauge_ph.info(f"HR: {feats.get('mean_hr',0):.0f} bpm")

    # append to history
    hist = st.session_state.get('history', [])
    hist.append({'t': time.time(), 'hr': float(feats.get('mean_hr',0.0)), 'stress': float(prob)})
    hist = hist[-200:]
    st.session_state['history'] = hist
    # persist
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(hist, f)
    except Exception:
        pass

    # Trend graph for HR + Stress
    df_hist = pd.DataFrame(hist)
    if not df_hist.empty and go is not None:
        df_hist['ts'] = pd.to_datetime(df_hist['t'], unit='s')
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=df_hist['ts'], y=df_hist['hr'], name='HR (bpm)', mode='lines+markers'), secondary_y=False)
        fig2.add_trace(go.Scatter(x=df_hist['ts'], y=df_hist['stress'], name='Stress (prob)', mode='lines+markers'), secondary_y=True)
        fig2.update_layout(height=300, margin=dict(t=10,b=0,l=0,r=0), legend=dict(orientation='h'))
        fig2.update_yaxes(title_text='HR', secondary_y=False)
        fig2.update_yaxes(title_text='Stress (0-1)', secondary_y=True, range=[0,1])
        trend_ph.plotly_chart(fig2, use_container_width=True)
    elif not df_hist.empty:
        st.line_chart(pd.DataFrame({'HR':df_hist['hr'], 'Stress':df_hist['stress']}))

    # Waveform plot (use last_waveform if available)
    wf = st.session_state.get('last_waveform', None)
    if wf is not None and go is not None:
        sig = np.array(wf['sig'])
        fs = float(wf['fs'])
        t = np.arange(sig.shape[0]) / fs
        figw = go.Figure(data=go.Scatter(x=t, y=sig, mode='lines', name='rPPG signal'))
        figw.update_layout(height=250, title='HRV waveform (rPPG signal)', margin=dict(t=20,b=0,l=0,r=0))
        st.plotly_chart(figw, use_container_width=True)
    elif wf is not None:
        st.line_chart(wf['sig'])

# ----------SHAP Explainability (Fixed for MLP / single-output models)----------
if show_shap and shap is not None and (mlp is not None):
    st.markdown("<div class='card' style='margin-top:12px'>", unsafe_allow_html=True)
    st.subheader("SHAP Contributions")

    try:
        # Background dataset = mean vector (prevents dimension errors)
        background = np.zeros((20, len(FEATURE_ORDER)))
        explainer = shap.Explainer(mlp, background)

        sample = vectorize_features(feats)
        shap_values = explainer(sample)

        # MLP gives shap.values shape: (1, num_features)
        shap_vals = shap_values.values[0]

        sh_df = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "SHAP Value": shap_vals
        })

        st.table(sh_df)

    except Exception as e:
        st.info(f"SHAP not available: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # AI explanation (level-specific)
    st.markdown('---')
    st.subheader('AI Explanation (local)')
    expl = generate_ai_explanation(feats, prob, risk_pct, risk_cat, stress_label, lang=language)
    st.code(expl)

st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
