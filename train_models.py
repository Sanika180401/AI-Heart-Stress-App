import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from joblib import dump

# ---------------------------
# rPPG (CHROM) implementation (simple)
# ---------------------------
def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def chrom_extract(video_path, max_seconds=15, resize_width=320):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*max_seconds, fps*max_seconds))
    green = []
    red = []
    blue = []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # naive face-detection-free approach: take center region (works if face roughly centered)
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        wbox, hbox = int(w*0.3), int(h*0.4)
        x1, y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2, y2 = min(w, cx+wbox//2), min(h, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        # mean channels
        b_mean = np.mean(roi[:,:,0])
        g_mean = np.mean(roi[:,:,1])
        r_mean = np.mean(roi[:,:,2])
        blue.append(b_mean); green.append(g_mean); red.append(r_mean)
        frames += 1
    cap.release()
    rgb = np.vstack([red, green, blue]).T  # shape (n_frames, 3)
    if rgb.shape[0] < 30:
        return None, fps
    # CHROM: project normalized RGB to pulse signal
    # normalize per-channel
    rgb_norm = rgb / np.mean(rgb, axis=0)
    x1 = 3*rgb_norm[:,0] - 2*rgb_norm[:,1]
    x2 = 1.5*rgb_norm[:,0] + rgb_norm[:,1] - 1.5*rgb_norm[:,2]
    S = x1 - np.std(x1)/np.std(x2) * x2
    S = bandpass(S, fps)
    return S, fps

def peaks_and_rr(signal, fps):
    # detect peaks
    distance = int(0.4 * fps)
    peaks, _ = find_peaks(signal, distance=distance, prominence=np.std(signal)*0.3)
    if len(peaks) < 2:
        peaks, _ = find_peaks(signal, distance=distance, prominence=np.std(signal)*0.1)
    if len(peaks) < 2:
        return None, None
    times = peaks / fps
    rr_ms = np.diff(times) * 1000.0
    hr_series = 60000.0 / rr_ms
    return rr_ms, hr_series

# HRV metrics
def compute_hrv(rr_ms, hr_ser=None):
    import math
    features = {}
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    features['rmssd'] = float(np.sqrt(np.mean(diff**2)))
    features['pnn50'] = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = np.sqrt(np.var(diff) / 2.0)
    sd2 = np.sqrt(2*np.var(rr_ms) - np.var(diff)/2.0)
    features['sd1'] = float(sd1)
    features['sd2'] = float(sd2)
    features['rr_mean'] = float(np.mean(rr_ms))
    features['rr_std'] = float(np.std(rr_ms))
    features['mean_hr'] = float(np.mean(hr_ser)) if hr_ser is not None else float(60000.0/np.mean(rr_ms))
    # LF/HF via simple interpolation method
    try:
        fs_interp = 4.0
        times = np.cumsum(rr_ms)/1000.0
        t_interp = np.arange(0, times[-1], 1.0/fs_interp)
        inst_hr = 60000.0/rr_ms
        beat_times = times[:-1]
        interp = np.interp(t_interp, beat_times, inst_hr[:len(beat_times)])
        f, p = welch(detrend(interp), fs=fs_interp, nperseg=min(256, len(interp)))
        lf_mask = (f>=0.04) & (f<=0.15)
        hf_mask = (f>0.15) & (f<=0.4)
        lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
        hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
        features['lf_hf'] = float(lf/hf) if hf>0 else 0.0
    except Exception:
        features['lf_hf'] = 0.0
    return features

# ---------------------------
# Main dataset builder + training
# ---------------------------
def build_dataset_from_videos(videos_dir, labels_csv):
    labels_df = pd.read_csv(labels_csv)
    X = []
    y = []
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        fname = row['filename']
        label = int(row['label'])
        vid_path = os.path.join(videos_dir, fname)
        if not os.path.exists(vid_path):
            print("Missing:", vid_path)
            continue
        sig, fps = chrom_extract(vid_path)
        if sig is None:
            print("Failed rPPG:", fname)
            continue
        rr, hr_ser = peaks_and_rr(sig, fps)
        feats = compute_hrv(rr, hr_ser)
        if feats is None:
            print("Failed peaks:", fname)
            continue
        X.append([feats['mean_hr'], feats['rmssd'], feats['pnn50'], feats['sd1'], feats['sd2'], feats['lf_hf'], feats['rr_mean'], feats['rr_std']])
        y.append(label)
    cols = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
    return pd.DataFrame(X, columns=cols), np.array(y)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, required=True, help="Folder with mp4 videos")
    parser.add_argument("--labels_csv", type=str, required=True, help="CSV with filename,label")
    args = parser.parse_args()

    X_df, y = build_dataset_from_videos(args.videos_dir, args.labels_csv)
    print("Built dataset:", X_df.shape)
    # drop rows with NaN
    X_df = X_df.dropna()
    y = y[:len(X_df)]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df)

    # Train models
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400)
    rf = RandomForestClassifier(n_estimators=200)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200)

    mlp.fit(Xs, y)
    rf.fit(Xs, y)
    xgb.fit(Xs, y)

    # stacking
    meta_X = np.vstack([mlp.predict_proba(Xs)[:,1], rf.predict_proba(Xs)[:,1], xgb.predict_proba(Xs)[:,1]]).T
    stacker = LogisticRegression()
    stacker.fit(meta_X, y)

    os.makedirs("models", exist_ok=True)
    dump(mlp, "models/mlp.pkl")
    dump(rf, "models/rf.pkl")
    dump(xgb, "models/xgb.pkl")
    dump(stacker, "models/stacker.pkl")
    dump(scaler, "models/scaler.pkl")
    print("Saved models in models/")
