import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    XGBClassifier = None
    _HAS_XGB = False

# ---------------------------
# rPPG (CHROM) implementation (simple)
# ---------------------------
def bandpass(signal, fs, low=0.7, high=4.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def chrom_extract(video_path, max_seconds=15, resize_width=320):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(min(frame_count if frame_count>0 else fps*max_seconds, fps*max_seconds))
    red, green, blue = [], [], []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        # center ROI (naive)
        h, w = frame.shape[:2]
        cx, cy = w//2, h//2
        wbox, hbox = int(w*0.3), int(h*0.42)
        x1, y1 = max(0, cx-wbox//2), max(0, cy-hbox//2)
        x2, y2 = min(w, cx+wbox//2), min(h, cy+hbox//2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            frames += 1
            continue
        b_mean = float(np.mean(roi[:,:,0]))
        g_mean = float(np.mean(roi[:,:,1]))
        r_mean = float(np.mean(roi[:,:,2]))
        blue.append(b_mean); green.append(g_mean); red.append(r_mean)
        frames += 1
    cap.release()
    rgb = np.vstack([red, green, blue]).T
    if rgb.shape[0] < 30:
        return None, fps
    # CHROM (simple projection)
    rgb_norm = rgb / np.mean(rgb, axis=0)
    x1 = 3*rgb_norm[:,0] - 2*rgb_norm[:,1]
    x2 = 1.5*rgb_norm[:,0] + rgb_norm[:,1] - 1.5*rgb_norm[:,2]
    # avoid division by zero
    s2_std = np.std(x2) if np.std(x2) != 0 else 1.0
    S = x1 - (np.std(x1)/s2_std) * x2
    try:
        S = bandpass(S, fps)
    except Exception:
        pass
    return S, fps

def peaks_and_rr(signal, fps):
    if signal is None or len(signal) < 10:
        return None, None
    distance = max(1, int(0.4 * fps))
    peaks, _ = find_peaks(signal, distance=distance, prominence=np.std(signal)*0.25)
    if len(peaks) < 2:
        peaks, _ = find_peaks(signal, distance=distance, prominence=np.std(signal)*0.08)
    if len(peaks) < 2:
        return None, None
    times = peaks / float(fps)
    rr_ms = np.diff(times) * 1000.0
    hr_series = 60000.0 / rr_ms
    return rr_ms, hr_series

def compute_hrv(rr_ms, hr_ser=None):
    if rr_ms is None or len(rr_ms) < 2:
        return None
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2)))
    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100.0)
    sd1 = float(np.sqrt(np.var(diff) / 2.0))
    sd2 = float(np.sqrt(max(0.0, 2*np.var(rr_ms) - np.var(diff)/2.0)))
    rr_mean = float(np.mean(rr_ms))
    rr_std = float(np.std(rr_ms))
    mean_hr = float(np.mean(hr_ser)) if hr_ser is not None else float(60000.0/np.mean(rr_ms))
    # LF/HF estimation (may be noisy for short signals)
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
            lf_mask = (f>=0.04) & (f<=0.15)
            hf_mask = (f>0.15) & (f<=0.4)
            lf = np.trapz(p[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0.0
            hf = np.trapz(p[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0.0
            lf_hf = float(lf/hf) if hf > 0 else 0.0
    except Exception:
        lf_hf = 0.0
    return {
        "mean_hr": mean_hr,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "sd1": sd1,
        "sd2": sd2,
        "lf_hf": lf_hf,
        "rr_mean": rr_mean,
        "rr_std": rr_std
    }

# ---------------------------
# Dataset builder + training
# ---------------------------
def build_dataset_from_videos(videos_dir, labels_csv):
    df_labels = pd.read_csv(labels_csv)
    X_rows = []
    y_rows = []
    print(f"[INFO] Found {len(df_labels)} labeled items in {labels_csv}")
    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Processing videos"):
        fname = str(row['filename'])
        lbl = int(row['label'])
        vid_path = os.path.join(videos_dir, fname)
        if not os.path.exists(vid_path):
            print(f"[WARN] Missing: {vid_path}")
            continue
        sig, fps = chrom_extract(vid_path)
        if sig is None:
            print(f"[WARN] rPPG failed: {fname}")
            continue
        rr_ms, hr_ser = peaks_and_rr(sig, fps)
        feats = compute_hrv(rr_ms, hr_ser)
        if feats is None:
            print(f"[WARN] Peak extraction failed: {fname}")
            continue
        X_rows.append([feats['mean_hr'], feats['rmssd'], feats['pnn50'],
                       feats['sd1'], feats['sd2'], feats['lf_hf'],
                       feats['rr_mean'], feats['rr_std']])
        y_rows.append(lbl)
    cols = ['mean_hr','rmssd','pnn50','sd1','sd2','lf_hf','rr_mean','rr_std']
    if len(X_rows) == 0:
        return pd.DataFrame(columns=cols), np.array([], dtype=int)
    X_df = pd.DataFrame(X_rows, columns=cols)
    y_arr = np.array(y_rows, dtype=int)
    return X_df, y_arr

def train_and_save(X_df, y_arr, out_dir="models"):
    if X_df.shape[0] == 0 or len(y_arr) == 0:
        raise ValueError("Empty dataset. Nothing to train.")
    # drop NA rows safely and align labels
    notna_mask = ~X_df.isna().any(axis=1)
    X_clean = X_df[notna_mask].reset_index(drop=True)
    y_clean = y_arr[notna_mask.values]
    print(f"[INFO] Dataset after cleaning: {X_clean.shape}, labels: {y_clean.shape}")
    scaler = StandardScaler().fit(X_clean)
    Xs = scaler.transform(X_clean)

    # Train base models
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    mlp.fit(Xs, y_clean)
    rf.fit(Xs, y_clean)

    xgb_model = None
    if _HAS_XGB:
        try:
            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42)
            xgb_model.fit(Xs, y_clean)
            print("[INFO] XGBoost trained.")
        except Exception as e:
            print(f"[WARN] XGBoost failed to train: {e}")
            xgb_model = None

    # stacking (meta)
    preds = [mlp.predict_proba(Xs)[:,1], rf.predict_proba(Xs)[:,1]]
    if xgb_model is not None:
        preds.append(xgb_model.predict_proba(Xs)[:,1])
    meta_X = np.vstack(preds).T
    stacker = LogisticRegression(max_iter=400)
    stacker.fit(meta_X, y_clean)

    # save
    os.makedirs(out_dir, exist_ok=True)
    dump(mlp, os.path.join(out_dir, "mlp.pkl"))
    dump(rf, os.path.join(out_dir, "rf.pkl"))
    if xgb_model is not None:
        dump(xgb_model, os.path.join(out_dir, "xgb.pkl"))
    dump(stacker, os.path.join(out_dir, "stacker.pkl"))
    dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print(f"[INFO] Models saved to {out_dir}/")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str, default="datasets", help="Folder with mp4 videos")
    parser.add_argument("--labels_csv", type=str, default="labels.csv", help="CSV with filename,label")
    parser.add_argument("--out_dir", type=str, default="models", help="Where to save models")
    args = parser.parse_args()

    X_df, y_arr = build_dataset_from_videos(args.videos_dir, args.labels_csv)
    print(f"[INFO] Built dataset: {X_df.shape}")
    if X_df.shape[0] == 0:
        print("[ERROR] No valid training data found. Exiting.")
        raise SystemExit(1)
    train_and_save(X_df, y_arr, out_dir=args.out_dir)
