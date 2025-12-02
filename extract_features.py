import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import iqr
import os

mp_face = mp.solutions.face_mesh

def butter_bandpass_filter(data, low, high, fs, order=5):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def extract_ppg_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    green_sig = []

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as fm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            result = fm.process(rgb)

            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0]

                forehead = []
                for i in [10, 109, 338, 297, 332, 284]:
                    x = int(lm.landmark[i].x * w)
                    y = int(lm.landmark[i].y * h)
                    forehead.append(rgb[y, x, 1])

                green_sig.append(np.mean(forehead))

    cap.release()
    return np.array(green_sig)

def compute_hrv_features(ppg, fps=30):
    if len(ppg) < fps * 5:
        return None  # too short

    ppg_f = butter_bandpass_filter(ppg, 0.7, 3.0, fps)

    peaks, _ = find_peaks(ppg_f, distance=fps*0.5)

    if len(peaks) < 3:
        return None

    rr = np.diff(peaks) / fps * 1000  # ms

    mean_hr = 60000 / np.mean(rr)
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))

    sd1 = np.sqrt(np.var((rr[:-1] - rr[1:]) / 2))
    sd2 = np.sqrt(2 * np.var(rr) - sd1**2)

    return {
        "mean_hr": mean_hr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "sd1": sd1,
        "sd2": sd2
    }

def process_dataset(dataset_dir, labels_csv):
    labels = pd.read_csv(labels_csv)
    rows = []

    for _, row in labels.iterrows():
        file = row["filename"]
        label = row["label"]
        video_path = os.path.join(dataset_dir, file)

        print("Processing:", video_path)
        ppg = extract_ppg_from_video(video_path)
        feats = compute_hrv_features(ppg)

        if feats:
            feats["label"] = label
            rows.append(feats)
        else:
            print("⚠ No HRV extracted from:", file)

    df = pd.DataFrame(rows)
    df.to_csv("dataset_features.csv", index=False)
    print("\n✔ Features saved to dataset_features.csv")

process_dataset("datasets", "labels.csv")
