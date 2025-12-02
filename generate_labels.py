import os
import csv

# Output file
csv_file = "labels.csv"
rows = [["filename", "label"]]

# ------------ UBFC (Relaxed by default) ------------
for f in os.listdir("datasets/ubfc"):
    if f.lower().endswith((".mp4", ".avi", ".mov")):
        rows.append([f, 0])   # UBFC videos = Relaxed baseline

# ------------ RAVDESS: Stress vs Relax Mapping ------------

# Emotions that map to RELAX (0)
relaxed_keywords = ["neutral", "calm"]

# Emotions that map to STRESS (1)
stress_keywords = ["angry", "fear", "disgust", "surprise", "sad"]

for f in os.listdir("datasets/ravdess"):
    fname = f.lower()

    if fname.endswith((".mp4", ".avi")):

        # Default stress (to catch unknown labels)
        label = 1

        for word in relaxed_keywords:
            if word in fname:
                label = 0

        for word in stress_keywords:
            if word in fname:
                label = 1

        rows.append([f, label])

# Save CSV
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print("labels.csv created successfully!")
