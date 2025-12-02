import kagglehub
import shutil
import os

# ---------- DOWNLOAD RAVDESS ----------
print("Downloading RAVDESS dataset...")
ravdess_path = kagglehub.dataset_download("orvile/ravdess-dataset")
print("Downloaded RAVDESS at:", ravdess_path)

# ---------- DOWNLOAD UBFC-rPPG ----------
print("Downloading UBFC-rPPG dataset...")
ubfc_path = kagglehub.dataset_download("malekdinarito/ubfc-rppg-dataset")
print("Downloaded UBFC-rPPG at:", ubfc_path)

# ---------- CREATE DATASET FOLDER ----------
os.makedirs("datasets/ravdess", exist_ok=True)
os.makedirs("datasets/ubfc", exist_ok=True)

# ---------- MOVE FILES ----------
def copy_all(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                full_src = os.path.join(root, file)
                full_dst = os.path.join(dst, file)
                shutil.copy(full_src, full_dst)
                print("Copied:", file)

copy_all(ravdess_path, "datasets/ravdess")
copy_all(ubfc_path, "datasets/ubfc")

print("\nDataset download and arrangement complete!")
