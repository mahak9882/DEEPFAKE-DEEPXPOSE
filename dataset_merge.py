

from datasets import load_dataset
import os
from PIL import Image
import shutil
import random

# =========================
# Setup folders
# =========================
root_dir = "deepfake_dataset"
real_dir = os.path.join(root_dir, "real")
fake_dir = os.path.join(root_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

def save_image(img_obj, dest_dir, filename):
    """Save HF Image object to dest_dir with given filename."""
    img = img_obj.convert("RGB")  # Ensure 3 channels
    img.save(os.path.join(dest_dir, filename))

# =========================
# 1. DF40 dataset
# =========================
print("Downloading DF40...")
from datasets import load_dataset, concatenate_datasets

df40_val = load_dataset("pujanpaudel/deepfake_face_classification", split="validation")
df40_test = load_dataset("pujanpaudel/deepfake_face_classification", split="test")

df40 = concatenate_datasets([df40_val, df40_test])
print(f"DF40 total images: {len(df40)}")



count_real = 0
count_fake = 0
for split in ["train", "validation", "test"]:
    for idx, row in enumerate(df40[split]):
        label = "real" if row["label"] == 0 else "fake"
        dest = real_dir if label == "real" else fake_dir
        save_image(row["image"], dest, f"df40_{split}_{idx}.jpg")
        if label == "real":
            count_real += 1
        else:
            count_fake += 1

print(f"DF40 → real: {count_real}, fake: {count_fake}")

# =========================
# 2. AI vs Deepfake vs Real
# =========================
print("Downloading AI-vs-Deepfake-vs-Real...")
ai_df_real = load_dataset("prithivMLmods/AI-vs-Deepfake-vs-Real")

for idx, row in enumerate(ai_df_real["train"]):
    label_id = row["label"]
    if label_id == 0:   # AI generated
        dest = fake_dir
    elif label_id == 1: # Deepfake
        dest = fake_dir
    else:               # Real
        dest = real_dir
    save_image(row["image"], dest, f"aivsdf_{idx}.jpg")

# =========================
# 3. FFHQ (subset)
# =========================
print("Downloading FFHQ thumbnails...")

ffhq_images = list(os.scandir("ffhq-dataset-master/thumbnails128x128"))
random.shuffle(ffhq_images)
ffhq_subset = ffhq_images[:5000]  # adjust if needed

for idx, img_entry in enumerate(ffhq_subset):
    shutil.copy(img_entry.path, os.path.join(real_dir, f"ffhq_{idx}.png"))

# =========================
# 4. Balance classes
# =========================
real_files = os.listdir(real_dir)
fake_files = os.listdir(fake_dir)

min_count = min(len(real_files), len(fake_files))
print(f"Balancing to {min_count} images per class...")

# Remove extra files to balance dataset
for folder, files in [(real_dir, real_files), (fake_dir, fake_files)]:
    if len(files) > min_count:
        to_remove = random.sample(files, len(files) - min_count)
        for f in to_remove:
            os.remove(os.path.join(folder, f))

print(f"✅ Dataset ready at '{root_dir}' → {len(os.listdir(real_dir))} real, {len(os.listdir(fake_dir))} fake")
