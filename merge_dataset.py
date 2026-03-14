from huggingface_hub import snapshot_download
import os
import shutil

# Repo ID of DF40
REPO_ID = "pujanpaudel/deepfake_face_classification"

# Download the entire dataset snapshot (all files + structure)
dataset_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")

print("✅ Dataset downloaded at:", dataset_dir)

# Optional: Move dataset into a custom folder for training
target_dir = "./DF40_dataset"
if not os.path.exists(target_dir):
    shutil.copytree(dataset_dir, target_dir)

print("📂 Local copy created at:", target_dir)

# Now your folder structure will look like:
# DF40_dataset/
#   ├── train/
#   │     ├── real/
#   │     └── fake/
#   ├── validation/
#   │     ├── real/
#   │     └── fake/
#   └── test/
#         ├── real/
#         └── fake/
