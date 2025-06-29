import kagglehub
import os
import shutil

print("➡️ Starting dataset download...")

# ✅ Step 1: Download ADE20K via kagglehub
dataset_path = kagglehub.dataset_download("kallurivasanthsai/ade20k-2021-17-01")
print("✅ Downloaded to:", dataset_path)

# ✅ Step 2: Where we want it to go
target_dir = os.path.join("IndoorSegmentation", "data", "ADE20K_2021_17_01")
os.makedirs(target_dir, exist_ok=True)

# ✅ Step 3: Move files if not already present
if not os.listdir(target_dir):
    print("📂 Copying dataset into your project folder...")
    shutil.copytree(dataset_path, target_dir, dirs_exist_ok=True)
    print(f"✅ Dataset is ready at: {target_dir}")
else:
    print("⚠️ Dataset already exists at:", target_dir)
