import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_mask(mask_path, save_path):
    mask = np.array(Image.open(mask_path))

    # New mask: 0 = background, 1 = wall, 2 = floor
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    new_mask[mask == 12] = 1  # wall → 1
    new_mask[mask == 4] = 2   # floor → 2

    Image.fromarray(new_mask).save(save_path)

def process_directory(annotation_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for filename in tqdm(os.listdir(annotation_dir)):
        if filename.endswith('.png'):
            src = os.path.join(annotation_dir, filename)
            dst = os.path.join(output_dir, filename)
            convert_mask(src, dst)
            count += 1
    return count

if __name__ == "__main__":
    print("📦 Processing 'training' set...")
    annotation_train = "IndoorSegmentation/data/ADEChallengeData2016/annotations/training"
    output_train = "IndoorSegmentation/data/train/masks"
    processed_train = process_directory(annotation_train, output_train)
    print(f"✅ Total processed in 'training': {processed_train}")

    print("📦 Processing 'validation' set...")
    annotation_val = "IndoorSegmentation/data/ADEChallengeData2016/annotations/validation"
    output_val = "IndoorSegmentation/data/val/masks"
    processed_val = process_directory(annotation_val, output_val)
    print(f"✅ Total processed in 'validation': {processed_val}")

