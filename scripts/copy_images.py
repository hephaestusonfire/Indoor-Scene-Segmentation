import os
import shutil
from tqdm import tqdm

def copy_images(mask_dir, src_img_dir, dst_img_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    for mask_file in tqdm(mask_files, desc=f"Copying to {dst_img_dir}"):
        image_file = mask_file.replace('.png', '.jpg')
        src_path = os.path.join(src_img_dir, image_file)
        dst_path = os.path.join(dst_img_dir, image_file)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ Missing image: {image_file}")

if __name__ == "__main__":
    copy_images(
        "IndoorSegmentation/data/train/masks",
        "IndoorSegmentation/data/ADEChallengeData2016/images/training",
        "IndoorSegmentation/data/train/images"
    )

    copy_images(
        "IndoorSegmentation/data/val/masks",
        "IndoorSegmentation/data/ADEChallengeData2016/images/validation",
        "IndoorSegmentation/data/val/images"
    )