import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

class ADE20KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128), max_samples=1500):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        image_files = sorted(os.listdir(image_dir))
        mask_files = sorted(os.listdir(mask_dir))

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST)
        ])

        all_valid_pairs = []
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            try:
                mask = Image.open(mask_path).convert("L")
                mask_resized = self.mask_transform(mask)
                mask_np = np.array(mask_resized)

                # Accept if any of class 1 (floor) or 2 (wall) exists
                if 1 in mask_np or 2 in mask_np:
                    all_valid_pairs.append((img_file, mask_file))
            except Exception as e:
                print(f"Skipping corrupt file: {mask_file} ({e})")
                continue

        # Random selection of up to max_samples
        self.image_mask_pairs = random.sample(all_valid_pairs, min(max_samples, len(all_valid_pairs)))

        print(f"✅ Using {len(self.image_mask_pairs)} valid image/mask pairs.")

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.image_mask_pairs[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        # Remap: floor (1) → 0, wall (2) → 1, ignore all others → 255
        remapped = torch.full_like(mask, 255)
        remapped[mask == 1] = 0
        remapped[mask == 2] = 1

        return image, remapped
