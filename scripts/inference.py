import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.planenet import PlaneNet
from utils.dataset import ADE20KDataset

# Configuration
image_dir = "IndoorSegmentation/data/val/images"
mask_dir = "IndoorSegmentation/data/val/masks"
checkpoint_path = "planenet_wall_floor.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

# Load model
model = PlaneNet(num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def decode_segmentation(mask_tensor):
    """Convert class indices to RGB colors for visualization."""
    palette = {
        0: [0, 255, 0],    # Floor: green
        1: [0, 0, 255],    # Wall: blue
        255: [0, 0, 0]     # Ignore: black
    }
    mask_np = mask_tensor.cpu().numpy()
    color_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_id, color in palette.items():
        color_mask[mask_np == class_id] = color
    return color_mask

def visualize_prediction(image_path, mask_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).byte()

    # Decode for display
    pred_vis = decode_segmentation(pred_mask)
    gt_mask = np.array(mask.resize((128, 128), resample=Image.NEAREST))
    gt_vis = decode_segmentation(torch.from_numpy(gt_mask))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.resize((128, 128)))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_vis)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred_vis)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run on a few random examples that have masks
all_images = sorted(os.listdir(image_dir))
valid_image_mask_pairs = []

for fname in all_images:
    mask_name = fname.replace(".jpg", ".png").replace(".JPEG", ".png")
    mask_path = os.path.join(mask_dir, mask_name)
    if os.path.exists(mask_path):
        valid_image_mask_pairs.append((fname, mask_name))

# Randomly choose up to 5 pairs
sample_pairs = random.sample(valid_image_mask_pairs, min(5, len(valid_image_mask_pairs)))

for image_file, mask_file in sample_pairs:
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    print(f"🔍 Visualizing {image_file}")
    visualize_prediction(image_path, mask_path)
