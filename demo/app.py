import os
import sys
import random
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.planenet import PlaneNet
from utils.dataset import ADE20KDataset

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
image_size = (128, 128)

# Paths
val_image_dir = "IndoorSegmentation/data/val/images"
val_mask_dir = "IndoorSegmentation/data/val/masks"
checkpoint_path = "planenet_wall_floor.pth"

# Load model
@st.cache_resource
def load_model():
    model = PlaneNet(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize(image_size),
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

def predict(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).byte()
    return pred_mask

# Streamlit UI
st.set_page_config(page_title="Wall/Floor Segmentation", layout="wide")
st.title("🏠 Indoor Wall and Floor Segmentation")

uploaded_file = st.file_uploader("📷 Upload an image (or use the default)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    mask_path = None
else:
    # Load a random default image from val set
    image_files = sorted(os.listdir(val_image_dir))
    default_image_name = random.choice(image_files)
    image = Image.open(os.path.join(val_image_dir, default_image_name)).convert("RGB")
    mask_path = os.path.join(val_mask_dir, default_image_name.replace(".jpg", ".png").replace(".JPEG", ".png"))

# Run prediction
pred_mask = predict(image)
pred_vis = decode_segmentation(pred_mask)

# Try loading ground truth if available
gt_vis = None
if mask_path and os.path.exists(mask_path):
    gt_mask = Image.open(mask_path).convert("L").resize(image_size, resample=Image.NEAREST)
    gt_tensor = torch.from_numpy(np.array(gt_mask))
    gt_vis = decode_segmentation(gt_tensor)

# Display results
st.subheader("🔍 Segmentation Results")
col1, col2, col3 = st.columns(3)

with col1:
    st.image(image.resize(image_size), caption="Input Image", use_container_width=True)

with col2:
    if gt_vis is not None:
        st.image(gt_vis, caption="Ground Truth", use_container_width=True)
    else:
        st.info("Ground truth not available.")

with col3:
    st.image(pred_vis, caption="Prediction", use_container_width=True)
