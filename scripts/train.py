import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Fix imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.dataset import ADE20KDataset
from utils.metrics import SegmentationMetrics
from models.planenet import PlaneNet

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
train_image_dir = "IndoorSegmentation/data/train/images"
train_mask_dir = "IndoorSegmentation/data/train/masks"
val_image_dir = "IndoorSegmentation/data/val/images"
val_mask_dir = "IndoorSegmentation/data/val/masks"

# Datasets & Loaders
train_dataset = ADE20KDataset(train_image_dir, train_mask_dir)
val_dataset = ADE20KDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# Model
model = PlaneNet(num_classes=2).to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
metrics = SegmentationMetrics(num_classes=2)

for epoch in range(num_epochs):
    print(f"\n🚀 Epoch {epoch+1}/{num_epochs}")

    # ---------- Training ----------
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"[Training] Epoch {epoch+1}")):
        images, masks = images.to(device), masks.to(device)

        if torch.all(masks == 255):
            continue  # Skip batch if all pixels are ignore index

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        if torch.isnan(loss):
            print(f"⚠️ Skipping batch {i+1} due to NaN loss")
            continue

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        valid_batches += 1

        if (i + 1) % 100 == 0:
            print(f"   Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")

    if valid_batches == 0:
        print("❌ No valid batches in training set this epoch.")
        continue

    avg_loss = running_loss / valid_batches
    print(f"✅ Training Loss: {avg_loss:.4f}")

    # ---------- Validation ----------
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"[Validation] Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Convert to CPU and NumPy arrays for metric calculations
            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            metrics.update(preds_np, masks_np)

    print("📊 Validation Metrics:")
    metrics.print_scores()

# Save the model
torch.save(model.state_dict(), "planenet_wall_floor.pth")
print("\n📂 Model saved as planenet_wall_floor.pth")
