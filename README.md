# Indoor Scene Segmentation with PlaneNet (Wall & Floor Detection)

This project uses a customized PlaneNet architecture to perform semantic segmentation of **walls** and **floors** from the ADE20K dataset. It supports training, validation, inference, and a user-friendly Streamlit demo.

---

##  Project Structure

```
IndoorSegmentation/
├── data/
│   ├── train/
│   └── val/
├── models/
│   └── planenet.py
├── scripts/
│   ├── train.py             # Main training script
│   ├── inference.py         # Predict and visualize segmentations
│   ├── preprocess.py        # Preprocess masks, filter classes, resize, remap
│   ├── download_dataset.py  # Downloads ADE20K from Kaggle
│   └── copy_images.py       # Organizes and copies RGB/mask files
├── utils/
│   ├── dataset.py           # ADE20K dataloader (wall/floor only)
│   └── metrics.py           # Evaluation metrics (IoU, Pixel Acc, etc.)
├── demo/
│   └── app.py               # Streamlit app interface
├── requirements.txt
└── README.md

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/indoor-scene-segmentation.git
cd indoor-scene-segmentation
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv planenet_env
source planenet_env/bin/activate     # Linux/macOS
planenet_env\Scripts\activate        # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup (ADE20K)

1. **Download the ADE20K dataset** using Kaggle (your account must have access):

```bash
python scripts/download_dataset.py
```

2. **Organize the data**:

```bash
python scripts/copy_images.py
```

This places the resized and labeled wall/floor images in:

```
IndoorSegmentation/data/train/images + masks
IndoorSegmentation/data/val/images + masks
```

---

## Training

Train PlaneNet on ADE20K (filtered for walls and floors):

```bash
python scripts/train.py
```

Model weights will be saved as `planenet_wall_floor.pth`.

---

## Inference

To visualize predictions on validation images:

```bash
python scripts/inference.py
```

This will display side-by-side:

* Original image
* Ground truth mask
* Predicted mask

---

## Streamlit Demo

Run an interactive web app for wall/floor segmentation on custom or default images:

```bash
streamlit run demo/app.py
```

You can upload your own image or preview a default ADE20K image with segmentation.

---

## Requirements

Dependencies are listed in `requirements.txt`, including:

* `torch`
* `torchvision`
* `streamlit`
* `numpy`, `matplotlib`, `Pillow`
* `tqdm`

Install via:

```bash
pip install -r requirements.txt
```

---

## Metrics

The model evaluates performance using:

* **Pixel Accuracy**
* **Mean Accuracy**
* **Mean IoU**
* **Frequency-Weighted IoU (FWIoU)**

Results are printed after each validation epoch.

---

## Visualization Color Legend

| Class  | Label ID | Color           |
| ------ | -------- | --------------- |
| Floor  | 1        | Green (0,255,0) |
| Wall   | 2        | Blue (0,0,255)  |
| Ignore | Others   | Black (0,0,0)   |

---

## Status

*  PlaneNet integrated and adapted for binary class segmentation
*  Mask preprocessing and class remapping
*  Streamlit demo for visualization
*  Modular dataset loader and metrics

---

##  Notes

* The ADE20K masks are remapped:

  * Floor → 0
  * Wall → 1
  * All others → 255 (ignored)

* Dataset is limited to **1500 random valid images**  for faster CPU training. You can increase this in `utils/dataset.py`.

---

