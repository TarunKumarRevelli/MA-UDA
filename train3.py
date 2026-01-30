"""
Medical Image Segmentation Training (SOTA, ConvNeXt, Auto-Stop)
==============================================================
✔ U-Net++ with ConvNeXt encoder
✔ LayerNorm decoder (ConvNeXt-correct)
✔ AdamW optimizer
✔ Dice + Weighted CE Loss
✔ ReduceLROnPlateau + Early Stopping
✔ ImageNet normalization
✔ Best-model visualization
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# VISUALIZATION COLORS
# ============================================================================
COLORS = np.array([
    [0, 0, 0],       # Background
    [255, 0, 0],     # Necrotic
    [0, 255, 0],     # Edema
    [0, 0, 255]      # Enhancing
], dtype=np.uint8)

# ============================================================================
# DATASET
# ============================================================================
class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])

        image = np.load(img_path) if img_path.endswith(".npy") \
            else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask = np.load(mask_path) if mask_path.endswith(".npy") \
            else cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = image.astype(np.float32)
        if image.max() > 1:
            image /= 255.0

        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

# ============================================================================
# AUGMENTATIONS
# ============================================================================
def gray_to_rgb(image, **kwargs):
    return np.stack([image, image, image], axis=-1)

def get_transforms(img_size=256, phase="train"):
    if phase == "train":
        return A.Compose([
            A.Resize(img_size, img_size),

            A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

            # ✅ FIXED: new Albumentations API
            A.GaussNoise(std_range=(0.02, 0.10), p=0.3),

            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=15, p=0.5),

            # ✅ FIXED: multiprocessing-safe
            A.Lambda(image=gray_to_rgb),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Lambda(image=gray_to_rgb),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])


# ============================================================================
# LOSS
# ============================================================================
class WeightedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True
        )
        weights = torch.tensor([0.1, 1.0, 1.0, 1.0])
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, preds, targets):
        return 0.5 * self.dice(preds, targets) + 0.5 * self.ce(preds, targets)

# ============================================================================
# VISUAL CHECK
# ============================================================================
def save_visual_check(model, dataset, indices, filename, device):
    model.eval()
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    def colorize(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for c in range(4):
            rgb[mask == c] = COLORS[c]
        return rgb

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, gt = dataset[idx]
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            axes[i, 0].imshow(img[0].cpu(), cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 1].imshow(colorize(gt.cpu().numpy()))
            axes[i, 1].set_title("GT")
            axes[i, 2].imshow(colorize(pred))
            axes[i, 2].set_title("Prediction")

            for j in range(3):
                axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    model.train()

# ============================================================================
# TRAINING
# ============================================================================
def train(resume_checkpoint=None, epochs=1000, batch_size=8, patience=15):
    out_dir = "/kaggle/working/outputs_unetpp"
    img_dir = "/kaggle/working/synthetic_t2_cyclegan"
    mask_dir = "/kaggle/working/synthetic_masks_cyclegan"

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = sorted(Path(img_dir).glob("*"))
    masks = sorted(Path(mask_dir).glob("*"))

    train_x, val_x, train_y, val_y = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    train_ds = BrainTumorDataset(train_x, train_y, get_transforms(phase="train"))
    val_ds = BrainTumorDataset(val_x, val_y, get_transforms(phase="val"))

    train_loader = DataLoader(train_ds, batch_size, True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size, False, num_workers=2)

    # 🔥 SOTA MODEL
    model = smp.Unet(encoder_name="mit_b5",encoder_weights="imagenet",classes=4,activation=None).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = WeightedCombinedLoss().to(device)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 0

    print("🚀 Training started")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0

        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                val_loss += criterion(model(imgs), msks).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"📉 Train {train_loss:.4f} | Val {val_loss:.4f}")

        torch.save(model.state_dict(), f"{out_dir}/latest.pth")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{out_dir}/best.pth")
            save_visual_check(
                model, val_ds,
                [0, len(val_ds)//2, len(val_ds)-1],
                f"{out_dir}/best_preds.png", device
            )
            print("⭐ New best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("🛑 Early stopping triggered")
            break

# ============================================================================
# ENTRY
# ============================================================================
if __name__ == "__main__":
    train()
