"""
Medical Image Segmentation Training (With "Best Model" Visualization)
===================================================================
1. Trains U-Net with Class Weights (to fix "all black" predictions).
2. Saves visualizations automatically whenever the model improves.
"""

import os
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

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Visualization Colors
COLORS = np.array([
    [0, 0, 0],       # Background
    [255, 0, 0],     # Necrotic (Red)
    [0, 255, 0],     # Edema (Green)
    [0, 0, 255]      # Enhancing (Blue)
], dtype=np.uint8)

# ============================================================================
# DATASET & AUGMENTATION
# ============================================================================

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load Image
        img_path = str(self.image_paths[idx])
        if img_path.endswith('.npy'): image = np.load(img_path)
        else: image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load Mask
        mask_path = str(self.mask_paths[idx])
        if mask_path.endswith('.npy'): mask = np.load(mask_path)
        else: mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize
        if image.max() > 1.0: image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

def get_transforms(img_size=256, phase='train'):
    """Returns transform pipeline"""
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)), # Grayscale -> 3Ch
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
            ToTensorV2()
        ])

# ============================================================================
# WEIGHTED LOSS (Fixes "All Black" Prediction)
# ============================================================================

class WeightedCombinedLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Dice Loss (Overlap)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        
        # Weighted CrossEntropy (Pixel Accuracy)
        # Weight 0.1 for Background, 1.0 for Tumor classes
        weights = torch.tensor([0.1, 1.0, 1.0, 1.0]).float()
        if torch.cuda.is_available(): weights = weights.cuda()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, preds, targets):
        return 0.5 * self.dice_loss(preds, targets) + 0.5 * self.ce_loss(preds, targets)

# ============================================================================
# VISUALIZATION LOGIC
# ============================================================================

def save_visual_check(model, dataset, indices, filename, device):
    """Saves a comparison image (Input | GT | Pred)"""
    model.eval()
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1: axes = axes[None, :]
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(input_tensor)
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
        # Helper to colorize
        def colorize(m):
            rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
            for c in range(4): rgb[m == c] = COLORS[c]
            return rgb

        # Plot
        axes[i, 0].imshow(image[0].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(colorize(mask.cpu().numpy()))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorize(pred_mask))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    model.train() # Switch back to train mode

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    # Config
    img_dir = '/kaggle/working/synthetic_t2'
    mask_dir = '/kaggle/working/synthetic_masks'
    out_dir = '/kaggle/working/outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Data Setup
    images = sorted(list(Path(img_dir).glob('*')))
    masks = sorted(list(Path(mask_dir).glob('*')))
    train_x, val_x, train_y, val_y = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    train_ds = BrainTumorDataset(train_x, train_y, transform=get_transforms(phase='train'))
    val_ds = BrainTumorDataset(val_x, val_y, transform=get_transforms(phase='val'))
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)
    
    # Pick 3 fixed samples for live checking
    check_indices = [0, len(val_ds)//2, len(val_ds)-1]

    # Model Setup
    model = smp.Unet(encoder_name='resnet34', classes=4, activation=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedCombinedLoss()
    
    best_loss = float('inf')
    
    print("🎬 Starting Training...")
    
    for epoch in range(50): # 50 Epochs
        # 1. Train
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for imgs, msks in pbar:
            imgs, msks = imgs.to(device), msks.to(device).long()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 2. Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device).long()
                outputs = model(imgs)
                val_loss += criterion(outputs, msks).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"  📉 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # 3. Save Visualizations (Logic you asked for)
        # A. Save "Current Epoch" visualization (to see progress)
        save_visual_check(
            model, val_ds, check_indices, 
            filename=f"{out_dir}/epoch_{epoch+1}_check.png", 
            device=device
        )
        
        # B. If this is the BEST model so far, save a special "Best" visualization
        if val_loss < best_loss:
            best_loss = val_loss
            
            # Save Checkpoint
            torch.save(model.state_dict(), f"{out_dir}/best_model.pth")
            print("  ⭐ New Best Model! Saving checkpoint...")
            
            # Save "Best Model" Visualization
            save_visual_check(
                model, val_ds, check_indices, 
                filename=f"{out_dir}/BEST_MODEL_PREDICTIONS.png", 
                device=device
            )

if __name__ == "__main__":
    train()