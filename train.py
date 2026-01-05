"""
Medical Image Segmentation Training with Live Visualization
===========================================================
Trains a U-Net on BraTS data and saves visual comparisons every epoch.

Author: Senior CV Engineer
Date: 2026
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

# Set random seeds
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
# DATASET & TRANSFORMS
# ============================================================================

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = str(self.image_paths[idx])
        if img_path.endswith('.npy'):
            image = np.load(img_path)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask
        mask_path = str(self.mask_paths[idx])
        if mask_path.endswith('.npy'):
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        mask = mask.astype(np.int64)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

def get_train_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # Fix: Add **kwargs to absorb extra args
        A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])

def get_val_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])

# ============================================================================
# LOSS FUNCTIONS (Weighted)
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_classes=4):
        predictions = torch.softmax(predictions, dim=1)
        targets = targets.long() # Fix type
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        
        # FIX: Add Class Weights to solve "All Black" issue
        # Background (0.1) vs Tumor (1.0)
        class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0]).float()
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
            
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        targets = targets.long()
        dice = self.dice_loss(predictions, targets, self.num_classes)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce

# ============================================================================
# VISUALIZATION HELPER (The New Part)
# ============================================================================

def visualize_training_sample(model, dataset, indices, epoch, save_dir, device):
    """Saves a comparison plot for fixed samples during training"""
    model.eval()
    fig, axes = plt.subplots(len(indices), 4, figsize=(15, 4 * len(indices)))
    
    # Handle single sample case
    if len(indices) == 1: axes = axes[None, :]
    
    for row_idx, sample_idx in enumerate(indices):
        # Load sample
        image, mask = dataset[sample_idx]
        input_tensor = image.unsqueeze(0).to(device) # (1, 3, H, W)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Convert tensors to numpy for plotting
        img_np = image[0].cpu().numpy() # Take first channel for display
        mask_np = mask.cpu().numpy()
        
        # Create RGB masks
        def to_rgb(m):
            h, w = m.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(4): rgb[m == i] = COLORS[i]
            return rgb

        # Plot
        # 1. Input
        axes[row_idx, 0].imshow(img_np, cmap='gray')
        axes[row_idx, 0].set_title(f"Epoch {epoch} | Input")
        axes[row_idx, 0].axis('off')
        
        # 2. GT
        axes[row_idx, 1].imshow(to_rgb(mask_np))
        axes[row_idx, 1].set_title("Ground Truth")
        axes[row_idx, 1].axis('off')
        
        # 3. Pred
        axes[row_idx, 2].imshow(to_rgb(pred_mask))
        axes[row_idx, 2].set_title("Prediction")
        axes[row_idx, 2].axis('off')
        
        # 4. Overlay
        img_rgb = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(img_rgb, 0.6, to_rgb(pred_mask), 0.4, 0)
        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title("Overlay")
        axes[row_idx, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'viz_epoch_{epoch:03d}.png'))
    plt.close()
    model.train()

# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_segmentation_model(
    image_dir='/kaggle/working/synthetic_t2',
    mask_dir='/kaggle/working/synthetic_masks',
    output_dir='/kaggle/working/outputs',
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4
):
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'training_visuals')
    os.makedirs(viz_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # 1. Prepare Data
    image_paths = sorted(list(Path(image_dir).glob('*')))
    mask_paths = sorted(list(Path(mask_dir).glob('*')))
    
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    train_dataset = BrainTumorDataset(train_img, train_mask, transform=get_train_transform())
    val_dataset = BrainTumorDataset(val_img, val_mask, transform=get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Pick 3 fixed samples for visualization
    viz_indices = [0, len(val_dataset)//2, len(val_dataset)-1]
    
    # 2. Setup Model
    print("🏗️ Building Model...")
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=4,
        activation=None
    ).to(device)
    
    criterion = CombinedLoss(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Training Loop
    best_loss = float('inf')
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Train Step
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"  Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- LIVE VISUALIZATION ---
        # Visualize the SAME 3 samples every epoch to see progress
        visualize_training_sample(model, val_dataset, viz_indices, epoch+1, viz_dir, device)
        # --------------------------
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
            print("  ✅ Saved Best Model")

if __name__ == "__main__":
    train_segmentation_model()