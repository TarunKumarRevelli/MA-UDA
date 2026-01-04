"""
Medical Image Segmentation Training with FDA-preprocessed Data
===============================================================
Train a U-Net segmentation model on synthetic T2 images created via FDA.

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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================================
# DATASET CLASS
# ============================================================================

class BrainTumorDataset(Dataset):
    """
    Dataset for Brain Tumor Segmentation
    Handles grayscale medical images and multi-class masks
    """
    
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to corresponding masks
            transform: Albumentations transform pipeline
        """
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
        
        # Normalize image to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Ensure mask is integer type
        mask = mask.astype(np.int64)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


# ============================================================================
# DATA AUGMENTATION (CORRECTED)
# ============================================================================

def get_train_transform(img_size=256):
    """Training augmentation pipeline"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # FIX: Add **kwargs to accept extra args passed by Albumentations
        A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])


def get_val_transform(img_size=256):
    """Validation transform (no augmentation)"""
    return A.Compose([
        A.Resize(img_size, img_size),
        # FIX: Add **kwargs here as well
        A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Handles class imbalance better than pure CrossEntropy
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets, num_classes=4):
        """
        Args:
            predictions: (B, C, H, W) logits
            targets: (B, H, W) class indices
        """
        # Convert predictions to probabilities
        predictions = torch.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - mean dice (loss)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + CrossEntropy Loss
    Balances pixel-wise accuracy with overlap-based metric
    """
    
    def __init__(self, num_classes=4, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets, self.num_classes)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce


# ============================================================================
# METRICS
# ============================================================================

def calculate_dice_score(predictions, targets, num_classes=4):
    """Calculate Dice score for evaluation"""
    predictions = torch.argmax(predictions, dim=1)
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = (predictions == c).float()
        target_c = (targets == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection / union).item())
    
    return np.mean(dice_scores)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice = calculate_dice_score(outputs, masks)
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice:.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            dice = calculate_dice_score(outputs, masks)
            running_loss += loss.item()
            running_dice += dice
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice:.4f}"})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_dice


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_segmentation_model(
    image_dir='/kaggle/working/synthetic_t2',
    mask_dir='/kaggle/working/synthetic_masks',
    output_dir='/kaggle/working/outputs',
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    img_size=256,
    num_classes=4,
    val_split=0.2
):
    """
    Main training pipeline
    
    Args:
        image_dir: Path to synthetic T2 images
        mask_dir: Path to segmentation masks
        output_dir: Where to save model checkpoints and logs
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        img_size: Input image size
        num_classes: Number of segmentation classes
        val_split: Validation split ratio
    """
    
    print("=" * 80)
    print("BRAIN TUMOR SEGMENTATION TRAINING")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Get file lists
    image_paths = sorted(list(Path(image_dir).glob('*')))
    mask_paths = sorted(list(Path(mask_dir).glob('*')))
    
    print(f"\n📁 Dataset:")
    print(f"  Images: {len(image_paths)}")
    print(f"  Masks: {len(mask_paths)}")
    
    if len(image_paths) != len(mask_paths):
        raise ValueError("❌ Mismatch between number of images and masks!")
    
    # Train/Val split
    train_img, val_img, train_mask, val_mask = train_test_split(
        image_paths, mask_paths, test_size=val_split, random_state=42
    )
    
    print(f"  Train samples: {len(train_img)}")
    print(f"  Val samples: {len(val_img)}")
    
    # Create datasets
    train_dataset = BrainTumorDataset(
        train_img, train_mask, transform=get_train_transform(img_size)
    )
    val_dataset = BrainTumorDataset(
        val_img, val_mask, transform=get_val_transform(img_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Create model
    print(f"\n🏗️  Building U-Net with ResNet34 encoder...")
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None  # We'll apply softmax in loss/inference
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(num_classes=num_classes, dice_weight=0.5, ce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    best_dice = 0.0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # Validate
        val_loss, val_dice = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f}   | Val Dice: {val_dice:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, checkpoint_path)
            print(f"  ✅ New best model saved! (Dice: {best_dice:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # Plot training curves
    print(f"\n📈 Plotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_dices, label='Train Dice', linewidth=2)
    axes[1].plot(val_dices, label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Training & Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print(f"\n✅ Training complete!")
    print(f"  Best Dice Score: {best_dice:.4f}")
    print(f"  Model saved to: {output_dir}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'image_dir': '/kaggle/working/synthetic_t2',
        'mask_dir': '/kaggle/working/synthetic_masks',
        'output_dir': '/kaggle/working/outputs',
        'num_epochs': 50,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'img_size': 256,
        'num_classes': 4,
        'val_split': 0.2
    }
    
    # Train model
    train_segmentation_model(**CONFIG)