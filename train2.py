"""
Medical Image Segmentation Training (Auto-Stop & LR Scheduler)
==============================================================
1. Early Stopping: Stops if no improvement for 'patience' epochs.
2. LR Scheduler: Lowers learning rate when loss plateaus.
3. Saves Best Model automatically.
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
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            
            # --- AGGRESSIVE CONTRAST AUGMENTATION (For CycleGAN) ---
            # Randomly brighten images to match Real T2 intensity
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            
            # Sharpen edges to fix any blur from generation
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            
            # Add noise to prevent overfitting to smooth synthetic textures
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            # -------------------------------------------------------

            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            
            # Ensure 3-channel output
            A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
            ToTensorV2()
        ])
    else:
        # Validation
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
            ToTensorV2()
        ])

# ============================================================================
# WEIGHTED LOSS
# ============================================================================

class WeightedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        # Class weights: Background (0.1), Tumor (1.0)
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
    
    # Helper to colorize
    def colorize(m):
        rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
        for c in range(4): rgb[m == c] = COLORS[c]
        return rgb
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(input_tensor)
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
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
    model.train()

# ============================================================================
# MAIN TRAINING LOOP (With Early Stopping & Scheduler)
# ============================================================================
def train(
    resume_checkpoint=None, 
    num_epochs=1000, 
    batch_size=8,
    patience=15
):
    # --- SAFETY FIX: New Folder for U-Net++ ---
    out_dir = '/kaggle/working/outputs_unet'
    img_dir = '/kaggle/working/synthetic_t2_cyclegan'
    mask_dir = '/kaggle/working/synthetic_masks_cyclegan'
    
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    print(f"📂 Saving results to: {out_dir}") # Confirm path

    # Data Setup
    images = sorted(list(Path(img_dir).glob('*')))
    masks = sorted(list(Path(mask_dir).glob('*')))
    
    if not images:
        print("❌ Error: No images found! Run the CycleGAN generation script first.")
        return

    train_x, val_x, train_y, val_y = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    # Ensure get_transforms and BrainTumorDataset are defined in your notebook
    train_ds = BrainTumorDataset(train_x, train_y, transform=get_transforms(phase='train'))
    val_ds = BrainTumorDataset(val_x, val_y, transform=get_transforms(phase='val'))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    check_indices = [0, len(val_ds)//2, len(val_ds)-1]

    # Model & Optimizer Setup
    print("🔥 Architecture: U-Net++ (Nested U-Net)")
    model = smp.Unet(encoder_name='resnet50', classes=4, activation=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = WeightedCombinedLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )

    # State Variables
    start_epoch = 0
    best_loss = float('inf')
    epochs_no_improve = 0 

    # Resume Logic
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"🔄 Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device) # weights_only=False is default usually safe
        
        # Robust loading
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint: best_loss = checkpoint['val_loss']
        print(f"  ✅ Resuming at Epoch {start_epoch+1}")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    print(f"🎬 Starting Training (Max {num_epochs} Epochs)...")
    
    for epoch in range(start_epoch, num_epochs):
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  📉 Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.1e}")
        
        # 3. Scheduler
        scheduler.step(val_loss)

        # 4. Checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f"{out_dir}/latest_checkpoint.pth")
        
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }, f"{out_dir}/best_model.pth")
            
            print("  ⭐ New Best Model!")
            save_visual_check(
                model, val_ds, check_indices, 
                filename=f"{out_dir}/BEST_MODEL_PREDICTIONS.png", device=device
            )
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            
        if epochs_no_improve >= patience:
            print(f"\n🛑 EARLY STOPPING. Best Loss: {best_loss:.4f}")
            break

# --- JUPYTER-SAFE EXECUTION BLOCK ---
if __name__ == "__main__":
    # If running in a notebook, call function directly
    # If running as script, use args
    import sys
    if 'ipykernel' in sys.modules:
        # Notebook Mode
        train(resume_checkpoint=None, num_epochs=1000)
    else:
        # Script Mode
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--resume', type=str, default=None)
        parser.add_argument('--epochs', type=int, default=1000)
        args = parser.parse_args()
        train(resume_checkpoint=args.resume, num_epochs=args.epochs)