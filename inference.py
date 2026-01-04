"""
Medical Image Segmentation Inference Script
============================================
Run inference on real T2-weighted MRI scans using the trained model.

Author: Senior CV Engineer
Date: 2026
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Color map for visualization (4 classes)
CLASS_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [255, 0, 0],      # Necrotic Core - Red
    2: [0, 255, 0],      # Edema - Green
    3: [0, 0, 255],      # Enhancing Tumor - Blue
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Necrotic Core',
    2: 'Edema',
    3: 'Enhancing Tumor'
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_image(path):
    """Load image from various formats"""
    path = str(path)
    
    if path.endswith('.npy'):
        img = np.load(path)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img


def get_inference_transform(img_size=256):
    """Transform pipeline for inference"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Lambda(image=lambda x: x.astype(np.float32) / 255.0),
        A.Lambda(image=lambda x: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])


def mask_to_rgb(mask, class_colors=CLASS_COLORS):
    """Convert class mask to RGB visualization"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color
    
    return rgb_mask


def create_overlay(image, mask, alpha=0.5):
    """Create overlay of prediction on original image"""
    # Convert grayscale to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create colored mask
    mask_rgb = mask_to_rgb(mask)
    
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_rgb, alpha, 0)
    
    return overlay


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_trained_model(checkpoint_path, num_classes=4, device='cuda'):
    """Load trained segmentation model"""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    # Create model architecture
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'best_dice' in checkpoint:
        print(f"  ✅ Loaded model with Dice Score: {checkpoint['best_dice']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  📊 Trained for {checkpoint['epoch'] + 1} epochs")
    
    return model


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_single_image(model, image, transform, device='cuda', original_size=None):
    """
    Run inference on a single image
    
    Args:
        model: Trained segmentation model
        image: Input image (H, W) numpy array
        transform: Albumentations transform
        device: Device to run inference on
        original_size: Tuple (H, W) to resize prediction back to
    
    Returns:
        Predicted mask (H, W) numpy array with class indices
    """
    # Store original size
    if original_size is None:
        original_size = image.shape
    
    # Preprocess
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Resize back to original size if needed
        if output.shape[2:] != original_size:
            output = F.interpolate(
                output, size=original_size, mode='bilinear', align_corners=False
            )
        
        # Get predicted class
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return predicted_mask


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_prediction(image, predicted_mask, save_path):
    """Create comprehensive visualization of prediction"""
    fig = plt.figure(figsize=(20, 5))
    
    # Original image
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input T2 Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Predicted mask (colored)
    ax2 = plt.subplot(1, 4, 2)
    mask_rgb = mask_to_rgb(predicted_mask)
    ax2.imshow(mask_rgb)
    ax2.set_title('Predicted Segmentation', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    ax3 = plt.subplot(1, 4, 3)
    overlay = create_overlay(image, predicted_mask, alpha=0.4)
    ax3.imshow(overlay)
    ax3.set_title('Overlay (40% opacity)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Class distribution
    ax4 = plt.subplot(1, 4, 4)
    class_counts = []
    class_labels = []
    for class_id in sorted(CLASS_NAMES.keys()):
        count = np.sum(predicted_mask == class_id)
        if count > 0:  # Only show non-zero classes
            class_counts.append(count)
            class_labels.append(CLASS_NAMES[class_id])
    
    colors = [np.array(CLASS_COLORS[i]) / 255.0 for i in sorted(CLASS_NAMES.keys()) if np.sum(predicted_mask == i) > 0]
    ax4.barh(class_labels, class_counts, color=colors)
    ax4.set_xlabel('Pixel Count', fontsize=12)
    ax4.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_batch_comparison(images, masks, save_path, max_samples=16):
    """Create grid comparison of multiple predictions"""
    n_samples = min(len(images), max_samples)
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx in range(n_samples):
        ax = axes[idx]
        
        # Create overlay
        overlay = create_overlay(images[idx], masks[idx], alpha=0.4)
        ax.imshow(overlay)
        ax.set_title(f'Sample {idx + 1}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_inference(
    model_path='/kaggle/working/outputs/best_model.pth',
    test_image_dir='/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t2',
    output_dir='/kaggle/working/predictions',
    img_size=256,
    num_classes=4,
    save_visualizations=True,
    batch_viz_samples=16
):
    """
    Main inference pipeline
    
    Args:
        model_path: Path to trained model checkpoint
        test_image_dir: Directory containing test T2 images
        output_dir: Where to save predictions and visualizations
        img_size: Input size for model
        num_classes: Number of segmentation classes
        save_visualizations: Whether to save visualization images
        batch_viz_samples: Number of samples for batch visualization
    """
    
    print("=" * 80)
    print("BRAIN TUMOR SEGMENTATION INFERENCE")
    print("=" * 80)
    
    # Create output directories
    pred_dir = os.path.join(output_dir, 'masks')
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load model
    model = load_trained_model(model_path, num_classes, device)
    
    # Get test images
    test_paths = sorted(list(Path(test_image_dir).glob('*')))
    print(f"\n📁 Found {len(test_paths)} test images")
    
    if len(test_paths) == 0:
        raise ValueError("❌ No test images found!")
    
    # Prepare transform
    transform = get_inference_transform(img_size)
    
    # Run inference
    print(f"\n🔮 Running inference...")
    all_images = []
    all_masks = []
    
    for idx, img_path in enumerate(tqdm(test_paths, desc="Processing images")):
        try:
            # Load image
            image = load_image(img_path)
            original_size = image.shape
            
            # Predict
            predicted_mask = predict_single_image(
                model, image, transform, device, original_size
            )
            
            # Save prediction
            mask_save_path = os.path.join(pred_dir, img_path.name)
            if mask_save_path.endswith('.npy'):
                np.save(mask_save_path, predicted_mask)
            else:
                # Save as PNG
                mask_save_path = mask_save_path.replace(img_path.suffix, '.png')
                cv2.imwrite(mask_save_path, predicted_mask.astype(np.uint8))
            
            # Save individual visualization
            if save_visualizations:
                viz_save_path = os.path.join(
                    viz_dir, f'prediction_{idx:04d}.png'
                )
                visualize_prediction(image, predicted_mask, viz_save_path)
            
            # Store for batch visualization
            if len(all_images) < batch_viz_samples:
                all_images.append(image)
                all_masks.append(predicted_mask)
        
        except Exception as e:
            print(f"⚠️  Error processing {img_path.name}: {e}")
            continue
    
    # Create batch comparison
    if len(all_images) > 0:
        print(f"\n📊 Creating batch comparison visualization...")
        batch_viz_path = os.path.join(output_dir, 'batch_predictions.png')
        create_batch_comparison(all_images, all_masks, batch_viz_path, batch_viz_samples)
    
    # Calculate statistics
    print(f"\n✅ Inference complete!")
    print(f"  Predictions saved to: {pred_dir}")
    if save_visualizations:
        print(f"  Visualizations saved to: {viz_dir}")
    
    # Compute class statistics across all predictions
    print(f"\n📈 Overall Statistics:")
    class_totals = {i: 0 for i in range(num_classes)}
    for mask in all_masks:
        for class_id in range(num_classes):
            class_totals[class_id] += np.sum(mask == class_id)
    
    total_pixels = sum(class_totals.values())
    for class_id, count in class_totals.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"  {CLASS_NAMES[class_id]}: {percentage:.2f}%")
    
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'model_path': '/kaggle/working/outputs/best_model.pth',
        'test_image_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t2',
        'output_dir': '/kaggle/working/predictions',
        'img_size': 256,
        'num_classes': 4,
        'save_visualizations': True,
        'batch_viz_samples': 16
    }
    
    # Run inference
    run_inference(**CONFIG)