"""
Fourier Domain Adaptation (FDA) - Preprocessing Script
========================================================
This script applies FDA to transform Source (T1) images to look like Target (T2) images
while preserving the tumor structure for segmentation training.

Author: Senior CV Engineer
Date: 2026
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import random
import cv2

# ============================================================================
# FOURIER DOMAIN ADAPTATION (FDA) CORE FUNCTION
# ============================================================================

def apply_fda(source_img, target_img, beta=0.05):
    """
    Apply Fourier Domain Adaptation to transfer style from target to source.
    
    Args:
        source_img: Source image (H, W) numpy array [0, 255]
        target_img: Target image (H, W) numpy array [0, 255]
        beta: Float in [0, 1]. Controls how much low-frequency to swap.
              0.05-0.09 works well for medical images.
    
    Returns:
        Adapted image (H, W) numpy array [0, 255]
    
    FDA Algorithm:
    1. Compute 2D FFT of both images
    2. Swap low-frequency amplitude components (center of spectrum)
    3. Keep source phase (preserves structure/content)
    4. Reconstruct via Inverse FFT
    """
    # Ensure images are float32
    source_img = source_img.astype(np.float32)
    target_img = target_img.astype(np.float32)
    
    # Resize target to match source if needed
    if source_img.shape != target_img.shape:
        target_img = cv2.resize(target_img, (source_img.shape[1], source_img.shape[0]))
    
    # Step 1: Compute 2D FFT and shift zero frequency to center
    fft_source = np.fft.fft2(source_img)
    fft_target = np.fft.fft2(target_img)
    
    fft_source_shifted = np.fft.fftshift(fft_source)
    fft_target_shifted = np.fft.fftshift(fft_target)
    
    # Step 2: Extract amplitude and phase
    amp_source = np.abs(fft_source_shifted)
    phase_source = np.angle(fft_source_shifted)
    
    amp_target = np.abs(fft_target_shifted)
    
    # Step 3: Define the low-frequency region (center window)
    h, w = source_img.shape
    b = int(np.floor(min(h, w) * beta))  # Window size
    
    center_h, center_w = h // 2, w // 2
    
    # Step 4: Replace low-frequency amplitude of source with target
    amp_source[center_h - b:center_h + b, center_w - b:center_w + b] = \
        amp_target[center_h - b:center_h + b, center_w - b:center_w + b]
    
    # Step 5: Reconstruct FFT with modified amplitude and original phase
    fft_source_shifted_adapted = amp_source * np.exp(1j * phase_source)
    
    # Step 6: Inverse shift and Inverse FFT
    fft_source_adapted = np.fft.ifftshift(fft_source_shifted_adapted)
    adapted_img = np.fft.ifft2(fft_source_adapted)
    adapted_img = np.real(adapted_img)
    
    # Step 7: Clip values to valid range
    adapted_img = np.clip(adapted_img, 0, 255)
    
    return adapted_img.astype(np.uint8)


# ============================================================================
# IMAGE LOADING UTILITIES
# ============================================================================

def load_image(path):
    """Load image from various formats (.png, .jpg, .npy)"""
    path = str(path)
    
    if path.endswith('.npy'):
        img = np.load(path)
        # Normalize to 0-255 if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        return img
    else:
        # Load as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Try PIL as fallback
            img = np.array(Image.open(path).convert('L'))
        return img


def save_image(img, path):
    """Save image in appropriate format"""
    path = str(path)
    
    if path.endswith('.npy'):
        np.save(path, img)
    else:
        cv2.imwrite(path, img)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_fda_results(source_img, target_img, adapted_img, save_path):
    """Create before/after comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(source_img, cmap='gray')
    axes[0].set_title('Source (T1)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(target_img, cmap='gray')
    axes[1].set_title('Target Style (T2)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(adapted_img, cmap='gray')
    axes[2].set_title('FDA Result (Synthetic T2)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def run_fda_preprocessing(
    source_dir='/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1',
    mask_dir='/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/seg',
    target_dir='/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t2',
    output_img_dir='/kaggle/working/synthetic_t2',
    output_mask_dir='/kaggle/working/synthetic_masks',
    viz_dir='/kaggle/working/fda_visualizations',
    beta=0.05,
    num_viz_samples=10
):
    """
    Main FDA preprocessing pipeline
    
    Args:
        source_dir: Path to T1 images
        mask_dir: Path to segmentation masks
        target_dir: Path to T2 images (for style)
        output_img_dir: Where to save synthetic T2 images
        output_mask_dir: Where to copy masks
        viz_dir: Where to save visualizations
        beta: FDA beta parameter (0.05-0.09 recommended)
        num_viz_samples: Number of visualization examples to save
    """
    
    print("=" * 80)
    print("FOURIER DOMAIN ADAPTATION - PREPROCESSING")
    print("=" * 80)
    
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Get file lists
    source_files = sorted(list(Path(source_dir).glob('*')))
    target_files = sorted(list(Path(target_dir).glob('*')))
    
    print(f"\n📁 Dataset Statistics:")
    print(f"  Source (T1) images: {len(source_files)}")
    print(f"  Target (T2) images: {len(target_files)}")
    print(f"  Beta parameter: {beta}")
    
    if len(source_files) == 0 or len(target_files) == 0:
        raise ValueError("❌ No images found! Check your paths.")
    
    # Process each source image
    print(f"\n🔄 Starting FDA transformation...")
    viz_count = 0
    
    for idx, source_path in enumerate(tqdm(source_files, desc="Applying FDA")):
        try:
            # Load source image
            source_img = load_image(source_path)
            
            # Pick a random target image for style
            target_path = random.choice(target_files)
            target_img = load_image(target_path)
            
            # Apply FDA
            adapted_img = apply_fda(source_img, target_img, beta=beta)
            
            # Save adapted image with same filename
            output_path = Path(output_img_dir) / source_path.name
            save_image(adapted_img, output_path)
            
            # Copy corresponding mask
            mask_path = Path(mask_dir) / source_path.name
            if mask_path.exists():
                mask = load_image(mask_path)
                mask_output_path = Path(output_mask_dir) / source_path.name
                save_image(mask, mask_output_path)
            
            # Save visualization for first N samples
            if viz_count < num_viz_samples:
                viz_path = Path(viz_dir) / f"fda_example_{idx:04d}.png"
                visualize_fda_results(source_img, target_img, adapted_img, viz_path)
                viz_count += 1
                
        except Exception as e:
            print(f"⚠️  Error processing {source_path.name}: {e}")
            continue
    
    print(f"\n✅ FDA preprocessing complete!")
    print(f"  Synthetic images saved to: {output_img_dir}")
    print(f"  Masks copied to: {output_mask_dir}")
    print(f"  Visualizations saved to: {viz_dir}")
    
    # Summary statistics
    num_synthetic = len(list(Path(output_img_dir).glob('*')))
    num_masks = len(list(Path(output_mask_dir).glob('*')))
    print(f"\n📊 Output Statistics:")
    print(f"  Synthetic images created: {num_synthetic}")
    print(f"  Masks copied: {num_masks}")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'source_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1',
        'mask_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/seg',
        'target_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t2',
        'output_img_dir': '/kaggle/working/synthetic_t2',
        'output_mask_dir': '/kaggle/working/synthetic_masks',
        'viz_dir': '/kaggle/working/fda_visualizations',
        'beta': 0.01,  # Try 0.05, 0.09, or 0.01 for different style transfer strengths
        'num_viz_samples': 10
    }
    
    # Run FDA preprocessing
    run_fda_preprocessing(**CONFIG)