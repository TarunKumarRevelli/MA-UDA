"""
Visualization utilities for MA-UDA results
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

def denormalize_image(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize image from [-1, 1] to [0, 1]"""
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)

def visualize_segmentation(image, mask, prediction, class_names=None, save_path=None):
    """
    Visualize image, ground truth mask, and prediction
    
    Args:
        image: input image [C, H, W]
        mask: ground truth [H, W]
        prediction: predicted mask [H, W] or [C, H, W]
        class_names: list of class names
        save_path: path to save the figure
    """
    if class_names is None:
        class_names = ['Background', 'WT', 'TC', 'ET']
    
    # Convert tensors to numpy
    if torch.is_tensor(image):
        image = denormalize_image(image).cpu().numpy().transpose(1, 2, 0)
    
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    if torch.is_tensor(prediction):
        if len(prediction.shape) == 3:
            prediction = torch.argmax(prediction, dim=0)
        prediction = prediction.cpu().numpy()
    
    # Create color map
    colors = ['black', 'green', 'blue', 'red']
    cmap = ListedColormap(colors[:len(class_names)])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image)
    mask_plot = axes[1].imshow(mask, alpha=0.5, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(image)
    pred_plot = axes[2].imshow(prediction, alpha=0.5, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[2].set_title('Prediction', fontsize=14)
    axes[2].axis('off')
    
    # Create legend
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) 
              for i in range(len(class_names))]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names), 
              bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_attention_maps(attention_maps, save_path=None):
    """
    Visualize attention maps from different heads
    
    Args:
        attention_maps: attention tensor [num_heads, H, W]
        save_path: path to save the figure
    """
    if torch.is_tensor(attention_maps):
        attention_maps = attention_maps.cpu().numpy()
    
    num_heads = min(attention_maps.shape[0], 8)  # Show max 8 heads
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_heads):
        attn = attention_maps[i]
        axes[i].imshow(attn, cmap='viridis')
        axes[i].set_title(f'Head {i+1}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_heads, 8):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def visualize_comparison(source_img, target_img, source_to_target, target_to_source,
                        pred_source, pred_target, mask_source, class_names=None, save_path=None):
    """
    Comprehensive visualization of domain adaptation results
    
    Args:
        source_img: source domain image
        target_img: target domain image
        source_to_target: generated target-style image from source
        target_to_source: generated source-style image from target
        pred_source: prediction on source
        pred_target: prediction on target
        mask_source: ground truth for source
        class_names: list of class names
        save_path: path to save the figure
    """
    if class_names is None:
        class_names = ['Background', 'WT', 'TC', 'ET']
    
    # Denormalize images
    source_img = denormalize_image(source_img).cpu().numpy().transpose(1, 2, 0)
    target_img = denormalize_image(target_img).cpu().numpy().transpose(1, 2, 0)
    source_to_target = denormalize_image(source_to_target).cpu().numpy().transpose(1, 2, 0)
    target_to_source = denormalize_image(target_to_source).cpu().numpy().transpose(1, 2, 0)
    
    # Convert predictions
    if torch.is_tensor(pred_source):
        if len(pred_source.shape) == 3:
            pred_source = torch.argmax(pred_source, dim=0)
        pred_source = pred_source.cpu().numpy()
    
    if torch.is_tensor(pred_target):
        if len(pred_target.shape) == 3:
            pred_target = torch.argmax(pred_target, dim=0)
        pred_target = pred_target.cpu().numpy()
    
    if torch.is_tensor(mask_source):
        mask_source = mask_source.cpu().numpy()
    
    # Create color map
    colors = ['black', 'green', 'blue', 'red']
    cmap = ListedColormap(colors[:len(class_names)])
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Source domain
    axes[0, 0].imshow(source_img)
    axes[0, 0].set_title('Source (T1)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(source_img)
    axes[0, 1].imshow(mask_source, alpha=0.5, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[0, 1].set_title('Source GT', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(source_img)
    axes[0, 2].imshow(pred_source, alpha=0.5, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[0, 2].set_title('Source Prediction', fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Target domain
    axes[1, 0].imshow(target_img)
    axes[1, 0].set_title('Target (T2)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(source_to_target)
    axes[1, 1].set_title('T1→T2 Generated', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(target_img)
    axes[1, 2].imshow(pred_target, alpha=0.5, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[1, 2].set_title('Target Prediction', fontsize=12)
    axes[1, 2].axis('off')
    
    # Create legend
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) 
              for i in range(len(class_names))]
    fig.legend(handles=patches, loc='lower center', ncol=len(class_names), 
              bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses=None, metrics=None, save_path=None):
    """
    Plot training curves
    
    Args:
        train_losses: dict of training losses
        val_losses: dict of validation losses (optional)
        metrics: dict of metrics (optional)
        save_path: path to save the figure
    """
    num_plots = 1
    if val_losses is not None:
        num_plots += 1
    if metrics is not None:
        num_plots += 1
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Plot training losses
    for key, values in train_losses.items():
        axes[0].plot(values, label=key)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()
    axes[0].grid(True)
    
    plot_idx = 1
    
    # Plot validation losses
    if val_losses is not None:
        for key, values in val_losses.items():
            axes[plot_idx].plot(values, label=key)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Validation Losses')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # Plot metrics
    if metrics is not None:
        for key, values in metrics.items():
            axes[plot_idx].plot(values, label=key)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Score')
        axes[plot_idx].set_title('Metrics')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()