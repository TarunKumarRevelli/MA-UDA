"""
Evaluation metrics for segmentation
"""
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_dice_score(pred, target, num_classes, ignore_background=True):
    """
    Compute Dice Similarity Coefficient
    
    Args:
        pred: predicted segmentation [B, H, W] or [B, C, H, W]
        target: ground truth [B, H, W]
        num_classes: number of classes
        ignore_background: whether to ignore background class
    
    Returns:
        dice_scores: dict with per-class and average Dice scores
    """
    # Convert pred to class labels if needed
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    dice_scores = {}
    start_class = 1 if ignore_background else 0
    
    for c in range(start_class, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice = (2.0 * intersection) / union
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        dice_scores[f'class_{c}'] = dice.item()
    
    # Compute average
    dice_scores['average'] = np.mean([v for k, v in dice_scores.items() if k.startswith('class_')])
    
    return dice_scores

def compute_hausdorff_distance(pred, target, percentile=95):
    """
    Compute Hausdorff Distance (HD95)
    
    Args:
        pred: predicted segmentation [H, W]
        target: ground truth [H, W]
        percentile: percentile for HD calculation
    
    Returns:
        hausdorff distance
    """
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(~pred.astype(bool))
    target_dt = distance_transform_edt(~target.astype(bool))
    
    # Get surface points
    pred_surface = pred & ~np.pad(pred, pad_width=1, mode='constant')[1:-1, 1:-1]
    target_surface = target & ~np.pad(target, pad_width=1, mode='constant')[1:-1, 1:-1]
    
    # Compute distances
    pred_distances = pred_dt[target_surface]
    target_distances = target_dt[pred_surface]
    
    if len(pred_distances) == 0 or len(target_distances) == 0:
        return 0.0
    
    # Compute percentile
    all_distances = np.concatenate([pred_distances, target_distances])
    hd = np.percentile(all_distances, percentile)
    
    return hd

def compute_asd(pred, target):
    """
    Compute Average Symmetric Surface Distance
    
    Args:
        pred: predicted segmentation [H, W]
        target: ground truth [H, W]
    
    Returns:
        average symmetric surface distance
    """
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    
    # Compute distance transforms
    pred_dt = distance_transform_edt(~pred.astype(bool))
    target_dt = distance_transform_edt(~target.astype(bool))
    
    # Get surface points
    pred_surface = pred & ~np.pad(pred, pad_width=1, mode='constant')[1:-1, 1:-1]
    target_surface = target & ~np.pad(target, pad_width=1, mode='constant')[1:-1, 1:-1]
    
    # Compute distances
    pred_distances = pred_dt[target_surface]
    target_distances = target_dt[pred_surface]
    
    if len(pred_distances) == 0 or len(target_distances) == 0:
        return 0.0
    
    # Average
    asd = (pred_distances.mean() + target_distances.mean()) / 2.0
    
    return asd

def evaluate_segmentation(pred, target, num_classes=4):
    """
    Comprehensive evaluation of segmentation
    
    Args:
        pred: predicted segmentation [B, C, H, W] or [B, H, W]
        target: ground truth [B, H, W]
        num_classes: number of classes
    
    Returns:
        metrics: dict with all evaluation metrics
    """
    # Convert to class labels if needed
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    # Compute Dice score
    dice_scores = compute_dice_score(pred, target, num_classes)
    
    # Compute HD95 and ASD for each class
    batch_size = pred.shape[0]
    hd95_scores = {f'class_{c}': [] for c in range(1, num_classes)}
    asd_scores = {f'class_{c}': [] for c in range(1, num_classes)}
    
    for b in range(batch_size):
        pred_b = pred[b].cpu().numpy()
        target_b = target[b].cpu().numpy()
        
        for c in range(1, num_classes):
            pred_c = (pred_b == c).astype(np.uint8)
            target_c = (target_b == c).astype(np.uint8)
            
            if pred_c.sum() > 0 and target_c.sum() > 0:
                try:
                    hd = compute_hausdorff_distance(pred_c, target_c)
                    asd = compute_asd(pred_c, target_c)
                    hd95_scores[f'class_{c}'].append(hd)
                    asd_scores[f'class_{c}'].append(asd)
                except:
                    pass
    
    # Average HD95 and ASD
    for c in range(1, num_classes):
        key = f'class_{c}'
        if len(hd95_scores[key]) > 0:
            hd95_scores[key] = np.mean(hd95_scores[key])
        else:
            hd95_scores[key] = 0.0
        
        if len(asd_scores[key]) > 0:
            asd_scores[key] = np.mean(asd_scores[key])
        else:
            asd_scores[key] = 0.0
    
    # Compile all metrics
    metrics = {
        'dice': dice_scores,
        'hd95': hd95_scores,
        'asd': asd_scores
    }
    
    return metrics