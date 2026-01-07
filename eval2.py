import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial.distance import directed_hausdorff

# Try importing medpy, fallback to scipy if needed
try:
    from medpy.metric.binary import hd95
    USE_MEDPY = True
    print("✅ Using MedPy for accurate HD95 calculation.")
except ImportError:
    USE_MEDPY = False
    print("⚠️ MedPy not found. Using simple Scipy approximation (Slower/Less Accurate).")
    print("   Run '!pip install medpy' for best results.")

# CONFIG
CONFIG = {
    'model_path': '/kaggle/input/unet-/pytorch/default/1/unet.pth', 
    'img_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t2', # REAL T2
    'mask_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/seg',
    'img_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0: return 1.0 if inter == 0 else 0.0
    return 2.0 * inter / union

from scipy.spatial.distance import directed_hausdorff

def calculate_hd95(pred, gt, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Robust HD95 Calculation.
    1. Tries MedPy (Fast, Accurate).
    2. Falls back to Scipy (Slow, Approximate) if MedPy fails.
    3. Returns Max Distance (e.g., 373mm for 256^2 diag) if one mask is empty.
    """
    # 1. Edge Case: Empty Masks
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0 # Both empty = Perfect match
    
    if pred.sum() == 0 or gt.sum() == 0:
        # One is empty, other is not -> Infinite distance. 
        # We cap it at the image diagonal (approx 373 pixels for 256x256) to avoid NaNs.
        return 373.0 
    
    # 2. Try MedPy (Preferred)
    try:
        from medpy.metric.binary import hd95
        return hd95(pred, gt, voxelspacing=voxel_spacing)
    except Exception:
        pass # Fallback
    except ImportError:
        pass

    # 3. Fallback: Scipy (Approximate 95%)
    # Directed Hausdorff is "Max(Min(d))". HD95 requires sorting distances.
    # Scipy only gives HD100 (Max). This is a rough approximation.
    d_pred_gt = directed_hausdorff(pred, gt)[0]
    d_gt_pred = directed_hausdorff(gt, pred)[0]
    return max(d_pred_gt, d_gt_pred)

# ============================================================================
# INFERENCE LOGIC (With TTA)
# ============================================================================

def tta_inference(model, tensor):
    # 1. Forward
    out_norm = model(tensor)
    
    # 2. Horizontal Flip (The "Mirror" View)
    tensor_h = torch.flip(tensor, dims=[3])
    out_h = model(tensor_h)
    out_h = torch.flip(out_h, dims=[3]) # Un-flip
    
    # Average
    return (out_norm + out_h) / 2.0

def evaluate_metrics_full():
    print("="*80)
    print(" 🧊 FULL 3D EVALUATION (Dice + HD95)")
    print("="*80)
    
    device = CONFIG['device']
    
    # Load Model
    model = smp.UnetPlusPlus(encoder_name='resnet34', classes=4).to(device)
    state = torch.load(CONFIG['model_path'], map_location=device)
    if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'])
    else: model.load_state_dict(state)
    model.eval()

    # Transform
    transform = A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Lambda(image=lambda x, **kwargs: np.stack([x, x, x], axis=-1)),
        ToTensorV2()
    ])

    # Group by Patient
    img_files = sorted(list(Path(CONFIG['img_dir']).glob('*')))
    patient_map = {}
    for f in img_files:
        patient_id = "_".join(f.stem.split('_')[:-1]) 
        if patient_id not in patient_map: patient_map[patient_id] = []
        patient_map[patient_id].append(f)

    # Store Results
    metrics = {
        'WT_Dice': [], 'TC_Dice': [], 'ET_Dice': [],
        'WT_HD95': [], 'TC_HD95': [], 'ET_HD95': []
    }
    
    print(f"🔍 Processing {len(patient_map)} patients...")
    
    for patient, slices in tqdm(patient_map.items()):
        vol_pred = []
        vol_gt = []
        
        # 1. Build 3D Volume
        for img_path in slices:
            mask_path = Path(CONFIG['mask_dir']) / img_path.name
            if not mask_path.exists(): continue
            
            # Load
            if str(img_path).endswith('.npy'):
                img = np.load(img_path)
                gt = np.load(mask_path)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Norm
            if img.max() > 1.0: img = img.astype(np.float32) / 255.0
            else: img = img.astype(np.float32)
            
            # Inference
            original_h, original_w = gt.shape[-2:] if len(gt.shape) > 2 else gt.shape
            tensor = transform(image=img)['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = tta_inference(model, tensor)
                out = torch.nn.functional.interpolate(out, size=(original_h, original_w), mode='bilinear')
                pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
            
            vol_pred.append(pred)
            vol_gt.append(gt)
            
        vol_pred = np.stack(vol_pred)
        vol_gt = np.stack(vol_gt)
        
        # 2. Calculate Metrics per Region
        
        # --- WT (Whole Tumor: Labels 1, 2, 3) ---
        wt_pred = np.isin(vol_pred, [1, 2, 3])
        wt_gt = np.isin(vol_gt, [1, 2, 3])
        if wt_gt.sum() > 0:
            metrics['WT_Dice'].append(calculate_dice(wt_pred, wt_gt))
            hd = calculate_hd95(wt_pred, wt_gt)
            if not np.isnan(hd): metrics['WT_HD95'].append(hd)

        # --- TC (Tumor Core: Labels 1, 3) ---
        tc_pred = np.isin(vol_pred, [1, 3])
        tc_gt = np.isin(vol_gt, [1, 3])
        if tc_gt.sum() > 0:
            metrics['TC_Dice'].append(calculate_dice(tc_pred, tc_gt))
            hd = calculate_hd95(tc_pred, tc_gt)
            if not np.isnan(hd): metrics['TC_HD95'].append(hd)

        # --- ET (Enhancing Tumor: Label 3) ---
        et_pred = (vol_pred == 3)
        et_gt = (vol_gt == 3)
        if et_gt.sum() > 0:
            metrics['ET_Dice'].append(calculate_dice(et_pred, et_gt))
            hd = calculate_hd95(et_pred, et_gt)
            if not np.isnan(hd): metrics['ET_HD95'].append(hd)

    print("\n" + "="*60)
    print(" 🏆 FINAL RESULTS (Mean ± Std)")
    print("="*60)
    
    # Helper to print
    def print_stat(name, data):
        if len(data) == 0:
            print(f"  {name}: N/A (No valid samples)")
        else:
            arr = np.array(data)
            # Dice is usually %, HD95 is usually mm (or pixels)
            if 'Dice' in name:
                print(f"  {name}: {arr.mean()*100:.2f} ± {arr.std()*100:.2f} %")
            else:
                print(f"  {name}: {arr.mean():.2f} ± {arr.std():.2f} mm") # assuming voxel spacing 1.0

    print("--- DICE SCORES (Higher is Better) ---")
    print_stat('WT_Dice', metrics['WT_Dice'])
    print_stat('TC_Dice', metrics['TC_Dice'])
    print_stat('ET_Dice', metrics['ET_Dice'])
    
    print("\n--- HD95 SCORES (Lower is Better) ---")
    print_stat('WT_HD95', metrics['WT_HD95'])
    print_stat('TC_HD95', metrics['TC_HD95'])
    print_stat('ET_HD95', metrics['ET_HD95'])
    print("="*60)

    print("\n" + "="*60)
    print(" 🏆 FINAL AVERAGES (Across all regions)")
    print("="*60)

    # Calculate means from the stored lists
    avg_dice = (
        np.mean(metrics['WT_Dice']) + 
        np.mean(metrics['TC_Dice']) + 
        np.mean(metrics['ET_Dice'])
    ) / 3.0
    
    # Check if we have HD95 values before averaging
    if len(metrics['WT_HD95']) > 0:
        avg_hd95 = (
            np.mean(metrics['WT_HD95']) + 
            np.mean(metrics['TC_HD95']) + 
            np.mean(metrics['ET_HD95'])
        ) / 3.0
    else:
        avg_hd95 = np.nan

    print(f"  ⭐ Average Dice: {avg_dice*100:.2f} %")
    print(f"  ⭐ Average HD95: {avg_hd95:.2f} mm")
    print("="*60)

if __name__ == "__main__":
    evaluate_metrics_full()