import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import shutil

# ============================================================================
# 1. YOUR CYCLEGAN ARCHITECTURE (Paste exactly as provided)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_channels = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels

        # Downsampling
        for _ in range(2):
            out_channels *= 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_channels)]

        # Upsampling
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_channels, channels, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# ============================================================================
# 2. GENERATION CONFIGURATION
# ============================================================================
CONFIG = {
    'model_path': '/kaggle/input/cyclegan99/pytorch/default/1/cyclegan_epoch_99.pth', # <--- VERIFY THIS PATH
    'source_t1_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1',
    'mask_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/seg',
    'output_img_dir': '/kaggle/working/synthetic_t2',
    'output_mask_dir': '/kaggle/working/synthetic_masks',
    'img_shape': (1, 256, 256), # (Channels, H, W)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# 3. UTILITIES
# ============================================================================

def load_checkpoint(model, path):
    print(f"📂 Loading weights from {path}...")
    checkpoint = torch.load(path, map_location=CONFIG['device'])
    
    # CycleGAN checkpoints often save generators as 'G_AB' (A->B) or 'G_BA' (B->A)
    # We want T1 -> T2. Assuming 'G_AB' is the correct direction.
    if isinstance(checkpoint, dict):
        if 'G_AB' in checkpoint:
            state_dict = checkpoint['G_AB']
            print("   ✅ Found 'G_AB' key (Using Source -> Target generator)")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint # Assume entire dict is weights
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_data():
    print("="*80)
    print("🧠 CYCLEGAN DATA GENERATION (T1 -> Synthetic T2)")
    print("="*80)
    
    os.makedirs(CONFIG['output_img_dir'], exist_ok=True)
    os.makedirs(CONFIG['output_mask_dir'], exist_ok=True)
    
    # 1. Init Model
    generator = GeneratorResNet(input_shape=CONFIG['img_shape'], num_residual_blocks=9)
    generator = generator.to(CONFIG['device'])
    try:
        generator = load_checkpoint(generator, CONFIG['model_path'])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Tip: Check if input_shape channels matches your training (1 vs 3).")
        return

    # 2. Process Files
    t1_files = sorted(list(Path(CONFIG['source_t1_dir']).glob('*')))
    print(f"🚀 Processing {len(t1_files)} images...")
    
    for t1_path in tqdm(t1_files):
        # A. Load T1
        if str(t1_path).endswith('.npy'): img = np.load(t1_path)
        else: img = cv2.imread(str(t1_path), cv2.IMREAD_GRAYSCALE)
        
        # B. Preprocess for CycleGAN (Norm -1 to 1)
        # Assuming input is 0-255 or 0-1.
        if img.max() > 1.0: img_norm = (img.astype(np.float32) / 127.5) - 1.0
        else: img_norm = (img.astype(np.float32) * 2.0) - 1.0
            
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(CONFIG['device'])
        
        # C. Inference
        with torch.no_grad():
            fake_t2 = generator(tensor)
            
        # D. Postprocess (Norm -1 to 1 -> 0 to 255)
        fake_t2 = fake_t2.squeeze().cpu().numpy()
        fake_t2 = (fake_t2 + 1.0) / 2.0  # 0 to 1
        fake_t2 = (fake_t2 * 255).astype(np.uint8) # 0 to 255
        
        # E. Save Image
        save_path = Path(CONFIG['output_img_dir']) / t1_path.name
        if str(save_path).endswith('.npy'): np.save(save_path, fake_t2)
        else: cv2.imwrite(str(save_path), fake_t2)
        
        # F. Copy Mask
        mask_path = Path(CONFIG['mask_dir']) / t1_path.name
        if mask_path.exists():
            shutil.copy(mask_path, Path(CONFIG['output_mask_dir']) / t1_path.name)
            
    print("\n✅ Generation Complete!")
    print(f"   Images: {CONFIG['output_img_dir']}")
    print(f"   Masks:  {CONFIG['output_mask_dir']}")

if __name__ == "__main__":
    generate_data()