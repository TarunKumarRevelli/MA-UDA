import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# ============================================================================
# 1. CORRECTED ARCHITECTURE (Matches cyclegan99.pth)
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

        # --- FIX: UPSAMPLING WITH CONV TRANSPOSE (Matches Checkpoint) ---
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        # ---------------------------------------------------------------

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
# 2. CONFIGURATION
# ============================================================================
CONFIG = {
    'model_path': '/kaggle/input/cyclegan99/pytorch/default/1/cyclegan_epoch_99.pth', 
    'source_t1_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1',
    'mask_dir': '/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/seg',
    'output_img_dir': '/kaggle/working/synthetic_t2_cyclegan',
    'output_mask_dir': '/kaggle/working/synthetic_masks_cyclegan',
    'img_shape': (3, 256, 256), 
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# 3. GENERATION LOOP
# ============================================================================

def generate_cyclegan_data():
    print(f"🚀 Loading CycleGAN from {CONFIG['model_path']}...")
    os.makedirs(CONFIG['output_img_dir'], exist_ok=True)
    os.makedirs(CONFIG['output_mask_dir'], exist_ok=True)
    
    # Init Model
    generator = GeneratorResNet(input_shape=CONFIG['img_shape'], num_residual_blocks=9)
    generator = generator.to(CONFIG['device'])
    
    # Load Weights
    checkpoint = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    
    if isinstance(checkpoint, dict):
        if 'G_s2t' in checkpoint:
            print("  ✅ Found 'G_s2t' key")
            state_dict = checkpoint['G_s2t']
        elif 'G_AB' in checkpoint:
            print("  ✅ Found 'G_AB' key")
            state_dict = checkpoint['G_AB']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    try:
        generator.load_state_dict(state_dict)
        print("  ✅ Weights loaded successfully!")
    except Exception as e:
        print(f"\n❌ LOAD ERROR: {e}")
        return

    generator.eval()
    
    # Process Images
    t1_files = sorted(list(Path(CONFIG['source_t1_dir']).glob('*')))
    print(f"🔄 Converting {len(t1_files)} T1 images to Synthetic T2...")
    
    for t1_path in tqdm(t1_files):
        # Load
        if str(t1_path).endswith('.npy'): img = np.load(t1_path)
        else: img = cv2.imread(str(t1_path), cv2.IMREAD_GRAYSCALE)
            
        # Preprocess (Gray -> 3Ch -> Norm)
        img_3ch = np.stack([img, img, img], axis=0)
        
        if img_3ch.max() > 1.0:
            img_norm = (img_3ch.astype(np.float32) / 127.5) - 1.0
        else:
            img_norm = (img_3ch.astype(np.float32) * 2.0) - 1.0
            
        tensor = torch.from_numpy(img_norm).unsqueeze(0).to(CONFIG['device'])
        
        # Inference
        with torch.no_grad():
            fake_t2 = generator(tensor)
            
        # Postprocess
        fake_t2 = fake_t2.squeeze().cpu().numpy()[0, :, :] 
        fake_t2 = (fake_t2 + 1.0) / 2.0 
        fake_t2 = (fake_t2 * 255).astype(np.uint8)
        
        # Save
        save_path = Path(CONFIG['output_img_dir']) / t1_path.name
        if str(save_path).endswith('.npy'): np.save(save_path, fake_t2)
        else: cv2.imwrite(str(save_path), fake_t2)
        
        # Copy Mask
        mask_path = Path(CONFIG['mask_dir']) / t1_path.name
        if mask_path.exists():
            shutil.copy(mask_path, Path(CONFIG['output_mask_dir']) / t1_path.name)
            
    print("\n✅ Done! Data generated.")

if __name__ == "__main__":
    generate_cyclegan_data()