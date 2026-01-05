import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ============================================================================
# 1. CORRECTED ARCHITECTURE (Must match 'cyclegan99.pth' weights)
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

        # Upsampling (Using ConvTranspose2d to match your checkpoint)
        for _ in range(2):
            out_channels //= 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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
# 2. YOUR LOADING & METRIC LOGIC (Preserved Exactly)
# ============================================================================

def load_and_fix_image(path):
    if not os.path.exists(path): return None
    if path.endswith('.npy'):
        img_arr = np.load(path)
        if len(img_arr.shape) == 3: img_arr = img_arr.squeeze()
        if img_arr.dtype == np.float32 or img_arr.dtype == np.float64:
            if img_arr.max() <= 1.1: img_arr = (img_arr * 255.0)
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr).convert('RGB')
    else:
        return Image.open(path).convert('RGB')

def calculate_metrics(img_pred, img_gt):
    # Convert RGB to Grayscale for fair structural comparison
    # img_pred and img_gt are numpy arrays [H, W, 3] in range 0-1
    gray_pred = np.dot(img_pred[...,:3], [0.2989, 0.5870, 0.1140])
    gray_gt = np.dot(img_gt[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Calculate SSIM
    score_ssim = ssim(gray_gt, gray_pred, data_range=1.0)
    # Calculate PSNR
    score_psnr = psnr(gray_gt, gray_pred, data_range=1.0)
    
    return score_ssim, score_psnr

# ============================================================================
# 3. COMPARISON LOGIC
# ============================================================================

def compare_cyclegan_vs_gt(source_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = os.path.basename(source_path)
    
    # --- Load Generator ---
    print(f"⏳ Loading CycleGAN 99 from {model_path}...")
    gen = GeneratorResNet(input_shape=(3, 128, 128), num_residual_blocks=9).to(device)
    
    chk = torch.load(model_path, map_location=device)
    
    # Robust Key Finding
    if isinstance(chk, dict):
        if 'G_s2t' in chk: key = 'G_s2t'       # Common custom key
        elif 'G_AB' in chk: key = 'G_AB'       # CycleGAN standard
        elif 'model_state_dict' in chk: key = 'model_state_dict'
        else: key = None
        
        state_dict = chk[key] if key else chk
    else:
        state_dict = chk
        
    gen.load_state_dict(state_dict)
    gen.eval()
    
    # --- Load Data ---
    target_path = source_path.replace('/t1/', '/t2/')
    source_img = load_and_fix_image(source_path)
    target_img = load_and_fix_image(target_path)
    
    # Standard CycleGAN Transform (Resize 128, Norm 0.5)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(source_img).unsqueeze(0).to(device)
    target_numpy = None
    
    if target_img:
        t_tensor = transform(target_img).unsqueeze(0)
        # Convert to numpy 0-1 for metrics
        target_numpy = t_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        target_numpy = (target_numpy + 1) / 2
        target_numpy = np.clip(target_numpy, 0, 1)

    # --- Inference & Scoring ---
    print(f"📉 Calculating metrics for {filename}...")
    
    with torch.no_grad():
        out = gen(input_tensor)
        # Post-process
        out_np = out.squeeze().cpu().numpy().transpose(1, 2, 0)
        out_np = (out_np + 1) / 2
        out_np = np.clip(out_np, 0, 1)
        
        # Metrics
        if target_numpy is not None:
            s_score, p_score = calculate_metrics(out_np, target_numpy)
        else:
            s_score, p_score = 0, 0

    # --- Plotting ---
    plt.figure(figsize=(15, 6))
    
    # 1. Input T1
    plt.subplot(1, 3, 1)
    plt.imshow((input_tensor.squeeze().cpu().numpy().transpose(1,2,0)+1)/2)
    plt.title("Input Source (T1)", fontsize=12)
    plt.axis('off')
    
    # 2. CycleGAN Output
    plt.subplot(1, 3, 2)
    plt.imshow(out_np)
    plt.title(f"CycleGAN Epoch 99\nSSIM: {s_score:.3f} | PSNR: {p_score:.1f}", 
              fontsize=14, fontweight='bold', color='green')
    plt.axis('off')

    # 3. Ground Truth T2
    plt.subplot(1, 3, 3)
    if target_numpy is not None:
        plt.imshow(target_numpy)
        plt.title("Ground Truth Target (T2)", fontsize=12)
    else:
        plt.title("GT Missing")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# RUN IT
# ============================================================================
if __name__ == "__main__":
    # Update these paths to match your actual files
    IMG_PATH = "/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1/BraTS19_2013_10_1_s75.npy"
    MODEL_PATH = "/kaggle/input/cyclegan99/pytorch/default/1/cyclegan_epoch_99.pth" # Verify this path
    
    if os.path.exists(IMG_PATH) and os.path.exists(MODEL_PATH):
        compare_cyclegan_vs_gt(IMG_PATH, MODEL_PATH)
    else:
        print("❌ Error: Check your paths!")
        print(f"Image exists? {os.path.exists(IMG_PATH)}")
        print(f"Model exists? {os.path.exists(MODEL_PATH)}")