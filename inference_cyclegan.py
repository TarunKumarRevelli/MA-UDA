import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.cyclegan import Generator
from config.config import config

def infer_cyclegan(image_path, checkpoint_path, direction='s2t'):
    """
    Args:
        image_path: Path to a .npy or .png file
        checkpoint_path: Path to cyclegan_epoch_15.pth
        direction: 's2t' (Source to Target) or 't2s' (Target to Source)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model
    gen = Generator().to(device)
    
    # 2. Load Weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if direction == 's2t':
        gen.load_state_dict(checkpoint['G_s2t'])
        print("Loaded Generator: Source (T1) -> Target (T2)")
    else:
        gen.load_state_dict(checkpoint['G_t2s'])
        print("Loaded Generator: Target (T2) -> Source (T1)")
    
    gen.eval()
    
    # 3. Preprocess Image
    # Same transform as training: Resize 128, Normalize -1 to 1
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if image_path.endswith('.npy'):
        img_arr = np.load(image_path)
        # Normalize simple 0-255 if needed
        if img_arr.max() > 1: img_arr = img_arr.astype(np.uint8)
        # Handle shape
        if len(img_arr.shape) == 3: img_arr = img_arr.squeeze()
        image = Image.fromarray(img_arr).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
        
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dim
    
    # 4. Inference
    with torch.no_grad():
        output_tensor = gen(input_tensor)
    
    # 5. Post-process (Denormalize)
    def to_numpy(tensor):
        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2 # -1..1 -> 0..1
        return np.clip(img, 0, 1)
    
    input_display = to_numpy(input_tensor)
    output_display = to_numpy(output_tensor)
    
    # 6. Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(input_display)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Translated ({direction})")
    plt.imshow(output_display)
    plt.axis('off')
    
    plt.show()

# --- USE IT ---
# Change paths to match your files
chk_path = "/kaggle/input/cyclegan-15/pytorch/default/1/cyclegan_epoch_15.pth" 
test_img = "/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1/BraTS19_2013_10_1_s70.npy"

infer_cyclegan(test_img, chk_path, direction='s2t')