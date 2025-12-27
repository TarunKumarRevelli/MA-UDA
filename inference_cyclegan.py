import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.cyclegan import Generator
from config.config import config

def infer_cyclegan(image_path, checkpoint_path, direction='s2t'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model
    gen = Generator().to(device)
    
    # 2. Load Weights
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if direction == 's2t':
        gen.load_state_dict(checkpoint['G_s2t'])
        print("Loaded Generator: Source (T1) -> Target (T2)")
    else:
        gen.load_state_dict(checkpoint['G_t2s'])
        print("Loaded Generator: Target (T2) -> Source (T1)")
    
    gen.eval()
    
    # 3. Load and Robustly Normalize Image (The Fix)
    print(f"Loading image: {image_path}")
    if image_path.endswith('.npy'):
        img_arr = np.load(image_path)
        
        # Debugging: See what's actually in the file
        print(f"Raw Image Stats -> Shape: {img_arr.shape}, Min: {img_arr.min():.4f}, Max: {img_arr.max():.4f}")
        
        # Handle 3D shape (1, H, W) -> (H, W)
        if len(img_arr.shape) == 3: 
            img_arr = img_arr.squeeze()
            
        # Normalize float data to 0-255 uint8 (Crucial Step)
        if img_arr.max() > img_arr.min():
            img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()) * 255.0
        else:
            print("WARNING: Image is constant/empty!")
            
        image = Image.fromarray(img_arr.astype(np.uint8)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    # 4. Transform (Resize to 128 to match training!)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # <--- MUST MATCH TRAINING SIZE
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        output_tensor = gen(input_tensor)
    
    # 6. Post-process
    def to_numpy(tensor):
        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2
        return np.clip(img, 0, 1)
    
    input_display = to_numpy(input_tensor)
    output_display = to_numpy(output_tensor)
    
    # 7. Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Input")
    plt.imshow(input_display)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Translated ({direction})")
    plt.imshow(output_display)
    plt.axis('off')
    
    plt.show()

# --- RUN TEST ---
chk_path = "/kaggle/input/cyclegan-15/pytorch/default/1/cyclegan_epoch_15.pth" 
# Try a different file just in case s70 is actually empty
test_img = "/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1/BraTS19_2013_10_1_s75.npy"

infer_cyclegan(test_img, chk_path, direction='s2t')