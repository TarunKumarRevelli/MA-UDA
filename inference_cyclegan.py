import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.cyclegan import Generator

def infer_cyclegan(image_path, checkpoint_path, direction='s2t'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize Model and Load Weights
    print(f"Loading checkpoint: {checkpoint_path}")
    gen = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if direction == 's2t':
        gen.load_state_dict(checkpoint['G_s2t'])
        title_text = "Source (T1) -> Target (T2)"
    else:
        gen.load_state_dict(checkpoint['G_t2s'])
        title_text = "Target (T2) -> Source (T1)"
    
    gen.eval()
    print(f"Loaded Generator: {title_text}")
    
    # 2. Load and Fix Image
    print(f"Loading image: {image_path}")
    if image_path.endswith('.npy'):
        img_arr = np.load(image_path)
        if len(img_arr.shape) == 3: img_arr = img_arr.squeeze()

        # FIX: Convert 0.0-1.0 Float to 0-255 Uint8
        if img_arr.dtype == np.float32 or img_arr.dtype == np.float64:
            if img_arr.max() <= 1.1: 
                img_arr = (img_arr * 255.0)
        
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_arr).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad():
        output_tensor = gen(input_tensor)
    
    # 5. Visualization Helper
    def to_numpy(tensor):
        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2
        return np.clip(img, 0, 1)
    
    input_display = to_numpy(input_tensor)
    output_display = to_numpy(output_tensor)
    
    # 6. Plot & SAVE (Changed from .show())
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Input (Resized)")
    plt.imshow(input_display)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Translated Output")
    plt.imshow(output_display)
    plt.axis('off')
    
    # --- CHANGE IS HERE ---
    output_filename = "inference_result.png"
    plt.savefig(output_filename)
    print(f"✅ Success! Image saved to: {output_filename}")
    plt.close() # Close memory to free RAM
    # ----------------------

# --- RUN IT ---
test_img = "/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1/BraTS19_2013_10_1_s75.npy"
chk_path = "/kaggle/input/cyclegan-15/pytorch/default/1/cyclegan_epoch_15.pth"

infer_cyclegan(test_img, chk_path, direction='s2t')