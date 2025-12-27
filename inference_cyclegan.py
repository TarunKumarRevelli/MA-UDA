import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models.cyclegan import Generator

def load_and_fix_image(path):
    """Helper to load .npy or image files with robust normalization"""
    if not os.path.exists(path):
        return None
        
    if path.endswith('.npy'):
        img_arr = np.load(path)
        if len(img_arr.shape) == 3: img_arr = img_arr.squeeze()

        # FIX: Convert 0.0-1.0 Float to 0-255 Uint8
        if img_arr.dtype == np.float32 or img_arr.dtype == np.float64:
            if img_arr.max() <= 1.1: 
                img_arr = (img_arr * 255.0)
        
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr).convert('RGB')
    else:
        return Image.open(path).convert('RGB')

def infer_cyclegan(source_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filename = os.path.basename(source_path)
    
    # 1. Initialize Model
    print(f"Loading checkpoint: {checkpoint_path}")
    gen = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load Source->Target (T1 to T2)
    gen.load_state_dict(checkpoint['G_s2t'])
    gen.eval()
    
    # 2. Try to find the Corresponding Target (Ground Truth)
    # Assumes structure: .../t1/file.npy -> .../t2/file.npy
    target_path = source_path.replace('/t1/', '/t2/')
    
    # 3. Load Images
    print(f"Processing: {filename}")
    source_img = load_and_fix_image(source_path)
    target_img = load_and_fix_image(target_path) # Might be None if not found

    # 4. Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        
    input_tensor = transform(source_img).unsqueeze(0).to(device)
    
    # 5. Inference
    with torch.no_grad():
        output_tensor = gen(input_tensor)
    
    # 6. Post-process
    def to_numpy(tensor):
        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2
        return np.clip(img, 0, 1)
    
    disp_source = to_numpy(input_tensor)
    disp_gen = to_numpy(output_tensor)
    
    if target_img:
        target_tensor = transform(target_img).unsqueeze(0)
        disp_target = to_numpy(target_tensor)
    
    # 7. Plot 3-Way Comparison
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"File: {filename}", fontsize=14, fontweight='bold')
    
    # Input
    plt.subplot(1, 3, 1)
    plt.title("Input (T1)")
    plt.imshow(disp_source)
    plt.axis('off')
    
    # Generated
    plt.subplot(1, 3, 2)
    plt.title("Generated (Fake T2)")
    plt.imshow(disp_gen)
    plt.axis('off')
    
    # Ground Truth
    plt.subplot(1, 3, 3)
    if target_img:
        plt.title("Ground Truth (Real T2)")
        plt.imshow(disp_target)
    else:
        plt.title("Ground Truth Not Found")
        plt.text(0.5, 0.5, "Missing File", ha='center')
    plt.axis('off')
    
    # Save output
    output_filename = "/kaggle/working/inference_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"✅ Comparison saved to: {output_filename}")
    plt.close()

# --- RUN IT ---
# Update these paths to match your folder structure exactly
test_img = "/kaggle/input/brats19-60-to-90-slices-0-to-3-relabelled/t1/BraTS19_2013_10_1_s75.npy"
chk_path = "/kaggle/input/cyclegan-15/pytorch/default/1/cyclegan_epoch_15.pth"

infer_cyclegan(test_img, chk_path)