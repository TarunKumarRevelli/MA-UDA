"""
Dataset classes for MA-UDA framework
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class BrainSegmentationDataset(Dataset):
    """Dataset for brain tumor segmentation with T1 and T2 images"""
    
    def __init__(self, source_img_dir, source_mask_dir, target_img_dir, 
                 transform=None, is_train=True):
        self.source_img_dir = source_img_dir
        self.source_mask_dir = source_mask_dir
        self.target_img_dir = target_img_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get list of images
        self.source_images = sorted([f for f in os.listdir(source_img_dir) 
                                    if f.endswith('.npy')])
        self.target_images = sorted([f for f in os.listdir(target_img_dir) 
                                    if f.endswith('.npy')])
        
        # For training, we need paired access
        self.length = max(len(self.source_images), len(self.target_images))
        
    def __len__(self):
        return self.length

    def _load_file(self, path, is_mask=False):
        """Helper to load .npy files and convert to PIL Image"""
        if path.endswith('.npy'):
            # Load the array
            data = np.load(path)
            
            # If it's a mask, keep it as integers (labels)
            if is_mask:
                data = data.astype(np.uint8)
                # Handle 3D shape (1, H, W) -> (H, W)
                if len(data.shape) == 3:
                    data = data.squeeze()
                return Image.fromarray(data).convert('L')
            
            # If it's an image, normalize to 0-255
            else:
                # Handle 3D shape (1, H, W) -> (H, W)
                if len(data.shape) == 3:
                    data = data.squeeze()
                    
                # Normalize float data to 0-255 uint8
                if data.dtype != np.uint8:
                    min_val = data.min()
                    max_val = data.max()
                    if max_val > min_val:
                        data = (data - min_val) / (max_val - min_val) * 255.0
                    else:
                        data = data * 0 # Handle blank images
                    data = data.astype(np.uint8)
                
                # Convert to RGB (replicates grayscale to 3 channels for models)
                return Image.fromarray(data).convert('RGB')
        else:
            # Fallback for standard image files
            mode = 'L' if is_mask else 'RGB'
            return Image.open(path).convert(mode)
    
    def __getitem__(self, idx):
        # Get source image and mask
        source_idx = idx % len(self.source_images)
        source_img_name = self.source_images[source_idx]
        
        source_img_path = os.path.join(self.source_img_dir, source_img_name)
        source_mask_path = os.path.join(self.source_mask_dir, source_img_name)
        
        # LOAD USING HELPER
        source_img = self._load_file(source_img_path, is_mask=False)
        source_mask = self._load_file(source_mask_path, is_mask=True)
        
        # Get target image
        target_idx = idx % len(self.target_images)
        target_img_name = self.target_images[target_idx]
        target_img_path = os.path.join(self.target_img_dir, target_img_name)
        
        # LOAD USING HELPER
        target_img = self._load_file(target_img_path, is_mask=False)
        
        # Apply transforms
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
            
            # Mask transform (custom resizing for mask)
            # We use Nearest Neighbor interpolation for masks to avoid creating new labels (e.g. 1.5)
            source_mask = transforms.functional.resize(
                source_mask, 
                (256, 256), 
                interpolation=transforms.InterpolationMode.NEAREST
            )
            # Convert to tensor manually to keep integer type
            source_mask = torch.from_numpy(np.array(source_mask)).long()
        
        return {
            'source_img': source_img,
            'source_mask': source_mask,
            'target_img': target_img,
            'source_name': source_img_name,
            'target_name': target_img_name
        }

class CycleGANDataset(Dataset):
    """Dataset for CycleGAN training"""
    
    def __init__(self, source_dir, target_dir, transform=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform
        
        self.source_images = sorted([f for f in os.listdir(source_dir) 
                                    if f.endswith('.npy')])
        self.target_images = sorted([f for f in os.listdir(target_dir) 
                                    if f.endswith('.npy')])
        
        self.length = max(len(self.source_images), len(self.target_images))
    
    def __len__(self):
        return self.length

    def _load_npy_image(self, path):
        """Helper specifically for loading CycleGAN images"""
        data = np.load(path)
        
        # Squeeze channel dim if present: (1, H, W) -> (H, W)
        if len(data.shape) == 3:
            data = data.squeeze()
            
        # Normalize to 0-255
        if data.dtype != np.uint8:
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val) * 255.0
            data = data.astype(np.uint8)
            
        return Image.fromarray(data).convert('RGB')
    
    def __getitem__(self, idx):
        source_idx = idx % len(self.source_images)
        target_idx = idx % len(self.target_images)
        
        source_path = os.path.join(self.source_dir, self.source_images[source_idx])
        target_path = os.path.join(self.target_dir, self.target_images[target_idx])
        
        # USE NEW LOADING LOGIC
        source_img = self._load_npy_image(source_path)
        target_img = self._load_npy_image(target_path)
        
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
        
        return {'source': source_img, 'target': target_img}

def get_transforms(is_train=True):
    """Get image transforms"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])