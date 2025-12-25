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
                                    if f.endswith('.png')])
        self.target_images = sorted([f for f in os.listdir(target_img_dir) 
                                    if f.endswith('.png')])
        
        # For training, we need paired access
        self.length = max(len(self.source_images), len(self.target_images))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get source image and mask
        source_idx = idx % len(self.source_images)
        source_img_name = self.source_images[source_idx]
        
        source_img_path = os.path.join(self.source_img_dir, source_img_name)
        source_mask_path = os.path.join(self.source_mask_dir, source_img_name)
        
        source_img = Image.open(source_img_path).convert('RGB')
        source_mask = Image.open(source_mask_path).convert('L')
        
        # Get target image
        target_idx = idx % len(self.target_images)
        target_img_name = self.target_images[target_idx]
        target_img_path = os.path.join(self.target_img_dir, target_img_name)
        target_img = Image.open(target_img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            source_img = self.transform(source_img)
            target_img = self.transform(target_img)
            # Mask transform (no normalization)
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            source_mask = mask_transform(source_mask)
            source_mask = (source_mask * 255).long().squeeze(0)
        
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
                                    if f.endswith('.png')])
        self.target_images = sorted([f for f in os.listdir(target_dir) 
                                    if f.endswith('.png')])
        
        self.length = max(len(self.source_images), len(self.target_images))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        source_idx = idx % len(self.source_images)
        target_idx = idx % len(self.target_images)
        
        source_path = os.path.join(self.source_dir, self.source_images[source_idx])
        target_path = os.path.join(self.target_dir, self.target_images[target_idx])
        
        source_img = Image.open(source_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
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