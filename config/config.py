"""
Configuration file for MA-UDA framework
"""
import torch

class Config:
    def __init__(self):
        # Paths
        self.source_images_path = "data/t1/images"
        self.source_masks_path = "data/t1/masks"
        self.target_images_path = "data/t2/images"
        self.output_dir = "outputs"
        self.checkpoint_dir = "checkpoints"
        
        # Data settings
        self.image_size = (256, 256)
        self.num_classes = 4  # Background + 3 tumor regions (WT, TC, ET)
        
        # Training settings - Stage 1 (CycleGAN)
        self.cyclegan_epochs = 100
        self.cyclegan_lr = 2e-6
        self.cyclegan_batch_size = 6
        
        # Training settings - Stage 2 (Segmentation)
        self.seg_epochs = 100
        self.seg_lr = 0.01
        self.seg_batch_size = 8
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.discriminator_lr = 5e-5
        
        # Loss weights
        self.lambda_syn = 1.0
        self.lambda_seg = 1.0
        self.lambda_pred = 0.001
        self.lambda_ma = 0.1
        self.lambda_mha = 0.1
        
        # Model settings
        self.swin_embed_dim = 96
        self.swin_depths = [2, 2, 6, 2]
        self.swin_num_heads = [3, 6, 12, 24]
        self.window_size = 7
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4
        
        # Visualization
        self.vis_interval = 10
        self.save_interval = 5

        self.debug = False

config = Config()