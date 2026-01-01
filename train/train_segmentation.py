# """
# Training script for Segmentation with MA-UDA (Stage 2)
# """
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np

# import sys
# sys.path.append('..')

# from config.config import config
# from data.dataset import BrainSegmentationDataset, get_transforms
# from models.swin_transformer import SwinTransformerSegmentation
# from models.cyclegan import Generator
# from models.meta_attention import MetaAttention, AttentionAlignmentDiscriminator
# from losses.losses import MAUDALoss
# from utils.metrics import compute_dice_score

# class MAUDATrainer:
#     def __init__(self, config):
#         # In train_segmentation.py inside __init__
#         print("Initializing MAUDATrainer...")   # DEBUG
#         self.config = config
#         self.device = config.device
#         self.scaler = torch.cuda.amp.GradScaler()
        
#         # Create output directories
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         os.makedirs(os.path.join(config.output_dir, 'predictions'), exist_ok=True)
        
#         # Initialize segmentation model
#         self.seg_model = SwinTransformerSegmentation(
#             img_size=config.image_size[0],
#             num_classes=config.num_classes,
#             embed_dim=config.swin_embed_dim,
#             depths=config.swin_depths,
#             num_heads=config.swin_num_heads,
#             use_checkpoint=True
#         ).to(self.device)
        
#         # Load pre-trained CycleGAN generators
#         self.G_s2t = Generator().to(self.device)
#         self.G_t2s = Generator().to(self.device)
#         self.load_cyclegan_generators()
        
#         # Freeze CycleGAN generators
#         for param in self.G_s2t.parameters():
#             param.requires_grad = False
#         for param in self.G_t2s.parameters():
#             param.requires_grad = False
        
#         # Meta Attention module
#         self.meta_attention = MetaAttention(
#             num_heads=config.swin_num_heads[0],
#             hidden_dim=config.swin_embed_dim
#         ).to(self.device)
        
#         # Discriminators for attention alignment
#         self.disc_s = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
#         self.disc_t = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
        
#         # Optimizers
#         self.optimizer_seg = torch.optim.SGD(
#             list(self.seg_model.parameters()) + list(self.meta_attention.parameters()),
#             lr=config.seg_lr,
#             momentum=config.momentum,
#             weight_decay=config.weight_decay
#         )
        
#         self.optimizer_disc = torch.optim.Adam(
#             list(self.disc_s.parameters()) + list(self.disc_t.parameters()),
#             lr=config.discriminator_lr
#         )
        
#         # Learning rate scheduler
#         self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
#             self.optimizer_seg,
#             total_iters=config.seg_epochs,
#             power=0.75
#         )
        
#         # Loss function
#         self.loss_fn = MAUDALoss(config)
        
#         # Data
#         transform = get_transforms(is_train=True)
#         self.train_dataset = BrainSegmentationDataset(
#             config.source_images_path,
#             config.source_masks_path,
#             config.target_images_path,
#             transform=transform,
#             is_train=True
#         )
#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=config.seg_batch_size,
#             shuffle=True,
#             num_workers=config.num_workers
#         )
        
#         # Training stats
#         self.best_dice = 0.0
    
#     def load_cyclegan_generators(self):
#         """Load pre-trained CycleGAN generators"""
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, 
#                                       f'cyclegan_epoch_{self.config.cyclegan_epochs}.pth')
        
#         if os.path.exists(checkpoint_path):
#             print(f"Loading CycleGAN from {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.G_s2t.load_state_dict(checkpoint['G_s2t'])
#             self.G_t2s.load_state_dict(checkpoint['G_t2s'])
#         else:
#             print("Warning: CycleGAN checkpoint not found. Using random initialization.")
    
#     def train_epoch(self, epoch):
#         self.seg_model.train()
#         self.G_s2t.eval()
#         self.G_t2s.eval()
        
#         epoch_losses = {'total': 0, 'seg': 0, 'pred': 0, 'ma': 0}
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.seg_epochs}")
        
#         for i, batch in enumerate(pbar):
#             source_img = batch['source_img'].to(self.device)
#             source_mask = batch['source_mask'].to(self.device)
#             target_img = batch['target_img'].to(self.device)
            
#             # Generate translated images
#             with torch.no_grad():
#                 source_to_target = self.G_s2t(source_img)
#                 target_to_source = self.G_t2s(target_img)
            
#             # ==================== Train Segmentation Network ====================
#             self.optimizer_seg.zero_grad()
            
#             # Forward pass through segmentation network
#             pred_source = self.seg_model(source_img)
#             pred_source_to_target = self.seg_model(source_to_target)
#             pred_target = self.seg_model(target_img)
#             pred_target_to_source = self.seg_model(target_to_source)
            
#             # Get attention masks
#             attn_masks_source = self.seg_model.get_attention_masks()
            
#             # Compute meta attention (using first attention mask as example)
#             if len(attn_masks_source) > 0:
#                 ma_source = self.meta_attention(attn_masks_source[0])
                
#                 # Get attention for other domains
#                 _ = self.seg_model(target_to_source)
#                 attn_masks_t2s = self.seg_model.get_attention_masks()
#                 ma_target_to_source = self.meta_attention(attn_masks_t2s[0]) if len(attn_masks_t2s) > 0 else None
                
#                 _ = self.seg_model(target_img)
#                 attn_masks_target = self.seg_model.get_attention_masks()
#                 ma_target = self.meta_attention(attn_masks_target[0]) if len(attn_masks_target) > 0 else None
                
#                 _ = self.seg_model(source_to_target)
#                 attn_masks_s2t = self.seg_model.get_attention_masks()
#                 ma_source_to_target = self.meta_attention(attn_masks_s2t[0]) if len(attn_masks_s2t) > 0 else None
                
#                 # Discriminator outputs (for generator training)
#                 disc_s_output = self.disc_s(ma_target_to_source) if ma_target_to_source is not None else None
#                 disc_t_output = self.disc_t(ma_target) if ma_target is not None else None
#             else:
#                 disc_s_output = None
#                 disc_t_output = None
            
#             # Compute losses
#             loss_dict = self.loss_fn.compute_total_loss(
#                 pred_source=pred_source,
#                 target_source=source_mask,
#                 pred_source_to_target=pred_source_to_target,
#                 target_source_to_target=source_mask,
#                 pred_target=pred_target,
#                 pred_target_to_source=pred_target_to_source,
#                 disc_s_output=disc_s_output,
#                 disc_t_output=disc_t_output
#             )
            
#             loss = loss_dict['total']
#             loss.backward()
#             self.optimizer_seg.step()
            
#             # ==================== Train Discriminators ====================
#             if disc_s_output is not None and disc_t_output is not None:
#                 self.optimizer_disc.zero_grad()
                
#                 # Forward through discriminators
#                 real_s = self.disc_s(ma_source.detach())
#                 fake_s = self.disc_s(ma_target_to_source.detach())
                
#                 real_t = self.disc_t(ma_source_to_target.detach())
#                 fake_t = self.disc_t(ma_target.detach())
                
#                 # Compute discriminator losses
#                 loss_disc_s = self.loss_fn.attn_alignment_loss.forward_discriminator(real_s, fake_s)
#                 loss_disc_t = self.loss_fn.attn_alignment_loss.forward_discriminator(real_t, fake_t)
                
#                 loss_disc = loss_disc_s + loss_disc_t
#                 loss_disc.backward()
#                 self.optimizer_disc.step()
            
#             # Update stats
#             for key in epoch_losses:
#                 if key in loss_dict:
#                     epoch_losses[key] += loss_dict[key].item()
            
#             # Update progress bar
#             pbar.set_postfix({
#                 'loss': f"{loss.item():.4f}",
#                 'seg': f"{loss_dict['seg'].item():.4f}",
#                 'pred': f"{loss_dict['pred'].item():.4f}"
#             })

#             if self.config.debug and i >= 5:
#                 print("Debug mode: Breaking epoch early...")
#                 break
        
#         # Compute averages
#         for key in epoch_losses:
#             epoch_losses[key] /= len(self.train_loader)
        
#         print(f"Epoch {epoch+1} - Total Loss: {epoch_losses['total']:.4f}, "
#               f"Seg Loss: {epoch_losses['seg']:.4f}, "
#               f"Pred Loss: {epoch_losses['pred']:.4f}")
        
#         return epoch_losses
    
#     def save_checkpoint(self, epoch, dice_score):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': epoch,
#             'seg_model': self.seg_model.state_dict(),
#             'meta_attention': self.meta_attention.state_dict(),
#             'disc_s': self.disc_s.state_dict(),
#             'disc_t': self.disc_t.state_dict(),
#             'optimizer_seg': self.optimizer_seg.state_dict(),
#             'optimizer_disc': self.optimizer_disc.state_dict(),
#             'dice_score': dice_score
#         }
        
#         # Save latest
#         path = os.path.join(self.config.checkpoint_dir, 'latest_segmentation.pth')
#         torch.save(checkpoint, path)
        
#         # Save best
#         if dice_score > self.best_dice:
#             self.best_dice = dice_score
#             best_path = os.path.join(self.config.checkpoint_dir, 'best_segmentation.pth')
#             torch.save(checkpoint, best_path)
#             print(f"Best model saved with Dice: {dice_score:.4f}")
    
#     def train(self):
#         """Main training loop"""
#         print("Starting MA-UDA segmentation training...")
        
#         for epoch in range(self.config.seg_epochs):
#             # Train one epoch
#             losses = self.train_epoch(epoch)
            
#             # Update learning rate
#             self.scheduler.step()
            
#             # Save checkpoint
#             if (epoch + 1) % self.config.save_interval == 0:
#                 self.save_checkpoint(epoch, 0.0)  # Placeholder dice score
        
#         print("MA-UDA training completed!")

# def main():
#     trainer = MAUDATrainer(config)
#     trainer.train()

# if __name__ == '__main__':
#     main()


# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# import gc

# import sys
# sys.path.append('..')

# from config.config import config
# from data.dataset import BrainSegmentationDataset, get_transforms
# from models.swin_transformer import SwinTransformerSegmentation
# from models.cyclegan import Generator
# from models.meta_attention import MetaAttention, AttentionAlignmentDiscriminator
# from losses.losses import MAUDALoss

# class MAUDATrainer:
#     def __init__(self, config):
#         print("Initializing MAUDATrainer...")
#         self.config = config
#         self.device = config.device
#         self.scaler = torch.cuda.amp.GradScaler()
        
#         # Create output directories
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         os.makedirs(os.path.join(config.output_dir, 'predictions'), exist_ok=True)
        
#         # ---------------------------------------------------------
#         # 🛑 FORCE OVERRIDE SETTINGS (Safety Net)
#         # ---------------------------------------------------------
#         print("!"*40)
#         print("🛑 FORCING SAFE MEMORY SETTINGS 🛑")
#         forced_embed_dim = 48
#         forced_img_size = 128
#         forced_window_size = 4
#         print(f"-> Embed Dim: {forced_embed_dim}")
#         print(f"-> Image Size: {forced_img_size}")
#         print("!"*40)
        
#         # Initialize segmentation model with FORCED settings
#         self.seg_model = SwinTransformerSegmentation(
#             img_size=forced_img_size,      # Hardcoded 128
#             num_classes=config.num_classes,
#             embed_dim=forced_embed_dim,    # Hardcoded 48
#             depths=config.swin_depths,
#             num_heads=config.swin_num_heads,
#             use_checkpoint=True,
#             window_size=forced_window_size # Hardcoded 4
#         ).to(self.device)
        
#         # Load pre-trained CycleGAN generators
#         self.G_s2t = Generator().to(self.device)
#         self.G_t2s = Generator().to(self.device)
#         self.load_cyclegan_generators()
        
#         # Freeze CycleGAN generators
#         for param in self.G_s2t.parameters():
#             param.requires_grad = False
#         for param in self.G_t2s.parameters():
#             param.requires_grad = False
        
#         # Meta Attention module (Update hidden_dim to match forced embed_dim)
#         self.meta_attention = MetaAttention(
#             num_heads=config.swin_num_heads[0],
#             hidden_dim=forced_embed_dim    # Must match model embed_dim
#         ).to(self.device)
        
#         # Discriminators for attention alignment
#         # Input size depends on meta attention output. 
#         # Usually it's roughly related to embed_dim or fixed. 
#         # Assuming 64 is safe or derived from embed_dim.
#         # If your discriminator expects specific size, check models/meta_attention.py
#         self.disc_s = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
#         self.disc_t = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
        
#         # Optimizers
#         self.optimizer_seg = torch.optim.SGD(
#             list(self.seg_model.parameters()) + list(self.meta_attention.parameters()),
#             lr=config.seg_lr,
#             momentum=config.momentum,
#             weight_decay=config.weight_decay
#         )
        
#         self.optimizer_disc = torch.optim.Adam(
#             list(self.disc_s.parameters()) + list(self.disc_t.parameters()),
#             lr=config.discriminator_lr
#         )
        
#         self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
#             self.optimizer_seg, total_iters=config.seg_epochs, power=0.75
#         )
#         self.loss_fn = MAUDALoss(config)
        
#         # Data
#         transform = get_transforms(is_train=True)
#         self.train_dataset = BrainSegmentationDataset(
#             config.source_images_path,
#             config.source_masks_path,
#             config.target_images_path,
#             transform=transform,
#             is_train=True
#         )
#         # FORCE num_workers=0 to prevent shared memory leaks
#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=config.seg_batch_size,
#             shuffle=True,
#             num_workers=0, 
#             pin_memory=False,
#             persistent_workers=False
#         )
#         self.best_dice = 0.0
    
#     def load_cyclegan_generators(self):
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, 
#                                       f'cyclegan_epoch_{self.config.cyclegan_epochs}.pth')
#         if os.path.exists(checkpoint_path):
#             print(f"Loading CycleGAN from {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.G_s2t.load_state_dict(checkpoint['G_s2t'])
#             self.G_t2s.load_state_dict(checkpoint['G_t2s'])
#         else:
#             print("Warning: CycleGAN checkpoint not found. Using random initialization.")
    
#     def train_epoch(self, epoch):
#         self.seg_model.train()
#         self.G_s2t.eval()
#         self.G_t2s.eval()
        
#         epoch_losses = {'total': 0, 'seg': 0, 'pred': 0, 'ma': 0}
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.seg_epochs}")
        
#         for i, batch in enumerate(pbar):
#             source_img = batch['source_img'].to(self.device)
#             source_mask = batch['source_mask'].to(self.device)
#             target_img = batch['target_img'].to(self.device)

#             # 🛑 DEBUG: Print shape on first iteration
#             if i == 0:
#                 print(f"\n[DEBUG] Input Shape: {source_img.shape}") 
#                 # Expect: [1, 3, 128, 128]
#                 print(f"[DEBUG] Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

#             # Generate translated images
#             with torch.no_grad():
#                 with torch.cuda.amp.autocast():
#                     source_to_target = self.G_s2t(source_img)
#                     target_to_source = self.G_t2s(target_img)
            
#             # Train Segmentation
#             self.optimizer_seg.zero_grad()
            
#             with torch.cuda.amp.autocast():
#                 # Forward passes
#                 pred_source = self.seg_model(source_img)
#                 pred_source_to_target = self.seg_model(source_to_target)
#                 pred_target = self.seg_model(target_img)
#                 pred_target_to_source = self.seg_model(target_to_source)
                
#                 # Get attention (Should only be 1 mask now due to our fix)
#                 attn_masks_source = self.seg_model.get_attention_masks()
                
#                 # Meta Attention Logic
#                 if len(attn_masks_source) > 0:
#                     ma_source = self.meta_attention(attn_masks_source[0])
                    
#                     # We need to forward others to get their attention maps
#                     # To save memory, we can wrap these in no_grad if they aren't used for backprop
#                     # But MA-UDA usually requires gradients.
                    
#                     # ... (Repeated logic for getting masks from other domains) ...
#                     # For simplicity/memory in debugging, let's assume we got them.
#                     # If your model crashes here, the graph is too big.
                    
#                     # Hack: Re-using the same mask structure for dry run stability if OOM persists
#                     ma_target_to_source = ma_source 
#                     ma_target = ma_source
#                     ma_source_to_target = ma_source
                    
#                     disc_s_output = self.disc_s(ma_target_to_source)
#                     disc_t_output = self.disc_t(ma_target)
#                 else:
#                     disc_s_output = None
#                     disc_t_output = None
                
#                 loss_dict = self.loss_fn.compute_total_loss(
#                     pred_source=pred_source,
#                     target_source=source_mask,
#                     pred_source_to_target=pred_source_to_target,
#                     target_source_to_target=source_mask,
#                     pred_target=pred_target,
#                     pred_target_to_source=pred_target_to_source,
#                     disc_s_output=disc_s_output,
#                     disc_t_output=disc_t_output
#                 )
#                 loss = loss_dict['total']

#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer_seg)
            
#             # Train Discriminators
#             if disc_s_output is not None:
#                 self.optimizer_disc.zero_grad()
#                 with torch.cuda.amp.autocast():
#                     real_s = self.disc_s(ma_source.detach())
#                     fake_s = self.disc_s(ma_target_to_source.detach())
#                     real_t = self.disc_t(ma_source_to_target.detach())
#                     fake_t = self.disc_t(ma_target.detach())
                    
#                     loss_disc = (self.loss_fn.attn_alignment_loss.forward_discriminator(real_s, fake_s) + 
#                                  self.loss_fn.attn_alignment_loss.forward_discriminator(real_t, fake_t))
                
#                 self.scaler.scale(loss_disc).backward()
#                 self.scaler.step(self.optimizer_disc)
            
#             self.scaler.update()
            
#             # update stats...
#             for key in epoch_losses:
#                 if key in loss_dict:
#                     epoch_losses[key] += loss_dict[key].item()
#             pbar.set_postfix({'loss': f"{loss.item():.4f}"})

#             # 🛑 BREAK EARLY
#             if hasattr(self.config, 'debug') and self.config.debug and i >= 5:
#                 print("Debug mode: Breaking epoch early...")
#                 break
                
#         # Compute averages...
#         for key in epoch_losses:
#             epoch_losses[key] /= len(self.train_loader)
#         print(f"Epoch {epoch+1} done.")
#         return epoch_losses

#     def save_checkpoint(self, epoch, dice_score):
#         # ... (Same as before) ...
#         pass
    
#     def train(self):
#         # ... (Same as before) ...
#         print("Starting MA-UDA segmentation training...")
#         for epoch in range(self.config.seg_epochs):
#             self.train_epoch(epoch)
#             if (epoch + 1) % self.config.save_interval == 0:
#                 self.save_checkpoint(epoch, 0.0)
#         print("MA-UDA training completed!")



# """
# Training script for Segmentation with MA-UDA (Stage 2)
# OPTIMIZED: Sequential Backward Execution to fix OOM
# """
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np

# import sys
# sys.path.append('..')

# from config.config import config
# from data.dataset import BrainSegmentationDataset, get_transforms
# from models.swin_transformer import SwinTransformerSegmentation
# from models.cyclegan import Generator
# from models.meta_attention import MetaAttention, AttentionAlignmentDiscriminator
# from losses.losses import MAUDALoss

# class MAUDATrainer:
#     def __init__(self, config):
#         print("Initializing MAUDATrainer (Sequential Mode)...")
#         self.config = config
#         self.device = config.device
#         self.scaler = torch.cuda.amp.GradScaler()
        
#         # Create output directories
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         os.makedirs(os.path.join(config.output_dir, 'predictions'), exist_ok=True)
        
#         # 🛑 FORCE SETTINGS
#         print("!"*40)
#         print("🛑 SEQUENTIAL GRADIENT MODE ACTIVE 🛑")
#         forced_embed_dim = 48
#         forced_img_size = 128
#         print(f"-> Embed Dim: {forced_embed_dim}")
#         print(f"-> Image Size: {forced_img_size}")
#         print("!"*40)
        
#         # Initialize segmentation model
#         self.seg_model = SwinTransformerSegmentation(
#             img_size=forced_img_size,
#             num_classes=config.num_classes,
#             embed_dim=forced_embed_dim,
#             depths=config.swin_depths,
#             num_heads=config.swin_num_heads,
#             use_checkpoint=True,
#             window_size=4
#         ).to(self.device)
        
#         # Load pre-trained CycleGAN generators
#         self.G_s2t = Generator().to(self.device)
#         self.G_t2s = Generator().to(self.device)
#         self.load_cyclegan_generators()
        
#         # Freeze CycleGAN
#         for param in self.G_s2t.parameters(): param.requires_grad = False
#         for param in self.G_t2s.parameters(): param.requires_grad = False
        
#         # Meta Attention
#         self.meta_attention = MetaAttention(
#             num_heads=config.swin_num_heads[0],
#             hidden_dim=forced_embed_dim
#         ).to(self.device)
        
#         # Discriminators
#         self.disc_s = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
#         self.disc_t = AttentionAlignmentDiscriminator(input_size=64).to(self.device)
        
#         # Optimizers
#         self.optimizer_seg = torch.optim.SGD(
#             list(self.seg_model.parameters()) + list(self.meta_attention.parameters()),
#             lr=config.seg_lr, momentum=config.momentum, weight_decay=config.weight_decay
#         )
        
#         self.optimizer_disc = torch.optim.Adam(
#             list(self.disc_s.parameters()) + list(self.disc_t.parameters()),
#             lr=config.discriminator_lr
#         )
        
#         self.loss_fn = MAUDALoss(config)
        
#         # Data Loading
#         transform = get_transforms(is_train=True)
#         self.train_dataset = BrainSegmentationDataset(
#             config.source_images_path, config.source_masks_path,
#             config.target_images_path, transform=transform, is_train=True
#         )
#         # WORKERS=0 IS CRITICAL FOR YOUR SETUP
#         self.train_loader = DataLoader(
#             self.train_dataset,
#             batch_size=config.seg_batch_size,
#             shuffle=True,
#             num_workers=0, 
#             pin_memory=False,
#             persistent_workers=False
#         )
    
#     def load_cyclegan_generators(self):
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, 
#                                       f'cyclegan_epoch_{self.config.cyclegan_epochs}.pth')
#         if os.path.exists(checkpoint_path):
#             print(f"Loading CycleGAN from {checkpoint_path}")
#             checkpoint = torch.load(checkpoint_path, map_location=self.device)
#             self.G_s2t.load_state_dict(checkpoint['G_s2t'])
#             self.G_t2s.load_state_dict(checkpoint['G_t2s'])
#         else:
#             print("Warning: CycleGAN checkpoint not found!")

#     def train_epoch(self, epoch):
#         self.seg_model.train()
#         self.G_s2t.eval()
#         self.G_t2s.eval()
        
#         pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.seg_epochs}")
        
#         for i, batch in enumerate(pbar):
#             source_img = batch['source_img'].to(self.device)
#             source_mask = batch['source_mask'].to(self.device)
#             target_img = batch['target_img'].to(self.device)

#             # 1. Generate Fake Images (No Gradients needed here)
#             with torch.no_grad():
#                 with torch.cuda.amp.autocast():
#                     source_to_target = self.G_s2t(source_img)
#                     target_to_source = self.G_t2s(target_img)

#             # === SEQUENTIAL BACKWARD PASS START ===
#             # We clear gradients once at the start
#             self.optimizer_seg.zero_grad()
            
#             # --- STEP A: Source Domain (Real) ---
#             with torch.cuda.amp.autocast():
#                 pred_source = self.seg_model(source_img)
#                 loss_s = self.loss_fn.seg_loss(pred_source, source_mask)
#                 # Apply lambda immediately
#                 term_s = loss_s * self.config.lambda_seg
            
#             # BACKWARD IMMEDIATELY -> Frees Graph A
#             self.scaler.scale(term_s).backward()
            
#             # Capture attention for later (detach to save memory if needed)
#             # attn_s = self.seg_model.get_attention_masks()[0].detach() 

#             # --- STEP B: Fake Target Domain (S -> T) ---
#             with torch.cuda.amp.autocast():
#                 pred_s2t = self.seg_model(source_to_target)
#                 loss_s2t = self.loss_fn.seg_loss(pred_s2t, source_mask)
#                 term_s2t = loss_s2t * self.config.lambda_seg
            
#             # BACKWARD IMMEDIATELY -> Frees Graph B
#             self.scaler.scale(term_s2t).backward()

#             # --- STEP C: Target Consistency (Real T vs Fake S) ---
#             with torch.cuda.amp.autocast():
#                 pred_t = self.seg_model(target_img)
#                 pred_t2s = self.seg_model(target_to_source)
#                 loss_cons = self.loss_fn.pred_consistency_loss(pred_t, pred_t2s)
#                 term_cons = loss_cons * self.config.lambda_pred
            
#             # BACKWARD IMMEDIATELY -> Frees Graph C
#             self.scaler.scale(term_cons).backward()

#             # --- STEP D: Meta Attention (Optional/Lightweight) ---
#             # Note: For strict OOM prevention, we skip backpropping meta-attention 
#             # to the backbone in this simplified loop, but we can train discriminator.
#             # To enable full MA-UDA, we would need 24GB+ VRAM or huge checkpointing.
#             # For now, we skip it to ensure segmentation works.
            
#             # UPDATE WEIGHTS
#             self.scaler.step(self.optimizer_seg)
#             self.scaler.update()
#             # === SEQUENTIAL BACKWARD PASS END ===

#             # Logging stats (Approximation)
#             total_loss = term_s.item() + term_s2t.item() + term_cons.item()
#             pbar.set_postfix({'loss': f"{total_loss:.4f}"})

#     def save_checkpoint(self, epoch, dice_score):
#         checkpoint = {
#             'epoch': epoch,
#             'seg_model': self.seg_model.state_dict(),
#             'optimizer_seg': self.optimizer_seg.state_dict(),
#         }
#         path = os.path.join(self.config.checkpoint_dir, f'seg_epoch_{epoch+1}.pth')
#         torch.save(checkpoint, path)
#         print(f"Saved checkpoint: {path}")
    
#     def train(self):
#         print("Starting MA-UDA segmentation training...")
#         for epoch in range(self.config.seg_epochs):
#             self.train_epoch(epoch)
#             if (epoch + 1) % self.config.save_interval == 0:
#                 self.save_checkpoint(epoch, 0.0)
#         self.save_checkpoint(self.config.seg_epochs-1, 0.0)
#         print("MA-UDA training completed!")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.config import config
from data.dataset import BrainSegmentationDataset, get_transforms
from models.swin_transformer import SwinTransformerSegmentation
from models.cyclegan import Generator
from losses.losses import SegmentationLoss 

class MAUDATrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        # Updated scaler for newer PyTorch versions
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Initialize Output Dirs
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'predictions'), exist_ok=True)

        # Models
        self.seg_model = SwinTransformerSegmentation(
            img_size=128, num_classes=4, embed_dim=48, 
            depths=config.swin_depths, num_heads=config.swin_num_heads, window_size=4
        ).to(self.device)
        
        # CycleGAN (Frozen)
        self.G_s2t = Generator().to(self.device)
        # Note: We usually only need G_s2t (Source -> Target) for the consistency loss
        
        self.load_cyclegan() 
        
        # Freeze CycleGAN
        for p in self.G_s2t.parameters(): p.requires_grad = False
        
        # Optims
        self.optimizer_seg = torch.optim.SGD(self.seg_model.parameters(), lr=config.seg_lr, momentum=0.9)
        
        # Loss
        self.loss_fn = SegmentationLoss().to(self.device) 

        # Data
        self.train_loader = DataLoader(
            BrainSegmentationDataset(config.source_images_path, config.source_masks_path, config.target_images_path, transform=get_transforms(True), is_train=True),
            batch_size=config.seg_batch_size, shuffle=True, num_workers=0, pin_memory=False
        )

    def load_cyclegan(self):
        # 🟢 MODIFIED: Point directly to your Epoch 99 file
        path = "/kaggle/input/cyclegan-99/pytorch/default/1/cyclegan_epoch_99.pth"
        
        if os.path.exists(path):
            print(f"🔄 Loading CycleGAN from: {path}")
            chk = torch.load(path, map_location=self.device)
            
            # Handle variable key names (G_s2t vs G_AB)
            if 'G_s2t' in chk: 
                self.G_s2t.load_state_dict(chk['G_s2t'])
            elif 'G_AB' in chk: 
                self.G_s2t.load_state_dict(chk['G_AB'])
            else:
                print(f"⚠️ Keys found: {chk.keys()}")
                raise KeyError("Could not find Generator state_dict in checkpoint")
                
            print("✅ CycleGAN Brain Installed (Epoch 99)")
        else:
            print(f"❌ CRITICAL ERROR: CycleGAN file not found at {path}")
            raise FileNotFoundError(f"Missing: {path}")

    def train_epoch(self, epoch):
        self.seg_model.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            src_img = batch['source_img'].to(self.device)
            src_mask = batch['source_mask'].to(self.device)
            
            # 1. Generate Fake (No Grad)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_tgt = self.G_s2t(src_img)

            self.optimizer_seg.zero_grad()

            # 2. Sequential Backward A (Real Source)
            with torch.amp.autocast('cuda'):
                pred_src = self.seg_model(src_img)
                loss_s = self.loss_fn(pred_src, src_mask)
            self.scaler.scale(loss_s).backward()
            
            # 3. Sequential Backward B (Fake Target - Consistency)
            with torch.amp.autocast('cuda'):
                pred_fake = self.seg_model(fake_tgt)
                loss_f = self.loss_fn(pred_fake, src_mask)
            self.scaler.scale(loss_f).backward()

            self.scaler.step(self.optimizer_seg)
            self.scaler.update()
            
            pbar.set_postfix({'loss': (loss_s.item() + loss_f.item())})

    def save_checkpoint(self, epoch):
        torch.save({'seg_model': self.seg_model.state_dict()}, 
                   os.path.join(self.config.checkpoint_dir, f'seg_epoch_{epoch+1}.pth'))

    def load_weights_only(self, path):
        chk = torch.load(path, map_location=self.device)
        self.seg_model.load_state_dict(chk['seg_model'])
        print(f"✅ Loaded weights from {path} (Optimizer Reset)")

    def train(self, start_epoch=0):
        for epoch in range(start_epoch, self.config.seg_epochs):
            self.train_epoch(epoch)
            if (epoch+1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)