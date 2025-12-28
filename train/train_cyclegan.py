# """
# Training script for CycleGAN (Stage 1)
# """
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import numpy as np
# from PIL import Image

# import sys
# sys.path.append('..')

# from config.config import config
# from data.dataset import CycleGANDataset, get_transforms
# from models.cyclegan import CycleGAN
# from losses.losses import AdversarialLoss, CycleConsistencyLoss

# class CycleGANTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.device = config.device
        
#         # Create output directories
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         os.makedirs(os.path.join(config.output_dir, 'cyclegan_samples'), exist_ok=True)
        
#         # Initialize model
#         self.model = CycleGAN().to(self.device)
        
#         # Optimizers
#         self.optimizer_G = torch.optim.Adam(
#             list(self.model.G_s2t.parameters()) + list(self.model.G_t2s.parameters()),
#             lr=config.cyclegan_lr,
#             betas=(0.5, 0.999)
#         )
        
#         self.optimizer_D = torch.optim.Adam(
#             list(self.model.D_s.parameters()) + list(self.model.D_t.parameters()),
#             lr=config.cyclegan_lr,
#             betas=(0.5, 0.999)
#         )
        
#         # Loss functions
#         self.adv_loss = AdversarialLoss(loss_type='lsgan').to(self.device)
#         self.cycle_loss = CycleConsistencyLoss().to(self.device)
#         self.identity_loss = nn.L1Loss().to(self.device)
        
#         # Data
#         transform = get_transforms(is_train=True)
#         self.dataset = CycleGANDataset(
#             config.source_images_path,
#             config.target_images_path,
#             transform=transform
#         )
#         self.dataloader = DataLoader(
#             self.dataset,
#             batch_size=config.cyclegan_batch_size,
#             shuffle=True,
#             num_workers=config.num_workers
#         )
    
#     def train_epoch(self, epoch):
#         self.model.train()
        
#         epoch_g_loss = 0
#         epoch_d_loss = 0
        
#         pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.cyclegan_epochs}")
        
#         for i, batch in enumerate(pbar):
#             source = batch['source'].to(self.device)
#             target = batch['target'].to(self.device)
            
#             # ==================== Train Generators ====================
#             self.optimizer_G.zero_grad()
            
#             # Forward pass
#             outputs = self.model(source, target)
#             fake_target = outputs['fake_target']
#             fake_source = outputs['fake_source']
#             reconstructed_source = outputs['reconstructed_source']
#             reconstructed_target = outputs['reconstructed_target']
            
#             # Adversarial loss
#             loss_G_s2t = self.adv_loss(self.model.D_t(fake_target), is_real=True)
#             loss_G_t2s = self.adv_loss(self.model.D_s(fake_source), is_real=True)
            
#             # Cycle consistency loss
#             loss_cycle_s = self.cycle_loss(source, reconstructed_source)
#             loss_cycle_t = self.cycle_loss(target, reconstructed_target)
            
#             # Identity loss (helps preserve color composition)
#             loss_identity_s = self.identity_loss(self.model.G_t2s(source), source)
#             loss_identity_t = self.identity_loss(self.model.G_s2t(target), target)
            
#             # Total generator loss
#             loss_G = (loss_G_s2t + loss_G_t2s + 
#                      10.0 * (loss_cycle_s + loss_cycle_t) +
#                      5.0 * (loss_identity_s + loss_identity_t))
            
#             loss_G.backward()
#             self.optimizer_G.step()
            
#             # ==================== Train Discriminators ====================
#             self.optimizer_D.zero_grad()
            
#             # Discriminator for source domain
#             loss_D_real_s = self.adv_loss(self.model.D_s(source), is_real=True)
#             loss_D_fake_s = self.adv_loss(self.model.D_s(fake_source.detach()), is_real=False)
#             loss_D_s = (loss_D_real_s + loss_D_fake_s) * 0.5
            
#             # Discriminator for target domain
#             loss_D_real_t = self.adv_loss(self.model.D_t(target), is_real=True)
#             loss_D_fake_t = self.adv_loss(self.model.D_t(fake_target.detach()), is_real=False)
#             loss_D_t = (loss_D_real_t + loss_D_fake_t) * 0.5
            
#             # Total discriminator loss
#             loss_D = loss_D_s + loss_D_t
            
#             loss_D.backward()
#             self.optimizer_D.step()
            
#             # Update progress bar
#             epoch_g_loss += loss_G.item()
#             epoch_d_loss += loss_D.item()
#             pbar.set_postfix({
#                 'G_loss': f'{loss_G.item():.4f}',
#                 'D_loss': f'{loss_D.item():.4f}'
#             })

#             if self.config.debug and i >= 5:
#                 print("Debug mode: Breaking epoch early...")
#                 break
        
#         avg_g_loss = epoch_g_loss / len(self.dataloader)
#         avg_d_loss = epoch_d_loss / len(self.dataloader)
        
#         print(f"Epoch {epoch+1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
        
#         return avg_g_loss, avg_d_loss
    
#     def save_checkpoint(self, epoch):
#         """Save model checkpoint"""
#         checkpoint = {
#             'epoch': epoch,
#             'G_s2t': self.model.G_s2t.state_dict(),
#             'G_t2s': self.model.G_t2s.state_dict(),
#             'D_s': self.model.D_s.state_dict(),
#             'D_t': self.model.D_t.state_dict(),
#             'optimizer_G': self.optimizer_G.state_dict(),
#             'optimizer_D': self.optimizer_D.state_dict()
#         }
        
#         path = os.path.join(self.config.checkpoint_dir, f'cyclegan_epoch_{epoch+1}.pth')
#         torch.save(checkpoint, path)
#         print(f"Checkpoint saved: {path}")
    
#     def save_samples(self, epoch):
#         """Save sample generated images"""
#         self.model.eval()
        
#         with torch.no_grad():
#             # Get a batch
#             batch = next(iter(self.dataloader))
#             source = batch['source'].to(self.device)
#             target = batch['target'].to(self.device)
            
#             # Calculate how many images to save (min of Batch Size or 4)
#             # This prevents the "IndexError" when batch_size=2
#             num_samples = min(len(source), 4)
            
#             # Slice the data to the correct size
#             source = source[:num_samples]
#             target = target[:num_samples]
            
#             # Generate images
#             outputs = self.model(source, target)
            
#             # Denormalize
#             def denorm(x):
#                 return (x + 1) / 2
            
#             # Save images
#             for i in range(num_samples): # <--- Uses calculated length, not hardcoded 4
#                 # Source to Target
#                 img_s = denorm(source[i]).cpu().numpy().transpose(1, 2, 0)
#                 img_s2t = denorm(outputs['fake_target'][i]).cpu().numpy().transpose(1, 2, 0)
                
#                 # Target to Source
#                 img_t = denorm(target[i]).cpu().numpy().transpose(1, 2, 0)
#                 img_t2s = denorm(outputs['fake_source'][i]).cpu().numpy().transpose(1, 2, 0)
                
#                 # Define paths (Ensure .png extension)
#                 save_path_s2t = os.path.join(self.config.output_dir, 'cyclegan_samples',
#                                             f'epoch_{epoch+1}_s2t_{i}.png')
#                 save_path_t2s = os.path.join(self.config.output_dir, 'cyclegan_samples',
#                                             f'epoch_{epoch+1}_t2s_{i}.png')
                
#                 # Save using PIL
#                 Image.fromarray((img_s2t * 255).astype(np.uint8)).save(save_path_s2t)
#                 Image.fromarray((img_t2s * 255).astype(np.uint8)).save(save_path_t2s)
    
#     def train(self):
#         """Main training loop"""
#         print("Starting CycleGAN training...")
        
#         for epoch in range(self.config.cyclegan_epochs):
#             # Train one epoch
#             self.train_epoch(epoch)
            
#             # Save samples
#             if (epoch + 1) % self.config.vis_interval == 0:
#                 self.save_samples(epoch)
            
#             # Save checkpoint
#             if (epoch + 1) % self.config.save_interval == 0:
#                 self.save_checkpoint(epoch)
        
#         # Save final checkpoint
#         self.save_checkpoint(self.config.cyclegan_epochs - 1)
#         print("CycleGAN training completed!")

# def main():
#     trainer = CycleGANTrainer(config)
#     trainer.train()

# if __name__ == '__main__':
#     main()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import itertools
# import os
# from tqdm import tqdm
# from models.cyclegan import Generator, Discriminator
# from data.dataset import BrainSegmentationDataset, get_transforms
# from utils.utils import ReplayBuffer

# class CycleGANTrainer:
#     def __init__(self, config):
#         self.config = config
#         self.device = config.device
        
#         # Paths
#         os.makedirs(config.checkpoint_dir, exist_ok=True)
#         os.makedirs(os.path.join(config.output_dir, 'images'), exist_ok=True)

#         # 1. Initialize Models
#         self.G_AB = Generator().to(self.device) # T1 -> T2
#         self.G_BA = Generator().to(self.device) # T2 -> T1
#         self.D_A = Discriminator().to(self.device)
#         self.D_B = Discriminator().to(self.device)

#         # 2. Losses
#         self.criterion_GAN = torch.nn.MSELoss()
#         self.criterion_cycle = torch.nn.L1Loss()
#         self.criterion_identity = torch.nn.L1Loss()

#         # 3. Optimizers
#         self.optimizer_G = torch.optim.Adam(
#             itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
#             lr=config.cyclegan_lr, betas=(0.5, 0.999)
#         )
#         self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=config.cyclegan_lr, betas=(0.5, 0.999))
#         self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=config.cyclegan_lr, betas=(0.5, 0.999))

#         # 4. Buffers
#         self.fake_A_buffer = ReplayBuffer()
#         self.fake_B_buffer = ReplayBuffer()

#         # 5. Data (Unpaired)
#         self.dataloader = DataLoader(
#             BrainSegmentationDataset(config.source_images_path, config.source_masks_path, config.target_images_path, 
#                                      transform=get_transforms(is_train=True), is_train=True),
#             batch_size=config.cyclegan_batch_size, shuffle=True, num_workers=2
#         )

#     def load_checkpoint(self, path):
#         print(f"🔄 Loading CycleGAN from {path}...")
#         checkpoint = torch.load(path, map_location=self.device)
        
#         # Load Models (Handle key naming differences)
#         if 'G_s2t' in checkpoint:
#             self.G_AB.load_state_dict(checkpoint['G_s2t'])
#             self.G_BA.load_state_dict(checkpoint['G_t2s'])
#         else:
#             self.G_AB.load_state_dict(checkpoint['G_AB'])
#             self.G_BA.load_state_dict(checkpoint['G_BA'])
            
#         self.D_A.load_state_dict(checkpoint['D_A'])
#         self.D_B.load_state_dict(checkpoint['D_B'])
        
#         # Load Optimizers
#         self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
#         self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
#         self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        
#         epoch = checkpoint.get('epoch', 0)
#         print(f"✅ Successfully loaded Epoch {epoch}")
#         return epoch

#     def save_checkpoint(self, epoch):
#         state = {
#             'epoch': epoch,
#             'G_s2t': self.G_AB.state_dict(),
#             'G_t2s': self.G_BA.state_dict(),
#             'D_A': self.D_A.state_dict(),
#             'D_B': self.D_B.state_dict(),
#             'optimizer_G': self.optimizer_G.state_dict(),
#             'optimizer_D_A': self.optimizer_D_A.state_dict(),
#             'optimizer_D_B': self.optimizer_D_B.state_dict()
#         }
#         path = os.path.join(self.config.checkpoint_dir, f'cyclegan_epoch_{epoch}.pth')
#         torch.save(state, path)
#         print(f"💾 Saved CycleGAN checkpoint: {path}")

#     def train(self, start_epoch=0, total_epochs=100):
#         print(f"🚀 Starting CycleGAN Training: Epoch {start_epoch} -> {total_epochs}")
        
#         for epoch in range(start_epoch, total_epochs):
#             pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            
#             for i, batch in enumerate(pbar):
#                 real_A = batch['source_img'].to(self.device)
#                 real_B = batch['target_img'].to(self.device)

#                 # --- Train Generators ---
#                 self.optimizer_G.zero_grad()
                
#                 loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * 5.0
#                 loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * 5.0
                
#                 fake_B = self.G_AB(real_A)
#                 loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
                
#                 fake_A = self.G_BA(real_B)
#                 loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
                
#                 rec_A = self.G_BA(fake_B)
#                 loss_cycle_A = self.criterion_cycle(rec_A, real_A) * 10.0
                
#                 rec_B = self.G_AB(fake_A)
#                 loss_cycle_B = self.criterion_cycle(rec_B, real_B) * 10.0
                
#                 loss_G = loss_id_A + loss_id_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
#                 loss_G.backward()
#                 self.optimizer_G.step()
                
#                 # --- Train Discriminators ---
#                 self.optimizer_D_A.zero_grad()
#                 loss_D_A = (self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) + 
#                             self.criterion_GAN(self.D_A(self.fake_A_buffer.push_and_pop(fake_A).detach()), torch.zeros_like(self.D_A(fake_A)))) * 0.5
#                 loss_D_A.backward()
#                 self.optimizer_D_A.step()
                
#                 self.optimizer_D_B.zero_grad()
#                 loss_D_B = (self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) + 
#                             self.criterion_GAN(self.D_B(self.fake_B_buffer.push_and_pop(fake_B).detach()), torch.zeros_like(self.D_B(fake_B)))) * 0.5
#                 loss_D_B.backward()
#                 self.optimizer_D_B.step()
                
#                 pbar.set_postfix({'G_loss': loss_G.item()})

#             if (epoch + 1) % 5 == 0:
#                 self.save_checkpoint(epoch + 1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import os
from tqdm import tqdm
from models.cyclegan import Generator, Discriminator
from data.dataset import BrainSegmentationDataset, get_transforms
from utils.utils import ReplayBuffer

class CycleGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Paths
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'images'), exist_ok=True)

        # 1. Initialize Models
        self.G_AB = Generator().to(self.device) # T1 -> T2
        self.G_BA = Generator().to(self.device) # T2 -> T1
        self.D_A = Discriminator().to(self.device)
        self.D_B = Discriminator().to(self.device)

        # 2. Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        # 3. Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=config.cyclegan_lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=config.cyclegan_lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=config.cyclegan_lr, betas=(0.5, 0.999))

        # 4. Buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # 5. Data
        self.dataloader = DataLoader(
            BrainSegmentationDataset(config.source_images_path, config.source_masks_path, config.target_images_path, 
                                     transform=get_transforms(is_train=True), is_train=True),
            batch_size=config.cyclegan_batch_size, shuffle=True, num_workers=2
        )

    def load_checkpoint(self, path):
        print(f"🔄 Loading CycleGAN from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load Models
        if 'G_s2t' in checkpoint:
            self.G_AB.load_state_dict(checkpoint['G_s2t'])
            self.G_BA.load_state_dict(checkpoint['G_t2s'])
        else:
            self.G_AB.load_state_dict(checkpoint['G_AB'])
            self.G_BA.load_state_dict(checkpoint['G_BA'])
            
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B'])
        
        # Load Optimizers (Crucial for Resuming!)
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        
        epoch = checkpoint.get('epoch', 0)
        print(f"✅ Successfully loaded Epoch {epoch}")
        return epoch

    def save_checkpoint(self, epoch):
        # We save ALL parts needed to resume training perfectly
        state = {
            'epoch': epoch,
            'G_s2t': self.G_AB.state_dict(),
            'G_t2s': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict()
        }
        path = os.path.join(self.config.checkpoint_dir, f'cyclegan_epoch_{epoch}.pth')
        torch.save(state, path)
        print(f"💾 Saved CycleGAN checkpoint: {path}")

    def train(self, start_epoch=0, total_epochs=100):
        print(f"🚀 Starting CycleGAN Training: Epoch {start_epoch} -> {total_epochs}")
        
        for epoch in range(start_epoch, total_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
            
            for i, batch in enumerate(pbar):
                real_A = batch['source_img'].to(self.device)
                real_B = batch['target_img'].to(self.device)

                # --- Train Generators ---
                self.optimizer_G.zero_grad()
                
                loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * 5.0
                loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * 5.0
                
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
                
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
                
                rec_A = self.G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(rec_A, real_A) * 10.0
                
                rec_B = self.G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(rec_B, real_B) * 10.0
                
                loss_G = loss_id_A + loss_id_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
                loss_G.backward()
                self.optimizer_G.step()
                
                # --- Train Discriminators ---
                self.optimizer_D_A.zero_grad()
                loss_D_A = (self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A))) + 
                            self.criterion_GAN(self.D_A(self.fake_A_buffer.push_and_pop(fake_A).detach()), torch.zeros_like(self.D_A(fake_A)))) * 0.5
                loss_D_A.backward()
                self.optimizer_D_A.step()
                
                self.optimizer_D_B.zero_grad()
                loss_D_B = (self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B))) + 
                            self.criterion_GAN(self.D_B(self.fake_B_buffer.push_and_pop(fake_B).detach()), torch.zeros_like(self.D_B(fake_B)))) * 0.5
                loss_D_B.backward()
                self.optimizer_D_B.step()
                
                pbar.set_postfix({'G_loss': loss_G.item()})

            # 🟢 UPDATED: SAVE EVERY EPOCH
            self.save_checkpoint(epoch + 1)