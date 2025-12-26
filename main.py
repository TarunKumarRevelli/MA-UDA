"""
Main execution script for MA-UDA framework
"""
import argparse
import os
import torch
from config.config import config
from train.train_cyclegan import CycleGANTrainer
from train.train_segmentation import MAUDATrainer

def main():
    parser = argparse.ArgumentParser(description='MA-UDA Framework for Medical Image Segmentation')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['cyclegan', 'segmentation', 'all'],
                       help='Training stage: cyclegan, segmentation, or all')
    parser.add_argument('--source_images', type=str, default='data/t1/images',
                       help='Path to source domain images')
    parser.add_argument('--source_masks', type=str, default='data/t1/masks',
                       help='Path to source domain masks')
    parser.add_argument('--target_images', type=str, default='data/t2/images',
                       help='Path to target domain images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--cyclegan_epochs', type=int, default=100,
                       help='Number of epochs for CycleGAN training')
    parser.add_argument('--seg_epochs', type=int, default=100,
                       help='Number of epochs for segmentation training')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Run a fast smoke test (2 epochs, 5 batches each)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config.source_images_path = args.source_images
    config.source_masks_path = args.source_masks
    config.target_images_path = args.target_images
    config.output_dir = args.output_dir
    config.checkpoint_dir = args.checkpoint_dir
    config.cyclegan_epochs = args.cyclegan_epochs
    config.seg_epochs = args.seg_epochs

    if args.dry_run:
        print("\n" + "!"*40)
        print("⚠️  DRY RUN MODE ACTIVATED ⚠️")
        print("!"*40)
        config.debug = True            # Enable debug flag
        config.cyclegan_epochs = 2     # Only run 2 epochs
        config.seg_epochs = 2          # Only run 2 epochs
        config.vis_interval = 1        # Save images every epoch
        config.save_interval = 1       # Save checkpoint every epoch
        config.cyclegan_batch_size = 2 # Tiny batch size
        config.seg_batch_size = 2      # Tiny batch size
        print("-> Epochs set to 2")
        print("-> Batches limited to 5 per epoch")
        print("-> Saving every epoch\n")
    
    if args.batch_size:
        config.cyclegan_batch_size = args.batch_size
        config.seg_batch_size = args.batch_size
    
    if args.lr:
        config.cyclegan_lr = args.lr
        config.seg_lr = args.lr
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("MA-UDA Configuration")
    print("="*60)
    print(f"Source images: {config.source_images_path}")
    print(f"Source masks: {config.source_masks_path}")
    print(f"Target images: {config.target_images_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Checkpoint directory: {config.checkpoint_dir}")
    print(f"Device: {config.device}")
    print(f"CycleGAN epochs: {config.cyclegan_epochs}")
    print(f"Segmentation epochs: {config.seg_epochs}")
    print("="*60)
    
    # Stage 1: Train CycleGAN
    if args.stage in ['cyclegan', 'all']:
        print("\n" + "="*60)
        print("Stage 1: Training CycleGAN for Image Translation")
        print("="*60)
        cyclegan_trainer = CycleGANTrainer(config)
        cyclegan_trainer.train()
        print("\nCycleGAN training completed!")

        # ====================================================
        # ⚠️ CRITICAL MEMORY FLUSH (ADD THIS BLOCK) ⚠️
        # ====================================================
        print("Freeing GPU memory before Stage 2...")
        
        # 1. Delete the object to remove python references
        del cyclegan_trainer
        
        # 2. Force Python Garbage Collection
        import gc
        gc.collect()
        
        # 3. Force PyTorch to release cached CUDA memory
        torch.cuda.empty_cache()
        
        print(f"Memory cleared. Current allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # ====================================================
    
    # Stage 2: Train Segmentation with MA-UDA
    if args.stage in ['segmentation', 'all']:
        print("\n" + "="*60)
        print("Stage 2: Training Segmentation with MA-UDA")
        print("="*60)
        
        # Check if CycleGAN checkpoint exists
        cyclegan_checkpoint = os.path.join(config.checkpoint_dir, 
                                          f'cyclegan_epoch_{config.cyclegan_epochs}.pth')
        if not os.path.exists(cyclegan_checkpoint):
            print(f"Warning: CycleGAN checkpoint not found at {cyclegan_checkpoint}")
            print("Segmentation training will use random initialization for generators.")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
        
        mauda_trainer = MAUDATrainer(config)
        mauda_trainer.train()
        print("\nSegmentation training completed!")
    
    print("\n" + "="*60)
    print("MA-UDA Training Pipeline Completed!")
    print("="*60)
    print(f"Checkpoints saved in: {config.checkpoint_dir}")
    print(f"Outputs saved in: {config.output_dir}")

if __name__ == '__main__':
    main()