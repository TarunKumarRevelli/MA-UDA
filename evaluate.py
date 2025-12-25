"""
Evaluation script for MA-UDA trained models
"""
import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

from config.config import config
from models.swin_transformer import SwinTransformerSegmentation
from models.cyclegan import Generator
from data.dataset import get_transforms
from utils.metrics import evaluate_segmentation
from utils.visualization import visualize_segmentation, visualize_comparison

class ModelEvaluator:
    def __init__(self, checkpoint_path, config, use_cyclegan=True):
        self.config = config
        self.device = config.device
        self.use_cyclegan = use_cyclegan
        
        # Load segmentation model
        self.seg_model = SwinTransformerSegmentation(
            img_size=config.image_size[0],
            num_classes=config.num_classes,
            embed_dim=config.swin_embed_dim,
            depths=config.swin_depths,
            num_heads=config.swin_num_heads
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.seg_model.load_state_dict(checkpoint['seg_model'])
            print(f"Loaded model from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.seg_model.eval()
        
        # Load CycleGAN if needed
        if use_cyclegan:
            self.G_s2t = Generator().to(self.device)
            self.G_t2s = Generator().to(self.device)
            
            cyclegan_path = os.path.join(config.checkpoint_dir, 
                                        f'cyclegan_epoch_{config.cyclegan_epochs}.pth')
            if os.path.exists(cyclegan_path):
                cyclegan_ckpt = torch.load(cyclegan_path, map_location=self.device)
                self.G_s2t.load_state_dict(cyclegan_ckpt['G_s2t'])
                self.G_t2s.load_state_dict(cyclegan_ckpt['G_t2s'])
                print(f"Loaded CycleGAN from {cyclegan_path}")
            
            self.G_s2t.eval()
            self.G_t2s.eval()
        
        self.transform = get_transforms(is_train=False)
    
    def evaluate_on_dataset(self, image_dir, mask_dir=None, save_visualizations=True):
        """
        Evaluate model on a dataset
        
        Args:
            image_dir: directory containing test images
            mask_dir: directory containing ground truth masks (optional)
            save_visualizations: whether to save visualization images
        """
        image_dir = Path(image_dir)
        if mask_dir:
            mask_dir = Path(mask_dir)
        
        image_files = sorted(list(image_dir.glob('*.png')))
        
        results = {
            'dice_scores': [],
            'per_class_dice': {f'class_{i}': [] for i in range(1, self.config.num_classes)},
            'hd95': [],
            'asd': []
        }
        
        vis_dir = Path(self.config.output_dir) / 'evaluation_visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluating on {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Load image
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                prediction = self.seg_model(image_tensor)
                pred_class = torch.argmax(prediction[0], dim=0)
            
            # If ground truth is available, compute metrics
            if mask_dir:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask)
                    mask_tensor = torch.from_numpy(mask_array).to(self.device).unsqueeze(0)
                    
                    # Compute metrics
                    metrics = evaluate_segmentation(
                        prediction.unsqueeze(0),
                        mask_tensor,
                        num_classes=self.config.num_classes
                    )
                    
                    results['dice_scores'].append(metrics['dice']['average'])
                    for c in range(1, self.config.num_classes):
                        key = f'class_{c}'
                        results['per_class_dice'][key].append(metrics['dice'][key])
                    
                    # Save visualization
                    if save_visualizations:
                        vis_path = vis_dir / f"{img_path.stem}_result.png"
                        visualize_segmentation(
                            image_tensor[0],
                            mask_tensor[0],
                            prediction[0],
                            class_names=['Background', 'WT', 'TC', 'ET'],
                            save_path=str(vis_path)
                        )
            
            # Save prediction even without ground truth
            else:
                if save_visualizations:
                    vis_path = vis_dir / f"{img_path.stem}_prediction.png"
                    visualize_segmentation(
                        image_tensor[0],
                        pred_class,
                        pred_class,
                        class_names=['Background', 'WT', 'TC', 'ET'],
                        save_path=str(vis_path)
                    )
        
        # Compute average metrics
        if len(results['dice_scores']) > 0:
            avg_dice = np.mean(results['dice_scores'])
            std_dice = np.std(results['dice_scores'])
            
            print("\n" + "="*60)
            print("Evaluation Results")
            print("="*60)
            print(f"Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
            print()
            print("Per-Class Dice Scores:")
            for c in range(1, self.config.num_classes):
                key = f'class_{c}'
                if len(results['per_class_dice'][key]) > 0:
                    avg = np.mean(results['per_class_dice'][key])
                    std = np.std(results['per_class_dice'][key])
                    print(f"  Class {c}: {avg:.4f} ± {std:.4f}")
            print("="*60)
            
            # Save results to JSON
            results_json = {
                'average_dice': float(avg_dice),
                'std_dice': float(std_dice),
                'per_class_dice': {
                    k: {
                        'mean': float(np.mean(v)),
                        'std': float(np.std(v))
                    }
                    for k, v in results['per_class_dice'].items() if len(v) > 0
                },
                'num_samples': len(results['dice_scores'])
            }
            
            results_path = Path(self.config.output_dir) / 'evaluation_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            print(f"\nResults saved to: {results_path}")
        
        return results
    
    def predict_single_image(self, image_path, save_path=None):
        """Predict on a single image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.seg_model(image_tensor)
            pred_class = torch.argmax(prediction[0], dim=0)
        
        if save_path:
            visualize_segmentation(
                image_tensor[0],
                pred_class,
                pred_class,
                class_names=['Background', 'WT', 'TC', 'ET'],
                save_path=save_path
            )
        
        return pred_class.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Evaluate MA-UDA trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_images', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--test_masks', type=str, default=None,
                       help='Directory containing test masks (for computing metrics)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for visualizations')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable saving visualizations')
    parser.add_argument('--single_image', type=str, default=None,
                       help='Evaluate on a single image')
    
    args = parser.parse_args()
    
    # Update config
    config.output_dir = args.output_dir
    
    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, config)
    
    if args.single_image:
        # Predict on single image
        print(f"Predicting on: {args.single_image}")
        save_path = Path(args.output_dir) / 'single_prediction.png'
        evaluator.predict_single_image(args.single_image, save_path=str(save_path))
        print(f"Prediction saved to: {save_path}")
    else:
        # Evaluate on dataset
        evaluator.evaluate_on_dataset(
            args.test_images,
            args.test_masks,
            save_visualizations=not args.no_visualization
        )

if __name__ == '__main__':
    main()