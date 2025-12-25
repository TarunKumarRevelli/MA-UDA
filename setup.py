"""
Setup script to initialize the MA-UDA project structure
"""
import os
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        'config',
        'data/t1/images',
        'data/t1/masks',
        'data/t2/images',
        'models',
        'losses',
        'train',
        'utils',
        'notebooks',
        'outputs',
        'outputs/cyclegan_samples',
        'outputs/predictions',
        'checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    # Create __init__.py files
    init_dirs = ['config', 'data', 'models', 'losses', 'train', 'utils']
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"✓ Created: {init_file}")

def create_readme_sections():
    """Create additional documentation"""
    
    # Create DATASET.md
    dataset_md = """# Dataset Preparation Guide

## Required Data Format

### Source Domain (T1)
Place T1 MRI images and corresponding segmentation masks:
- `data/t1/images/`: T1 MRI images (PNG format)
- `data/t1/masks/`: Segmentation masks (PNG format)

### Target Domain (T2)
Place T2 MRI images:
- `data/t2/images/`: T2 MRI images (PNG format)

## Image Requirements
- Format: PNG (grayscale or RGB)
- Size: Will be resized to 256x256 during preprocessing
- Naming: Use consistent naming between images and masks

## Mask Format
- Grayscale PNG images
- Pixel values represent class labels:
  - 0: Background
  - 1: Whole Tumor (WT)
  - 2: Tumor Core (TC)
  - 3: Enhancing Tumor (ET)

## Example File Structure
```
data/
├── t1/
│   ├── images/
│   │   ├── patient001_slice01.png
│   │   ├── patient001_slice02.png
│   │   └── ...
│   └── masks/
│       ├── patient001_slice01.png
│       ├── patient001_slice02.png
│       └── ...
└── t2/
    └── images/
        ├── patient001_slice01.png
        ├── patient001_slice02.png
        └── ...
```

## Data Preprocessing Tips
1. Normalize intensity values
2. Ensure consistent spatial dimensions
3. Remove slices with minimal information
4. Balance class distribution if possible
"""
    
    with open('DATASET.md', 'w') as f:
        f.write(dataset_md)
    print("✓ Created: DATASET.md")
    
    # Create QUICKSTART.md
    quickstart_md = """# Quick Start Guide

## 1. Installation
```bash
git clone https://github.com/yourusername/MA-UDA.git
cd MA-UDA
pip install -r requirements.txt
python setup.py  # Initialize directory structure
```

## 2. Prepare Data
Place your T1 and T2 images in the appropriate directories:
- `data/t1/images/` - T1 images
- `data/t1/masks/` - T1 segmentation masks
- `data/t2/images/` - T2 images

See DATASET.md for detailed requirements.

## 3. Train the Model

### Option A: Complete Pipeline
```bash
python main.py --stage all
```

### Option B: Stage-by-Stage
```bash
# Stage 1: Train CycleGAN
python main.py --stage cyclegan --cyclegan_epochs 100

# Stage 2: Train Segmentation
python main.py --stage segmentation --seg_epochs 100
```

## 4. Visualize Results
```bash
jupyter notebook notebooks/visualize_results.ipynb
```

## 5. Customize Configuration
Edit `config/config.py` to modify:
- Batch sizes
- Learning rates
- Model architecture
- Loss weights

## For Kaggle Users

```python
# In Kaggle notebook
!pip install timm
!git clone https://github.com/yourusername/MA-UDA.git
%cd MA-UDA

# Train with your data
!python main.py --stage all \\
    --source_images /kaggle/input/your-dataset/t1/images \\
    --source_masks /kaggle/input/your-dataset/t1/masks \\
    --target_images /kaggle/input/your-dataset/t2/images
```

## Troubleshooting

### Out of Memory
Reduce batch size in `config/config.py`

### Models Not Training
Check that data paths are correct and images are loading properly

### Poor Results
- Ensure sufficient training epochs
- Verify data quality
- Check that CycleGAN trained properly before segmentation
"""
    
    with open('QUICKSTART.md', 'w') as f:
        f.write(quickstart_md)
    print("✓ Created: QUICKSTART.md")

def main():
    print("="*60)
    print("MA-UDA Project Setup")
    print("="*60)
    print()
    
    print("Creating directory structure...")
    create_directory_structure()
    print()
    
    print("Creating documentation...")
    create_readme_sections()
    print()
    
    print("="*60)
    print("Setup Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Place your data in data/t1/ and data/t2/")
    print("2. Review and modify config/config.py if needed")
    print("3. Run: python main.py --stage all")
    print()
    print("For more details, see:")
    print("- README.md (general information)")
    print("- QUICKSTART.md (quick start guide)")
    print("- DATASET.md (data preparation)")
    print()

if __name__ == '__main__':
    main()