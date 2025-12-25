# MA-UDA: Meta Attention Unsupervised Domain Adaptation for Medical Image Segmentation

Implementation of the paper "Unsupervised Domain Adaptation for Medical Image Segmentation Using Transformer With Meta Attention" (IEEE TMI 2024).

## Overview

This repository implements MA-UDA, a Transformer-based framework for cross-modality medical image segmentation (T1 → T2 brain MRI). The method uses:
- CycleGAN for image synthesis
- Swin Transformer for segmentation
- Meta Attention for attention-level domain alignment

## Project Structure

```
MA-UDA/
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration settings
├── data/
│   ├── __init__.py
│   ├── dataset.py                # Dataset classes
│   └── transforms.py             # Data augmentation
├── models/
│   ├── __init__.py
│   ├── cyclegan.py               # CycleGAN for image translation
│   ├── swin_transformer.py       # Swin Transformer backbone
│   ├── meta_attention.py         # Meta Attention module
│   └── discriminator.py          # Attention discriminators
├── losses/
│   ├── __init__.py
│   └── losses.py                 # All loss functions
├── train/
│   ├── __init__.py
│   ├── train_cyclegan.py         # Stage 1: CycleGAN training
│   └── train_segmentation.py    # Stage 2: Segmentation with MA-UDA
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # Evaluation metrics (Dice, HD95, ASD)
│   └── visualization.py          # Visualization utilities
├── notebooks/
│   └── visualize_results.ipynb   # Jupyter notebook for visualization
├── requirements.txt
├── README.md
└── main.py                       # Main execution script
```

## Data Preparation

Organize your data in the following structure:

```
data/
├── t1/
│   ├── images/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── t2/
    └── images/
        ├── image_001.png
        ├── image_002.png
        └── ...
```

### Data Format
- Images should be in PNG format (grayscale or RGB)
- Masks should be grayscale PNG with pixel values representing class labels:
  - 0: Background
  - 1: Whole Tumor (WT)
  - 2: Tumor Core (TC)
  - 3: Enhancing Tumor (ET)

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MA-UDA.git
cd MA-UDA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Setup

1. Create a new Kaggle notebook
2. Upload the code as a dataset or clone from GitHub
3. Install additional dependencies if needed:

```python
!pip install timm
```

## Training

### Two-Stage Training Pipeline

#### Stage 1: Train CycleGAN (Image Translation)

```bash
python main.py --stage cyclegan \
    --source_images data/t1/images \
    --target_images data/t2/images \
    --cyclegan_epochs 100 \
    --output_dir outputs \
    --checkpoint_dir checkpoints
```

#### Stage 2: Train Segmentation with MA-UDA

```bash
python main.py --stage segmentation \
    --source_images data/t1/images \
    --source_masks data/t1/masks \
    --target_images data/t2/images \
    --seg_epochs 100 \
    --output_dir outputs \
    --checkpoint_dir checkpoints
```

#### Train Both Stages (Complete Pipeline)

```bash
python main.py --stage all \
    --source_images data/t1/images \
    --source_masks data/t1/masks \
    --target_images data/t2/images \
    --cyclegan_epochs 100 \
    --seg_epochs 100
```

### Configuration

Edit `config/config.py` to customize:
- Image size
- Number of classes
- Batch sizes
- Learning rates
- Loss weights
- Model architecture parameters

## Evaluation

The framework automatically computes:
- **Dice Similarity Coefficient (DSC)**: Overlap between predicted and ground truth
- **Hausdorff Distance 95 (HD95)**: 95th percentile of surface distances
- **Average Symmetric Surface Distance (ASD)**: Average distance between surfaces

## Visualization

### Using Jupyter Notebook

Open `notebooks/visualize_results.ipynb` to:
- Visualize segmentation results
- Compare source and target predictions
- View attention maps
- Plot training curves

### Programmatic Visualization

```python
from utils.visualization import visualize_segmentation, visualize_comparison
import torch
from PIL import Image

# Load and visualize a single result
image = Image.open('path/to/image.png')
mask = Image.open('path/to/mask.png')
prediction = model(image)

visualize_segmentation(
    image, mask, prediction,
    class_names=['Background', 'WT', 'TC', 'ET'],
    save_path='outputs/result.png'
)
```

## Results

Expected performance on BraTS (T1 → T2):

| Method | DSC (%) | HD95 (mm) |
|--------|---------|-----------|
| Source Only | 13.32 | 60.20 |
| CycleGAN | ~45.0 | ~35.0 |
| **MA-UDA (Ours)** | **72.89** | **13.02** |
| Target Only | 83.65 | 9.18 |

## Key Features

1. **Meta Attention**: Learns attention of multi-head attention for better domain alignment
2. **Attention-Based Alignment**: Transfers discriminative information across domains
3. **Staged Training**: Separates image translation and segmentation training
4. **Comprehensive Metrics**: Dice, HD95, and ASD evaluation

## Kaggle Workflow

### 1. Upload Data
Upload your dataset to Kaggle Datasets or use input mounting

### 2. Create Notebook
```python
# Install dependencies
!pip install timm

# Clone repository
!git clone https://github.com/yourusername/MA-UDA.git
%cd MA-UDA

# Train
!python main.py --stage all \
    --source_images /kaggle/input/your-dataset/t1/images \
    --source_masks /kaggle/input/your-dataset/t1/masks \
    --target_images /kaggle/input/your-dataset/t2/images
```

### 3. Save Results
```python
# Copy outputs to Kaggle output
!cp -r outputs /kaggle/working/
!cp -r checkpoints /kaggle/working/
```

## Customization

### Adding New Datasets
1. Create dataset class in `data/dataset.py`
2. Update configuration in `config/config.py`
3. Adjust `num_classes` and class names

### Modifying Architecture
- Change Swin Transformer parameters in config
- Modify encoder depths: `swin_depths`
- Adjust number of attention heads: `swin_num_heads`

### Custom Loss Functions
Add new losses in `losses/losses.py` and update the loss computation in training scripts

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size`
- Use gradient accumulation

### CycleGAN Not Converging
- Increase `cyclegan_epochs`
- Adjust learning rate
- Check data quality and preprocessing

### Poor Segmentation Performance
- Ensure CycleGAN is properly trained first
- Increase `seg_epochs`
- Adjust loss weights in config

## Citation

If you use this code, please cite:

```bibtex
@article{ji2024unsupervised,
  title={Unsupervised Domain Adaptation for Medical Image Segmentation Using Transformer With Meta Attention},
  author={Ji, Wen and Chung, Albert CS},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={2},
  pages={820--831},
  year={2024},
  publisher={IEEE}
}
```

## License

This project is for research purposes only.

## Acknowledgments

- Original paper: Ji & Chung, IEEE TMI 2024
- Swin Transformer: Liu et al.
- CycleGAN: Zhu et al.

## Contact

For questions or issues, please open an issue on GitHub.