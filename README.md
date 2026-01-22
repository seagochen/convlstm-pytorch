# ConvLSTM-PyTorch

A PyTorch implementation of ConvLSTM (Convolutional LSTM) for temporal sequence classification in video/image sequences.

## Overview

ConvLSTM-PyTorch is a spatiotemporal deep learning model designed to distinguish between **dynamic sequences** (with temporal motion patterns like flickering, spreading, moving objects) and **static sequences** (without significant movement patterns).

The model processes sequential video frames to capture temporal dynamics, outputting a probability heatmap where each cell represents a 32x32 pixel region of the original 640x640 image.

## Features

- Temporal pattern detection from video/image sequences
- Distinguishes dynamic content from static content through temporal analysis
- Outputs 20x20 probability heatmap for spatial localization
- Comprehensive data preprocessing pipeline
- Detailed training reports with visualization
- Support for batch inference

## Model Architecture

```
Input: (B, T, 3, 640, 640) - RGB video frames
                ↓
┌─────────────────────────────────────┐
│         Spatial Encoder (CNN)       │
│  5 conv layers with stride=2        │
│  640→320→160→80→40→20 (spatial)     │
│  3→16→32→64→128→64 (channels)       │
└─────────────────────────────────────┘
                ↓
        (B, T, 64, 20, 20)
                ↓
┌─────────────────────────────────────┐
│        ConvLSTM Layer               │
│  64 channels → 32 channels          │
│  Kernel size: 3x3                   │
│  Captures temporal dynamics         │
└─────────────────────────────────────┘
                ↓
        (B, 32, 20, 20)
                ↓
┌─────────────────────────────────────┐
│      Classification Head            │
│  1x1 conv: 32 → 1 channel           │
│  Sigmoid activation                 │
└─────────────────────────────────────┘
                ↓
Output: (B, 1, 20, 20) - Probability heatmap
```

## Installation

```bash
git clone https://github.com/your-username/convlstm-pytorch.git
cd convlstm-pytorch
pip install -r requirements.txt
```

### Dependencies

- torch>=1.9.0
- numpy>=1.19.0
- opencv-python>=4.5.0
- pandas>=1.3.0
- tqdm>=4.60.0
- pyyaml>=5.4.0
- matplotlib>=3.4.0
- scikit-learn>=0.24.0

## Usage

### Training

Using configuration file:
```bash
python scripts/train.py --config training.yaml
```

Using command-line arguments:
```bash
python scripts/train.py \
  --data_root /path/to/data \
  --epochs 10 \
  --batch_size 16 \
  --lr 0.0001
```

### Inference

Single folder detection:
```bash
python scripts/detect.py \
  --source /path/to/frames \
  --weights checkpoints/best_model.pth \
  --save_viz \
  --output ./results
```

Batch processing (multiple folders):
```bash
python scripts/detect.py \
  --source /path/to/data \
  --weights checkpoints/best_model.pth \
  --batch
```

## Configuration

Key parameters in `training.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | - | Dataset root directory (required) |
| `seq_length` | 10 | Number of consecutive frames per sample |
| `stride` | 3 | Sliding window step for sample generation |
| `target_size` | [640, 640] | Target image size (H, W) |
| `batch_size` | 16 | Batch size for training |
| `epochs` | 10 | Number of training epochs |
| `lr` | 0.0001 | Learning rate |
| `val_ratio` | 0.2 | Validation set proportion |
| `project` | runs/train | Base output directory |
| `name` | exp | Experiment name (auto-increments) |
| `report_interval` | 5 | Generate reports every N epochs |

## Data Format

### Directory Structure

```
data_root/
├── data.csv          # Manifest file
├── dynamic/          # Dynamic samples (with motion)
│   ├── sample_001/
│   │   ├── frame_000001.png
│   │   ├── frame_000002.png
│   │   └── ...
│   └── sample_002/
└── static/           # Static samples (no motion)
    ├── sample_001/
    └── ...
```

### data.csv Format

```csv
folder_name,type
dynamic/sample_001,dynamic
dynamic/sample_002,dynamic
static/sample_001,static
```

## Data Preprocessing Tools

The `toolkits/` directory provides utilities for data preparation:

| Tool | Purpose |
|------|---------|
| `generate_data_csv.py` | Generate data.csv from folder structure |
| `video_frame_extractor.py` | Extract frames from videos |
| `dynamic_segment_processor.py` | YOLO segmentation + masking |
| `dynamic_bbox_processor.py` | YOLO bounding box processing |
| `image_to_static_frames.py` | Generate static samples |
| `sequence_augmentation.py` | Apply consistent augmentation to sequences |
| `check_corrupted_images.py` | Validate image integrity |

## Project Structure

```
convlstm-pytorch/
├── convlstm/                    # Main package
│   ├── models/
│   │   ├── convlstm.py         # Core ConvLSTM implementation
│   │   └── temporal_classifier.py  # Classification model
│   ├── training/
│   │   └── trainer.py          # Training orchestration
│   └── utils/
│       ├── data/
│       │   └── dataset.py      # Dataset and dataloaders
│       └── callbacks/          # LR schedulers, EMA
├── scripts/
│   ├── train.py                # Training entry point
│   └── detect.py               # Inference script
├── toolkits/                   # Data preprocessing utilities
├── runs/                       # Training output directory
│   └── train/
│       └── exp/                # Experiment directory (auto-increments)
│           ├── weights/        # Model checkpoints
│           │   ├── best.pth    # Best model
│           │   └── last.pth    # Latest checkpoint
│           └── reports/        # Detection reports
├── training.yaml               # Configuration file
└── requirements.txt            # Dependencies
```

## Training Features

- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Optimizer**: Adam with learning rate scheduling
- **Scheduler**: ReduceLROnPlateau (reduces LR by 0.5x after 3 epochs of no improvement)
- **Checkpointing**: Saves best model and latest model
- **Reports**: Generates detection reports with heatmap visualizations

## Use Cases

This model can be applied to various temporal classification tasks:

- **Motion Detection**: Distinguish moving objects from static backgrounds
- **Activity Recognition**: Detect dynamic activities in surveillance footage
- **Quality Control**: Identify defects with temporal patterns in manufacturing
- **Medical Imaging**: Detect dynamic patterns in medical video sequences
- **Environmental Monitoring**: Detect temporal changes in satellite/drone imagery

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
