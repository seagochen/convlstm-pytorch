"""
Shared constants for video preprocessing toolkit.

Contains class configurations, color palettes, and default parameters
used across multiple processing tools.
"""

from typing import Dict, Set, Tuple, List

# =============================================================================
# Class Configuration (matching data.yaml: fire, person, smoke)
# =============================================================================

CLASS_NAMES: Dict[int, str] = {
    0: "fire",
    1: "person",  # negative sample, ignored in processing
    2: "smoke"
}

# Classes to process (person is excluded as negative sample)
TARGET_CLASSES: Set[int] = {0, 2}  # fire, smoke

# Fire class: use segmentation mask with EMA smoothing
SEGMENT_CLASSES: Set[int] = {0}  # fire

# Smoke class: use bounding box with union strategy
BBOX_CLASSES: Set[int] = {2}  # smoke

# Negative sample classes (ignored in processing)
NEGATIVE_CLASSES: Set[int] = {1}  # person

# Color palette for classes (BGR format for OpenCV)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),      # fire - red
    1: (0, 255, 0),      # person - green (for visualization only)
    2: (128, 128, 128),  # smoke - gray
}

# Track colors for visualization (BGR format)
TRACK_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
]

# =============================================================================
# Processing Parameters
# =============================================================================

INFERENCE_SIZE: Tuple[int, int] = (640, 640)  # YOLO inference size
DEFAULT_TARGET_SIZE: Tuple[int, int] = (640, 640)  # Default output size
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5

# Display settings
PREVIEW_DISPLAY_SIZE: Tuple[int, int] = (1820, 1024)  # 1K resolution (1024p, 16:9)
VALIDATION_PANEL_SIZE: Tuple[int, int] = (640, 480)

# =============================================================================
# ROI Processing Parameters
# =============================================================================

# EMA (Exponential Moving Average) for mask smoothing
EMA_ALPHA: float = 0.2  # Smoothing factor (lower = smoother, more lag)

# Gaussian blur for mask edge smoothing
MASK_BLUR_KERNEL_SIZE: int = 5  # Kernel size for edge smoothing

# Bbox padding
BBOX_PADDING_RATIO: float = 0.05  # 5% padding around union bbox

# =============================================================================
# File Extensions
# =============================================================================

DEFAULT_IMAGE_EXTENSIONS: List[str] = [
    "png", "jpg", "jpeg", "bmp",
    "PNG", "JPG", "JPEG", "BMP"
]

DEFAULT_VIDEO_EXTENSIONS: List[str] = [
    "mp4", "avi", "mov", "mkv", "webm",
    "MP4", "AVI", "MOV", "MKV", "WEBM"
]

# =============================================================================
# Augmentation Parameters
# =============================================================================

DEFAULT_NUM_AUGMENTED_FRAMES: int = 500
DEFAULT_NUM_AUGMENTED_DATASETS: int = 3


# =============================================================================
# Legacy aliases (for backward compatibility)
# =============================================================================

# These are kept for backward compatibility but should be migrated
MASK_ONLY_CLASSES = SEGMENT_CLASSES
BBOX_ONLY_CLASSES = BBOX_CLASSES
