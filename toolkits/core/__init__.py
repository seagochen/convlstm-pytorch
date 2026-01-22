"""
Core modules for video preprocessing toolkit.

This package provides shared functionality for video frame processing,
including augmentation, detection, and video utilities.
"""

from .constants import (
    CLASS_NAMES,
    CLASS_COLORS,
    TRACK_COLORS,
    TARGET_CLASSES,
    SEGMENT_CLASSES,
    BBOX_CLASSES,
    NEGATIVE_CLASSES,
    INFERENCE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TARGET_SIZE,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_VIDEO_EXTENSIONS,
    EMA_ALPHA,
    MASK_BLUR_KERNEL_SIZE,
    BBOX_PADDING_RATIO,
    # Legacy aliases
    MASK_ONLY_CLASSES,
    BBOX_ONLY_CLASSES,
)

from .detection import (
    Detection,
    extract_detections_from_yolo,
    get_class_color,
    compute_bbox_union,
    add_bbox_padding,
    apply_bbox_mask,
    apply_segmentation_mask,
)

from .augmentation import (
    add_salt_pepper_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_motion_blur,
    adjust_brightness_contrast,
    adjust_color_jitter,
    SmoothParameter,
    AugmentationParameters,
    apply_augmentation,
)

from .video import (
    extract_frame,
    VideoReader,
    find_image_files,
    find_video_files,
)

__all__ = [
    # Constants - Class configuration
    'CLASS_NAMES',
    'CLASS_COLORS',
    'TRACK_COLORS',
    'TARGET_CLASSES',
    'SEGMENT_CLASSES',
    'BBOX_CLASSES',
    'NEGATIVE_CLASSES',
    # Constants - Processing
    'INFERENCE_SIZE',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'DEFAULT_TARGET_SIZE',
    'DEFAULT_IMAGE_EXTENSIONS',
    'DEFAULT_VIDEO_EXTENSIONS',
    'EMA_ALPHA',
    'MASK_BLUR_KERNEL_SIZE',
    'BBOX_PADDING_RATIO',
    # Legacy aliases
    'MASK_ONLY_CLASSES',
    'BBOX_ONLY_CLASSES',
    # Detection
    'Detection',
    'extract_detections_from_yolo',
    'get_class_color',
    'compute_bbox_union',
    'add_bbox_padding',
    'apply_bbox_mask',
    'apply_segmentation_mask',
    # Augmentation
    'add_salt_pepper_noise',
    'add_gaussian_noise',
    'add_poisson_noise',
    'add_motion_blur',
    'adjust_brightness_contrast',
    'adjust_color_jitter',
    'SmoothParameter',
    'AugmentationParameters',
    'apply_augmentation',
    # Video
    'extract_frame',
    'VideoReader',
    'find_image_files',
    'find_video_files',
]
