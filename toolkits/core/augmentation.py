"""
Image augmentation utilities.

Provides various augmentation functions for image processing,
including noise, blur, and color adjustments.
"""

import cv2
import numpy as np
import random
from typing import Dict


class SmoothParameter:
    """
    Smooth parameter that varies continuously with optional periodic oscillation.
    Uses interpolated random values for natural variation.
    """
    def __init__(
        self,
        base_value: float,
        variance: float,
        num_steps: int,
        smoothness: float = 0.1
    ):
        """
        Args:
            base_value: Base value around which parameter varies
            variance: Maximum deviation from base value
            num_steps: Total number of steps/frames
            smoothness: How smooth the transitions are (0-1, lower = smoother)
        """
        self.base_value = base_value
        self.variance = variance
        self.values = self._generate_smooth_curve(num_steps, smoothness)

    def _generate_smooth_curve(self, num_steps: int, smoothness: float) -> np.ndarray:
        """Generate smooth varying curve using interpolated random values."""
        num_control_points = max(int(num_steps * smoothness), 3)
        control_values = np.random.uniform(-self.variance, self.variance, num_control_points)

        x_control = np.linspace(0, num_steps - 1, num_control_points)
        x_smooth = np.arange(num_steps)
        smooth_curve = np.interp(x_smooth, x_control, control_values)

        return self.base_value + smooth_curve

    def get(self, index: int) -> float:
        """Get parameter value at given index."""
        return self.values[min(index, len(self.values) - 1)]


class AugmentationParameters:
    """
    Container for a consistent set of augmentation parameters.
    Once sampled, these parameters are applied uniformly to all frames.
    """
    def __init__(self, seed: int = None):
        """
        Sample random augmentation parameters.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Sample parameters from predefined ranges
        self.sp_noise = random.uniform(0.0, 0.02)
        self.gaussian_std = random.uniform(0, 20)
        self.poisson_scale = random.uniform(0, 0.5)
        self.motion_blur_size = random.randint(1, 15)
        self.motion_blur_angle = random.uniform(0, 360)
        self.brightness = random.uniform(-40, 40)
        self.contrast = random.uniform(0.7, 1.3)
        self.hue_shift = random.uniform(-20, 20)
        self.saturation = random.uniform(0.8, 1.2)

    def to_dict(self) -> Dict:
        """Export parameters as dictionary for logging."""
        return {
            'salt_pepper_noise': self.sp_noise,
            'gaussian_noise_std': self.gaussian_std,
            'poisson_scale': self.poisson_scale,
            'motion_blur_size': self.motion_blur_size,
            'motion_blur_angle': self.motion_blur_angle,
            'brightness': self.brightness,
            'contrast': self.contrast,
            'hue_shift': self.hue_shift,
            'saturation': self.saturation
        }

    def __str__(self) -> str:
        """Human-readable parameter summary."""
        return (
            f"SP_noise={self.sp_noise:.4f}, Gauss_std={self.gaussian_std:.2f}, "
            f"Poisson={self.poisson_scale:.2f}, MotionBlur={self.motion_blur_size}px@{self.motion_blur_angle:.1f}deg, "
            f"Bright={self.brightness:.1f}, Contrast={self.contrast:.2f}, "
            f"Hue={self.hue_shift:.1f}, Sat={self.saturation:.2f}"
        )


def add_salt_pepper_noise(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Add salt and pepper noise.

    Args:
        image: Input image
        noise_ratio: Total noise ratio (0-1), split equally between salt and pepper

    Returns:
        Noisy image
    """
    if noise_ratio <= 0:
        return image.copy()

    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    salt_ratio = noise_ratio / 2
    pepper_ratio = noise_ratio / 2

    # Salt (white)
    num_salt = int(total_pixels * salt_ratio)
    if num_salt > 0:
        coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255

    # Pepper (black)
    num_pepper = int(total_pixels * pepper_ratio)
    if num_pepper > 0:
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0

    return noisy


def add_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """
    Add Gaussian noise (electronic/thermal noise).

    Args:
        image: Input image
        std: Standard deviation of noise

    Returns:
        Noisy image
    """
    if std <= 0:
        return image.copy()

    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_poisson_noise(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Add Poisson noise (photon/shot noise).

    Args:
        image: Input image
        scale: Scaling factor for noise intensity (higher = more noise)

    Returns:
        Noisy image
    """
    if scale <= 0:
        return image.copy()

    vals = image.astype(np.float32) * scale
    noisy = np.random.poisson(vals) / scale
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_motion_blur(image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
    """
    Add motion blur.

    Args:
        image: Input image
        kernel_size: Size of motion blur kernel
        angle: Angle of motion in degrees

    Returns:
        Blurred image
    """
    if kernel_size <= 1:
        return image.copy()

    # Ensure odd kernel size
    kernel_size = int(kernel_size) | 1

    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size

    # Rotate kernel
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: float,
    contrast: float
) -> np.ndarray:
    """
    Adjust brightness and contrast.

    Args:
        image: Input image
        brightness: Brightness offset (-100 to 100)
        contrast: Contrast multiplier (0.5 to 2.0)

    Returns:
        Adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted


def adjust_color_jitter(
    image: np.ndarray,
    hue_shift: float,
    saturation_scale: float
) -> np.ndarray:
    """
    Apply color jitter (HSV adjustments).

    Args:
        image: Input image (BGR)
        hue_shift: Hue shift in degrees (-180 to 180)
        saturation_scale: Saturation multiplier (0.5 to 1.5)

    Returns:
        Color adjusted image
    """
    if abs(hue_shift) < 0.1 and abs(saturation_scale - 1.0) < 0.01:
        return image.copy()

    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

    # Adjust saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)

    # Convert back to BGR
    hsv = hsv.astype(np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted


def apply_augmentation(image: np.ndarray, params: AugmentationParameters) -> np.ndarray:
    """
    Apply augmentation pipeline with fixed parameters.

    Args:
        image: Input image
        params: Augmentation parameters to apply

    Returns:
        Augmented image
    """
    frame = image.copy()

    # Apply augmentations in sequence
    # 1. Color jitter
    frame = adjust_color_jitter(frame, params.hue_shift, params.saturation)

    # 2. Brightness/Contrast
    frame = adjust_brightness_contrast(frame, params.brightness, params.contrast)

    # 3. Motion blur
    if params.motion_blur_size > 1:
        frame = add_motion_blur(frame, params.motion_blur_size, params.motion_blur_angle)

    # 4. Noises
    if params.poisson_scale > 0:
        frame = add_poisson_noise(frame, params.poisson_scale)

    if params.gaussian_std > 0:
        frame = add_gaussian_noise(frame, params.gaussian_std)

    if params.sp_noise > 0:
        frame = add_salt_pepper_noise(frame, params.sp_noise)

    return frame
