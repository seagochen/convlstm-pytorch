"""
Static Frames Generator

Generates static frame sequences from a single image.
Can optionally apply continuous augmentations to simulate camera behavior.

Usage:
  # Basic: Generate 500 identical frames (no augmentation)
  python static_frames_generator.py -i image.jpg -o ./output

  # With augmentation: Generate 500 frames with varying noise, blur, etc.
  python static_frames_generator.py -i image.jpg -o ./output --augment
"""

import cv2
from pathlib import Path
import argparse
from typing import Tuple

from core import DEFAULT_TARGET_SIZE
from core.constants import DEFAULT_NUM_AUGMENTED_FRAMES
from core.augmentation import (
    SmoothParameter,
    add_salt_pepper_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_motion_blur,
    adjust_brightness_contrast,
    adjust_color_jitter,
)


def generate_static_frames(
    image_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    num_frames: int = DEFAULT_NUM_AUGMENTED_FRAMES,
    augment: bool = False,
    verbose: bool = True
) -> int:
    """
    Generate static frame sequences from a single image.

    Args:
        image_path: Path to input image
        output_dir: Output directory for frames
        target_size: Target size (width, height)
        num_frames: Total number of frames to generate
        augment: Whether to apply continuous augmentations
        verbose: Print progress messages

    Returns:
        Total number of frames generated
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = image_path.stem

    if verbose:
        mode = "with augmentation" if augment else "without augmentation"
        print(f"Generating {num_frames} static frames ({mode})...")

    # Setup augmentation parameters if needed
    if augment:
        params = {
            'sp_noise': SmoothParameter(0.005, 0.015, num_frames, smoothness=0.15),
            'gaussian_noise': SmoothParameter(5, 15, num_frames, smoothness=0.12),
            'poisson_scale': SmoothParameter(0, 0.3, num_frames, smoothness=0.18),
            'motion_blur_size': SmoothParameter(1, 8, num_frames, smoothness=0.2),
            'motion_blur_angle': SmoothParameter(0, 180, num_frames, smoothness=0.25),
            'brightness': SmoothParameter(0, 30, num_frames, smoothness=0.1),
            'contrast': SmoothParameter(1.0, 0.3, num_frames, smoothness=0.1),
            'hue_shift': SmoothParameter(0, 15, num_frames, smoothness=0.15),
            'saturation': SmoothParameter(1.0, 0.2, num_frames, smoothness=0.12),
        }

    for i in range(num_frames):
        if augment:
            frame = resized.copy()

            # Apply augmentations in sequence
            hue = params['hue_shift'].get(i)
            sat = params['saturation'].get(i)
            frame = adjust_color_jitter(frame, hue, sat)

            bright = params['brightness'].get(i)
            cont = params['contrast'].get(i)
            frame = adjust_brightness_contrast(frame, bright, cont)

            mb_size = max(1, int(params['motion_blur_size'].get(i)))
            mb_angle = params['motion_blur_angle'].get(i)
            if mb_size > 1:
                frame = add_motion_blur(frame, mb_size, mb_angle)

            poisson_scale = params['poisson_scale'].get(i)
            if poisson_scale > 0:
                frame = add_poisson_noise(frame, poisson_scale)

            gaussian_std = params['gaussian_noise'].get(i)
            if gaussian_std > 0:
                frame = add_gaussian_noise(frame, gaussian_std)

            sp_ratio = params['sp_noise'].get(i)
            if sp_ratio > 0:
                frame = add_salt_pepper_noise(frame, sp_ratio)
        else:
            frame = resized

        # Save frame
        output_path = output_dir / f"{image_stem}_{i:03d}.png"
        cv2.imwrite(str(output_path), frame)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_frames} frames...")

    if verbose:
        print(f"Generated {num_frames} frames")

    return num_frames


def main():
    parser = argparse.ArgumentParser(
        description="Static Frames Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Description:
  Generates static frame sequences from a single image.
  By default, creates identical copies. Use --augment to add
  continuous variations (noise, blur, brightness, etc.).

Examples:
  # Generate 500 identical frames
  python static_frames_generator.py -i image.jpg -o ./output

  # Generate 500 frames with augmentation
  python static_frames_generator.py -i image.jpg -o ./output --augment

  # Custom number of frames
  python static_frames_generator.py -i image.jpg -o ./output -n 1000 --augment
        """
    )

    parser.add_argument("-i", "--input", type=str, required=True, help="Input image file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    parser.add_argument("-n", "--num-frames", type=int, default=DEFAULT_NUM_AUGMENTED_FRAMES,
                       help=f"Number of frames to generate (default: {DEFAULT_NUM_AUGMENTED_FRAMES})")
    parser.add_argument("--size", type=int, nargs=2, default=list(DEFAULT_TARGET_SIZE),
                       metavar=("W", "H"), help=f"Target output size (default: {DEFAULT_TARGET_SIZE})")
    parser.add_argument("--augment", "-a", action="store_true",
                       help="Apply continuous augmentations (noise, blur, brightness)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages")

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        print(f"Error: Input image not found: {input_path}")
        return 1

    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        return 1

    verbose = not args.quiet

    if verbose:
        print("=" * 60)
        print("Static Frames Generator")
        print("=" * 60)
        print(f"  Input image:  {input_path.name}")
        print(f"  Output dir:   {output_dir}")
        print(f"  Target size:  {args.size[0]}x{args.size[1]}")
        print(f"  Num frames:   {args.num_frames}")
        print(f"  Augmentation: {'ON' if args.augment else 'OFF'}")
        print("=" * 60)
        print()

    try:
        total_frames = generate_static_frames(
            image_path=input_path,
            output_dir=output_dir,
            target_size=tuple(args.size),
            num_frames=args.num_frames,
            augment=args.augment,
            verbose=verbose
        )

        if verbose:
            print()
            print(f"Success! Generated {total_frames} frames")
            print(f"  Output: {output_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
