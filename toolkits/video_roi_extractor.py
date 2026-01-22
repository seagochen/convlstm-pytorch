"""
Video ROI Extractor

Extracts ROI (Region of Interest) from video for fire/smoke detection.
- Fire: Uses segmentation mask with EMA smoothing + Gaussian blur for stable edges
- Smoke: Uses union bbox across all frames for consistent ROI

Processing Rules:
- fire (class 0): Extract pixels using segmentation mask
- smoke (class 2): Extract region using bounding box
- person (class 1): Ignored (negative sample)

When both fire and smoke are detected:
- Fire regions use mask extraction
- Smoke regions use bbox extraction
- Combined mask includes both
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from ultralytics import YOLO

# Detection mode type
DetectMode = Literal['all', 'fire', 'smoke']

from core import (
    CLASS_NAMES,
    CLASS_COLORS,
    SEGMENT_CLASSES,
    BBOX_CLASSES,
    TARGET_CLASSES,
    INFERENCE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
    MASK_BLUR_KERNEL_SIZE,
    BBOX_PADDING_RATIO,
)
from core.constants import VALIDATION_PANEL_SIZE
from core.detection import (
    extract_detections_from_yolo,
    compute_bbox_union,
    add_bbox_padding,
)
from core.augmentation import AugmentationParameters, apply_augmentation


class MaskEMA:
    """
    Exponential Moving Average for mask smoothing.
    Reduces flickering by smoothly transitioning mask regions over time.
    """
    def __init__(self, alpha: float = EMA_ALPHA, blur_kernel: int = MASK_BLUR_KERNEL_SIZE):
        self.alpha = alpha
        self.blur_kernel = blur_kernel
        self.ema_mask = None

    def update(self, current_mask: np.ndarray) -> np.ndarray:
        """
        Update EMA mask and return smoothed mask.

        Args:
            current_mask: Current frame's binary mask

        Returns:
            Smoothed binary mask
        """
        if self.ema_mask is None:
            self.ema_mask = current_mask.astype(np.float32)
        else:
            self.ema_mask = self.alpha * current_mask.astype(np.float32) + (1 - self.alpha) * self.ema_mask

        # Threshold to get binary mask
        smoothed_mask = (self.ema_mask > 0.5).astype(np.uint8)

        # Apply morphological operations and Gaussian blur for edge smoothing
        smoothed_mask = self._smooth_edges(smoothed_mask)

        return smoothed_mask

    def _smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations and Gaussian blur for smooth edges."""
        if mask.sum() == 0:
            return mask

        kernel_size = self.blur_kernel
        mask_uint8 = mask.astype(np.uint8) * 255

        # Morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # Gaussian blur for smooth edges
        mask_blurred = cv2.GaussianBlur(mask_closed, (kernel_size, kernel_size), 0)

        # Re-threshold
        _, mask_smooth = cv2.threshold(mask_blurred, 127, 1, cv2.THRESH_BINARY)

        return mask_smooth.astype(np.uint8)

    def reset(self):
        """Reset EMA state."""
        self.ema_mask = None


def analyze_video_for_smoke_bbox(
    video_path: str,
    model: YOLO,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    sample_rate: int = 5,
    padding_ratio: float = BBOX_PADDING_RATIO,
    detect_mode: DetectMode = 'all'
) -> Tuple[Optional[Tuple[int, int, int, int]], Dict]:
    """
    Analyze video to compute stable union bbox for smoke detections.

    Args:
        video_path: Path to video file
        model: YOLO segmentation model
        conf_threshold: Confidence threshold
        sample_rate: Process every Nth frame
        padding_ratio: Padding around union bbox
        detect_mode: Detection mode ('all', 'fire', 'smoke')

    Returns:
        Tuple of (union_bbox, stats_dict)
    """
    # Skip analysis if only detecting fire (no smoke bbox needed)
    if detect_mode == 'fire':
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Detection mode: fire only (skipping smoke bbox analysis)")
        return None, {
            'total_frames': total_frames,
            'sampled_frames': 0,
            'frames_with_fire': 0,
            'frames_with_smoke': 0,
            'smoke_bboxes_count': 0,
            'smoke_union_bbox': None,
            'fps': fps,
            'detect_mode': detect_mode
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    mode_str = "smoke only" if detect_mode == 'smoke' else "fire + smoke"
    print(f"Analyzing video for smoke bbox union... (mode: {mode_str})")
    print(f"Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Sample rate: every {sample_rate} frame(s)")

    all_smoke_bboxes = []
    frame_idx = 0
    sampled_frames = 0
    frames_with_smoke = 0
    frames_with_fire = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate == 0:
            resized_frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
            h, w = resized_frame.shape[:2]

            results = model(resized_frame, conf=conf_threshold, verbose=False)[0]
            detections = extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, conf_threshold)

            # Separate fire and smoke detections
            smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]
            fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]

            if smoke_dets:
                frames_with_smoke += 1
                for det in smoke_dets:
                    all_smoke_bboxes.append((det.x1, det.y1, det.x2, det.y2))

            if fire_dets:
                frames_with_fire += 1

            sampled_frames += 1

            if sampled_frames % 50 == 0:
                print(f"  Sampled {sampled_frames} frames... (smoke: {len(all_smoke_bboxes)} bboxes)")

        frame_idx += 1

    cap.release()

    # Compute union bbox for smoke
    smoke_union_bbox = compute_bbox_union(all_smoke_bboxes)

    if smoke_union_bbox is not None:
        h, w = INFERENCE_SIZE
        smoke_union_bbox = add_bbox_padding(smoke_union_bbox, (h, w), padding_ratio)

    print(f"\nAnalysis complete:")
    print(f"  Sampled {sampled_frames}/{total_frames} frames")
    print(f"  Frames with fire: {frames_with_fire} ({frames_with_fire/sampled_frames*100:.1f}%)")
    print(f"  Frames with smoke: {frames_with_smoke} ({frames_with_smoke/sampled_frames*100:.1f}%)")

    if smoke_union_bbox:
        print(f"  Smoke union bbox: ({smoke_union_bbox[0]}, {smoke_union_bbox[1]}) -> ({smoke_union_bbox[2]}, {smoke_union_bbox[3]})")

    stats = {
        'total_frames': total_frames,
        'sampled_frames': sampled_frames,
        'frames_with_fire': frames_with_fire,
        'frames_with_smoke': frames_with_smoke,
        'smoke_bboxes_count': len(all_smoke_bboxes),
        'smoke_union_bbox': smoke_union_bbox,
        'fps': fps,
        'detect_mode': detect_mode
    }

    return smoke_union_bbox, stats


def preprocess_frame(
    frame: np.ndarray,
    model: YOLO,
    smoke_union_bbox: Optional[Tuple[int, int, int, int]],
    mask_ema: Optional[MaskEMA],
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    augment_params: Optional[AugmentationParameters] = None,
    detect_mode: DetectMode = 'all'
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess a single frame for LSTM training.

    Processing logic:
    - Fire detected: Use segmentation mask (with EMA smoothing)
    - Smoke detected: Use pre-computed union bbox
    - Both detected: Combine mask and bbox regions

    Args:
        frame: Input frame
        model: YOLO segmentation model
        smoke_union_bbox: Pre-computed union bbox for smoke (None if no smoke in video)
        mask_ema: MaskEMA instance for fire mask smoothing
        conf_threshold: Confidence threshold
        augment_params: Optional augmentation parameters
        detect_mode: Detection mode ('all', 'fire', 'smoke')

    Returns:
        Tuple of (preprocessed_frame, info_dict)
    """
    resized_frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
    h, w = resized_frame.shape[:2]

    results = model(resized_frame, conf=conf_threshold, verbose=False)[0]
    detections = extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, conf_threshold)

    # Separate fire and smoke detections based on detect_mode
    if detect_mode == 'fire':
        fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]
        smoke_dets = []
    elif detect_mode == 'smoke':
        fire_dets = []
        smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]
    else:  # 'all'
        fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]
        smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]

    # Build detection info (only for target classes based on detect_mode)
    det_info = [
        {'class_id': d.class_id, 'class_name': CLASS_NAMES[d.class_id], 'confidence': d.confidence}
        for d in (fire_dets + smoke_dets)
    ]

    # Initialize combined mask
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    has_fire = len(fire_dets) > 0
    has_smoke = len(smoke_dets) > 0 and smoke_union_bbox is not None

    # Process fire: segmentation mask
    if has_fire:
        fire_mask = np.zeros((h, w), dtype=np.uint8)
        for det in fire_dets:
            if det.mask is not None:
                if det.mask.shape != (h, w):
                    mask_resized = cv2.resize(det.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = det.mask.astype(np.uint8)
                fire_mask = np.maximum(fire_mask, mask_resized)

        # Apply EMA smoothing
        if mask_ema is not None:
            fire_mask = mask_ema.update(fire_mask)

        combined_mask = np.maximum(combined_mask, fire_mask)

    # Process smoke: union bbox
    if has_smoke:
        x1, y1, x2, y2 = smoke_union_bbox
        bbox_mask = np.zeros((h, w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 1
        combined_mask = np.maximum(combined_mask, bbox_mask)

    # Apply combined mask to frame
    if combined_mask.sum() > 0:
        output_frame = resized_frame.copy()
        output_frame[combined_mask == 0] = 0
        roi_type = 'fire+smoke' if (has_fire and has_smoke) else ('fire' if has_fire else 'smoke')
    else:
        output_frame = np.zeros_like(resized_frame)
        roi_type = 'none'

    # Apply augmentation if parameters provided
    if augment_params is not None and combined_mask.sum() > 0:
        output_frame = apply_augmentation(output_frame, augment_params)

    info = {
        'type': roi_type,
        'has_fire': has_fire,
        'has_smoke': has_smoke,
        'detections': det_info
    }

    return output_frame, info


def preprocess_video_for_lstm(
    video_path: str,
    model: YOLO,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_ema: bool = True,
    ema_alpha: float = EMA_ALPHA,
    augment: bool = False,
    augment_seed: Optional[int] = None,
    detect_mode: DetectMode = 'all'
):
    """
    Generator that yields preprocessed frames from a video.

    First analyzes video to compute smoke union bbox, then processes each frame.

    Args:
        video_path: Path to video file
        model: YOLO segmentation model
        conf_threshold: Confidence threshold
        use_ema: Enable EMA smoothing for fire masks
        ema_alpha: EMA smoothing factor
        augment: Apply data augmentation to output frames
        augment_seed: Random seed for reproducible augmentation
        detect_mode: Detection mode ('all', 'fire', 'smoke')

    Yields:
        Tuple of (frame_idx, preprocessed_frame, info_dict)
    """
    # Step 1: Analyze video for smoke union bbox
    smoke_union_bbox, stats = analyze_video_for_smoke_bbox(
        video_path, model, conf_threshold, detect_mode=detect_mode
    )

    print(f"\nProcessing video with ROI extraction... (detect: {detect_mode})")
    if use_ema and detect_mode != 'smoke':
        print(f"Fire mask EMA smoothing enabled (alpha={ema_alpha:.2f})")
    if augment:
        print(f"Data augmentation enabled (seed={augment_seed})")
    print()

    # Step 2: Setup augmentation parameters (consistent across all frames)
    augment_params = AugmentationParameters(seed=augment_seed) if augment else None

    # Step 3: Process each frame
    mask_ema = MaskEMA(alpha=ema_alpha) if use_ema else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed, info = preprocess_frame(
            frame, model, smoke_union_bbox, mask_ema, conf_threshold, augment_params,
            detect_mode=detect_mode
        )

        yield frame_idx, preprocessed, info
        frame_idx += 1

    cap.release()


def validate_processing(
    video_path: str,
    model: YOLO,
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    window_name: str = "ROI Processing Validation",
    detect_mode: DetectMode = 'all'
):
    """
    Validate ROI processing with side-by-side view.

    Left panel: Original frame with detection overlays
    Right panel: Preprocessed frame (ROI extracted)
    """
    video_path = Path(video_path)
    video_name = video_path.stem

    # First analyze video
    smoke_union_bbox, stats = analyze_video_for_smoke_bbox(
        str(video_path), model, conf_threshold, detect_mode=detect_mode
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nValidating: {video_name} (detect: {detect_mode})")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"\nControls:")
    print(f"  'q' or ESC: Stop playback")
    print(f"  'p' or SPACE: Pause/Resume")
    print(f"  LEFT/RIGHT: Seek -/+ 10 frames")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, VALIDATION_PANEL_SIZE[0] * 2, VALIDATION_PANEL_SIZE[1])

    mask_ema = MaskEMA()
    paused = False
    frame_idx = 0
    frame = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                cv2.waitKey(0)
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame is None:
            continue

        # Process frame
        preprocessed, info = preprocess_frame(
            frame, model, smoke_union_bbox, mask_ema, conf_threshold,
            detect_mode=detect_mode
        )

        # Left panel: original with overlays
        resized_frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        left_frame = resized_frame.copy()

        # Draw smoke union bbox (only if detecting smoke)
        if smoke_union_bbox is not None and detect_mode != 'fire':
            x1, y1, x2, y2 = smoke_union_bbox
            cv2.rectangle(left_frame, (x1, y1), (x2, y2), CLASS_COLORS[2], 2)
            cv2.putText(left_frame, "smoke bbox", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[2], 2)

        # Draw fire detections (only if detecting fire)
        if detect_mode != 'smoke':
            results = model(resized_frame, conf=conf_threshold, verbose=False)[0]
            h, w = resized_frame.shape[:2]
            detections = extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, conf_threshold)

            for det in detections:
                if det.class_id in SEGMENT_CLASSES:
                    # Fire: show mask contour
                    if det.mask is not None:
                        contours, _ = cv2.findContours(
                            det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(left_frame, contours, -1, CLASS_COLORS[0], 2)

                    x1, y1, x2, y2 = det.bbox
                    cv2.putText(left_frame, f"fire {det.confidence:.2f}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[0], 2)

        # Right panel: preprocessed
        right_frame = preprocessed.copy()

        # Resize panels
        left_frame = cv2.resize(left_frame, VALIDATION_PANEL_SIZE, interpolation=cv2.INTER_LINEAR)
        right_frame = cv2.resize(right_frame, VALIDATION_PANEL_SIZE, interpolation=cv2.INTER_LINEAR)

        combined = np.hstack([left_frame, right_frame])

        # Add info overlay
        info_text = f"Frame: {frame_idx}/{total_frames} | Type: {info['type']}"
        if paused:
            info_text += " [PAUSED]"
        cv2.putText(combined, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, combined)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == 81 or key == 2:  # LEFT
            new_pos = max(0, frame_idx - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            mask_ema.reset()  # Reset EMA when seeking
            if paused:
                ret, frame = cap.read()
                if ret:
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elif key == 83 or key == 3:  # RIGHT
            new_pos = min(total_frames - 1, frame_idx + 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            mask_ema.reset()  # Reset EMA when seeking
            if paused:
                ret, frame = cap.read()
                if ret:
                    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Video ROI Extractor for Fire/Smoke Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Rules:
  - fire (class 0):  Segmentation mask with EMA smoothing
  - smoke (class 2): Union bbox across all frames
  - person (class 1): Ignored (negative sample)

Examples:
  # Validate processing (preview mode)
  python video_roi_extractor.py --model best.pt --video input.mp4 --validate

  # Preprocess video for LSTM training
  python video_roi_extractor.py --model best.pt --video input.mp4 --output ./output

  # Detect only fire (ignore smoke)
  python video_roi_extractor.py --model best.pt --video input.mp4 --output ./output --detect fire

  # Detect only smoke (ignore fire)
  python video_roi_extractor.py --model best.pt --video input.mp4 --output ./output --detect smoke

  # With data augmentation (single augmented dataset)
  python video_roi_extractor.py --model best.pt --video input.mp4 --output ./output --augment

  # Generate multiple augmented datasets
  python video_roi_extractor.py --model best.pt --video input.mp4 --output ./output --augment -n 3
        """
    )

    parser.add_argument("--model", "-m", type=str, required=True, help="Path to YOLO segmentation model")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for preprocessed frames")
    parser.add_argument("--validate", action="store_true", help="Run validation mode (preview)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--detect", "-d", type=str, choices=['all', 'fire', 'smoke'], default='all',
                       help="Detection mode: 'all' (default), 'fire' only, or 'smoke' only")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA smoothing for fire masks")
    parser.add_argument("--augment", "-a", action="store_true", help="Apply data augmentation to output frames")
    parser.add_argument("-n", "--num-datasets", type=int, default=1,
                       help="Number of augmented datasets to generate (default: 1, requires --augment)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible augmentation")

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    video_path = Path(args.video).expanduser()

    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    print()

    if args.validate or args.output is None:
        validate_processing(str(video_path), model, args.conf, detect_mode=args.detect)
    else:
        output_path = Path(args.output).expanduser()
        video_name = video_path.stem

        # Determine number of datasets to generate
        num_datasets = args.num_datasets if args.augment else 1

        print(f"Preprocessing video: {video_name}")
        print(f"Output directory: {output_path}")
        print(f"Detection mode: {args.detect}")
        if args.augment:
            print(f"Augmentation: ON (generating {num_datasets} dataset(s))")
        else:
            print(f"Augmentation: OFF")
        print()

        total_saved = 0

        for dataset_idx in range(num_datasets):
            # Determine output directory and augmentation seed
            if num_datasets > 1:
                dataset_output = output_path / f"aug_{dataset_idx + 1:03d}"
                print(f"[{dataset_idx + 1}/{num_datasets}] Generating dataset...")
            else:
                dataset_output = output_path

            dataset_output.mkdir(parents=True, exist_ok=True)

            # Compute augmentation seed
            augment_seed = None
            if args.augment:
                augment_seed = args.seed + dataset_idx if args.seed is not None else None

            frame_count = 0
            fire_count = 0
            smoke_count = 0
            both_count = 0
            none_count = 0
            saved_count = 0

            for frame_idx, preprocessed, info in preprocess_video_for_lstm(
                str(video_path), model, args.conf,
                use_ema=not args.no_ema,
                augment=args.augment,
                augment_seed=augment_seed,
                detect_mode=args.detect
            ):
                frame_count += 1

                if info['type'] == 'fire+smoke':
                    both_count += 1
                elif info['type'] == 'fire':
                    fire_count += 1
                elif info['type'] == 'smoke':
                    smoke_count += 1
                else:
                    none_count += 1
                    continue  # Skip frames with no detections

                frame_path = dataset_output / f"{video_name}_frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), preprocessed)
                saved_count += 1

                if saved_count % 100 == 0:
                    print(f"  Saved {saved_count} frames...")

            total_saved += saved_count

            if num_datasets > 1:
                print(f"  Dataset {dataset_idx + 1}: {saved_count} frames saved\n")

        print(f"\nCompleted! Processed {frame_count} frames per dataset")
        if num_datasets > 1:
            print(f"  Total datasets: {num_datasets}")
            print(f"  Total frames saved: {total_saved}")
        else:
            print(f"  Frames saved: {total_saved}")
        print(f"  Fire only: {fire_count}")
        print(f"  Smoke only: {smoke_count}")
        print(f"  Fire + Smoke: {both_count}")
        print(f"  No detection (skipped): {none_count}")

    return 0


if __name__ == "__main__":
    exit(main())
