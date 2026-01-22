"""
Video Frame Extractor

Simple tool to extract video frames as images without any preprocessing.
This is useful for testing whether the LSTM model can distinguish between
static and dynamic content based on raw frames only.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Generator, Optional


# Default configuration
DEFAULT_TARGET_SIZE = (640, 640)


def extract_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
) -> np.ndarray:
    """
    Extract and resize a single frame.

    Args:
        frame: Input frame (H, W, C)
        target_size: Target output size (width, height)

    Returns:
        Resized frame (target_size[1], target_size[0], C)
    """
    if frame.shape[:2][::-1] != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame


def extract_frames_from_video(
    video_path: str,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from a video.

    Args:
        video_path: Path to input video
        target_size: Target output size (width, height)

    Yields:
        Tuple of (frame_idx, frame) for each frame
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = extract_frame(frame, target_size)
        yield frame_idx, resized
        frame_idx += 1

    cap.release()


def extract_video_to_folder(
    video_path: str,
    output_dir: str,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    frame_interval: int = 1,
    max_frames: Optional[int] = None
) -> int:
    """
    Extract frames from video and save to folder.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        target_size: Target output size (width, height)
        frame_interval: Save every N-th frame (default: 1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        Number of frames saved
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_name}")
    print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"  Target size: {target_size}")
    print(f"  Frame interval: {frame_interval}")

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            resized = extract_frame(frame, target_size)
            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), resized)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"  Saved {saved_count} frames...")

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"  Done! Saved {saved_count} frames to {output_dir}")
    return saved_count


def preview_video(
    video_path: str,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    window_name: str = "Video Preview"
):
    """
    Preview video with resized frames.

    Controls:
        'q' or ESC: Stop playback
        'p' or SPACE: Pause/Resume

    Args:
        video_path: Path to input video
        target_size: Target output size (width, height)
        window_name: Name of the display window
    """
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    print(f"Previewing: {video_path.name}")
    print(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
    print(f"  Controls: 'q'=quit, 'p'/SPACE=pause")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, target_size[0], target_size[1])

    paused = False
    frame_idx = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame_idx += 1

        resized = extract_frame(frame, target_size)

        # Add frame info
        info_text = f"Frame: {frame_idx}/{total_frames}"
        if paused:
            info_text += " [PAUSED]"
        cv2.putText(resized, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, resized)

        key = cv2.waitKey(delay if not paused else 50) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple Video Frame Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames from video
  python video_frame_extractor.py --video input.mp4 --output ./frames

  # Extract every 5th frame
  python video_frame_extractor.py --video input.mp4 --output ./frames --interval 5

  # Extract max 100 frames
  python video_frame_extractor.py --video input.mp4 --output ./frames --max-frames 100

  # Preview video
  python video_frame_extractor.py --video input.mp4 --preview
        """
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./frames",
        help="Output directory for extracted frames (default: ./frames)"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("W", "H"),
        help="Target output size as width height (default: 640 640)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Save every N-th frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (default: all)"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Preview mode: display video without saving"
    )

    args = parser.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    target_size = tuple(args.size)

    if args.preview:
        preview_video(str(video_path), target_size)
    else:
        output_path = Path(args.output).expanduser()
        extract_video_to_folder(
            str(video_path),
            str(output_path),
            target_size=target_size,
            frame_interval=args.interval,
            max_frames=args.max_frames
        )

    return 0


if __name__ == "__main__":
    exit(main())
