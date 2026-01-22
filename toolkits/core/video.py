"""
Video and image file utilities.

Provides functions for video reading, frame extraction,
and file discovery.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Generator, Optional

from .constants import (
    DEFAULT_TARGET_SIZE,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_VIDEO_EXTENSIONS,
)


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
        Resized frame
    """
    if frame.shape[:2][::-1] != target_size:
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame


class VideoReader:
    """
    Context manager for reading video files.

    Example:
        with VideoReader('video.mp4') as reader:
            for frame_idx, frame in reader:
                process(frame)
    """

    def __init__(
        self,
        video_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            video_path: Path to video file
            target_size: Optional target size for frames (width, height)
        """
        self.video_path = str(video_path)
        self.target_size = target_size
        self.cap = None
        self._total_frames = 0
        self._fps = 0.0

    def __enter__(self) -> 'VideoReader':
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate over frames."""
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.target_size is not None:
                frame = extract_frame(frame, self.target_size)
            yield frame_idx, frame
            frame_idx += 1

    @property
    def total_frames(self) -> int:
        """Total number of frames in video."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Video frame rate."""
        return self._fps

    def seek(self, frame_idx: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_idx: Frame index to seek to

        Returns:
            True if seek was successful
        """
        if self.cap is None:
            return False
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret and self.target_size is not None:
            frame = extract_frame(frame, self.target_size)
        return ret, frame

    def get_frame_position(self) -> int:
        """Get current frame position."""
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))


def find_image_files(
    directory: Path,
    extensions: List[str] = None
) -> List[Path]:
    """
    Find all image files in a directory.

    Args:
        directory: Directory to search
        extensions: List of image extensions (default: png, jpg, jpeg, bmp)

    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = DEFAULT_IMAGE_EXTENSIONS

    image_files = []
    for ext in extensions:
        # Handle both with and without dot
        if not ext.startswith('.'):
            ext = f'.{ext}'
        image_files.extend(directory.glob(f'*{ext}'))

    return sorted(image_files)


def find_video_files(
    directory: Path,
    extensions: List[str] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Find all video files in a directory.

    Args:
        directory: Directory to search
        extensions: List of video extensions
        recursive: If True, search recursively

    Returns:
        Sorted list of video file paths
    """
    if extensions is None:
        extensions = DEFAULT_VIDEO_EXTENSIONS

    video_files = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        pattern = f'**/*{ext}' if recursive else f'*{ext}'
        video_files.extend(directory.glob(pattern))

    return sorted(video_files)


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
    with VideoReader(video_path, target_size) as reader:
        for frame_idx, frame in reader:
            yield frame_idx, frame


def save_frame(
    frame: np.ndarray,
    output_path: Path,
    quality: int = 95
) -> bool:
    """
    Save frame to file.

    Args:
        frame: Frame to save
        output_path: Output file path
        quality: JPEG quality (for jpg files)

    Returns:
        True if save was successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    else:
        params = []

    return cv2.imwrite(str(output_path), frame, params)
