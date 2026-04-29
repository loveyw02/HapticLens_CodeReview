"""Video reading utilities using FFmpeg.

This module provides an ffmpeg-based implementation of the
``get_video`` function which mirrors the interface of
``phase_utils.get_video`` but uses ffmpeg directly.  Using ffmpeg can
be considerably faster than relying on OpenCV for decoding.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from typing import Callable, List, Optional, Tuple

import numpy as np


def _probe_video(video_path: str, ffprobe_path: str = "ffprobe") -> Tuple[int, int, float, float]:
    """Return width, height, fps, and duration for ``video_path`` using ffprobe."""
    if shutil.which(ffprobe_path) is None:
        raise RuntimeError("ffprobe executable not found. Set FFPROBE_PATH environment variable to its location.")

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    lines = result.stdout.decode("utf-8").strip().splitlines()
    if len(lines) < 3:
        raise RuntimeError("Unable to parse ffprobe output")
    width = int(lines[0])
    height = int(lines[1])
    num, den = lines[2].split("/") if "/" in lines[2] else (lines[2], "1")
    fps = float(num) / float(den)
    # Duration might not always be available in stream, try format if needed
    duration = float(lines[3]) if len(lines) > 3 and lines[3] != "N/A" else 0.0

    # If duration not in stream, try format duration
    if duration == 0.0:
        cmd_format = [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result_format = subprocess.run(cmd_format, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        duration_str = result_format.stdout.decode("utf-8").strip()
        duration = float(duration_str) if duration_str != "N/A" else 0.0

    return width, height, fps, duration

class DurationExceededError(Exception):
    """Exception raised when video duration exceeds the specified maximum."""
    pass

def get_video(
    video_path: str,
    scale_factor: float,
    colorspace_func: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    load_size: Optional[Tuple[int, int]] = None,
    max_duration: Optional[float] = None,
) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """Read a video using FFmpeg.

    Parameters
    ----------
    video_path: str
        Path to the input video.
    scale_factor: float
        Scaling applied to width and height of the frames.
    colorspace_func: Callable
        Function applied to convert the frame from BGR after reading.
    max_size: Tuple[int, int], optional
        Maximum (width, height) for the loaded video. If specified,
        scale_factor will be adjusted to ensure video fits within these bounds.

    Returns
    -------
    frames: List[np.ndarray]
        List of processed frames.
    fs: float
        Frame rate of the video.
    resolution: Tuple[int, int]
        ``(width, height)`` of the returned frames.
    """

    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
    ffprobe_path = os.environ.get("FFPROBE_PATH", "ffprobe")

    if shutil.which(ffmpeg_path) is None:
        raise RuntimeError("ffmpeg executable not found. Set FFMPEG_PATH environment variable to its location.")

    # if video does not exist, raise an error
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    width, height, fps, duration = _probe_video(video_path, ffprobe_path)

    if max_duration is not None and duration > max_duration:
        raise DurationExceededError(f"Video duration {duration:.2f}s exceeds maximum allowed {max_duration:.2f}s")

    # Adjust scale_factor based on max_size if provided
    load_scale_factor = scale_factor
    if load_size is not None:
        max_width, max_height = load_size
        if height > max_height:
            load_scale_factor = min(load_scale_factor, max_height / height)
        if width > max_width:
            load_scale_factor = min(load_scale_factor, max_width / width)

    target_w = int(width * load_scale_factor)
    target_h = int(height * load_scale_factor)

    # Build ffmpeg command to read raw frames
    cmd = [
        ffmpeg_path,
        "-i",
        video_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
    ]

    if load_scale_factor != 1.0:
        cmd.extend(["-vf", f"scale={target_w}:{target_h}"])

    cmd.append("-")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None, "Failed to open ffmpeg process"

    frame_size = target_w * target_h * 3
    frames = []
    while True:
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) < frame_size:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((target_h, target_w, 3))
        frame = colorspace_func(frame.astype(np.float32) / 255.0)
        frames.append(frame)

    process.stdout.close()
    process.wait()

    return frames, fps, (target_w, target_h)