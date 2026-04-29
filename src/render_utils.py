import datetime
from typing import Optional

import cv2
import numpy as np


def draw_sparkline(
    signal: np.ndarray,
    perc: float,
    frame_size: tuple[int, int],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    playback_head_line: bool = False,
) -> np.ndarray:
    width, height = frame_size
    sparkline = np.zeros((height, width, 3), dtype=np.uint8)

    y_min = y_min if y_min is not None else np.min(signal)
    y_max = y_max if y_max is not None else np.max(signal)

    signal_scaled = (1 - (signal - y_min) / (y_max - y_min)) * (height - 10)
    x_vals = np.linspace(0, width - 1, len(signal)).astype(np.int32)
    y_vals = np.clip(signal_scaled, 0, height - 1).astype(np.int32)

    points = np.column_stack((x_vals, y_vals))
    cv2.polylines(sparkline, [points], isClosed=False, color=(255, 255, 255), thickness=1)

    current_x = int(perc * width)
    current_y = int(np.interp(current_x, x_vals, y_vals))
    if playback_head_line:
        cv2.line(sparkline, (current_x, 0), (current_x, height), (0, 0, 255), 1)
    else:
        cv2.circle(sparkline, (current_x, current_y), 4, (0, 0, 255), -1)

    return sparkline


def render_extract_point(
    bgr_mat: np.ndarray,
    extract_point: tuple[int, int],
    extract_radius: int,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    top_left = (extract_point[0] - extract_radius, extract_point[1] - extract_radius)
    bottom_right = (extract_point[0] + extract_radius, extract_point[1] + extract_radius)
    cv2.rectangle(bgr_mat, top_left, bottom_right, color=color, thickness=2)


def render_extract_point_rgb(bgr_mat: np.ndarray, extract_point: tuple[int, int], extract_radius: int) -> None:
    render_extract_point(bgr_mat, extract_point, extract_radius, color=(255, 0, 0))


def get_time_elapsed(tic: int, toc: int) -> str:
    time_elapsed = (toc - tic) / cv2.getTickFrequency()
    return str(datetime.timedelta(seconds=time_elapsed))
