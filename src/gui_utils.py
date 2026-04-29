import cv2
import numpy as np
import dearpygui.dearpygui as dpg
from typing import Tuple
from typing import TypeVar, Tuple


def bgr_to_float_rgb(bgr_img: np.ndarray) -> np.ndarray:
    rgba_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return (rgba_img.astype(np.float32) / 255.0).flatten()

def bgr_to_float_rgb_flat(bgr_img: np.ndarray, out_flat_array: np.ndarray) -> None:
    # np.copyto(out_flat_array, bgr_img[..., ::-1].astype(np.float32).ravel() / 255.0) # actually ~1.5ms slower for all runs
    temp_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    np.copyto(out_flat_array, temp_rgb.ravel())


T = TypeVar('T')
def update_from_gui(tag: str, current_value: T, change_flag: bool = False) -> Tuple[T, bool]:
    new_value: T = dpg.get_value(tag)
    changed = new_value != current_value
    return new_value, change_flag or changed

def angmag_to_rgba_vectorized(ang, mag_norm, agamma=2.5):
    hue = ang * (180.0 / np.pi)
    hue = np.mod(hue, 360)  # Make sure hue is within [0, 360)
    alpha = mag_norm ** agamma
    val = np.clip(mag_norm * 3, 0, 1)

    h = hue / 60.0
    i = np.floor(h).astype(int)
    f = h - i
    p = np.zeros_like(h)
    q = 1 - f
    t = f

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    mask = i == 0
    r[mask], g[mask], b[mask] = 1, t[mask], p[mask]

    mask = i == 1
    r[mask], g[mask], b[mask] = q[mask], 1, p[mask]

    mask = i == 2
    r[mask], g[mask], b[mask] = p[mask], 1, t[mask]

    mask = i == 3
    r[mask], g[mask], b[mask] = p[mask], q[mask], 1

    mask = i == 4
    r[mask], g[mask], b[mask] = t[mask], p[mask], 1

    mask = i >= 5
    r[mask], g[mask], b[mask] = 1, p[mask], q[mask]

    r *= val
    g *= val
    b *= val

    rgba = np.stack([r, g, b, alpha], axis=-1)
    rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return rgba

def generate_box_wireframe(min_corner, max_corner):
    """
    Generate the 12 edges of a 3D box wireframe.

    Parameters:
        min_corner (tuple): (x_min, y_min, z_min)
        max_corner (tuple): (x_max, y_max, z_max)

    Returns:
        list of tuple: Each tuple contains two points (start, end), representing an edge.
    """
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner

    corners = []
    for i in range(8):
        x = x1 if i & 1 else x0
        y = y1 if i & 2 else y0
        z = z1 if i & 4 else z0
        corners.append((x, y, z))

    edges = []
    for i in range(8):
        for bit in [1, 2, 4]:
            j = i ^ bit
            if i < j:
                edges.append((corners[i], corners[j]))

    return edges

def cubic_bezier_points(p1x, p1y, p2x, p2y, n=10):
    x0, y0 = 0.0, 0.0
    x1, y1 = p1x, p1y
    x2, y2 = p2x, p2y
    x3, y3 = 1.0, 1.0

    def bezier(t, p0, p1, p2, p3):
        u = 1 - t
        return (u**3)*p0 + 3*(u**2)*t*p1 + 3*u*(t**2)*p2 + (t**3)*p3

    def bezier_deriv(t, p0, p1, p2, p3):
        u = 1 - t
        return 3*(u**2)*(p1 - p0) + 6*u*t*(p2 - p1) + 3*(t**2)*(p3 - p2)

    xs = np.linspace(0.0, 1.0, n)
    t = xs.copy()

    for _ in range(6):
        xt = bezier(t, x0, x1, x2, x3)
        dxt = bezier_deriv(t, x0, x1, x2, x3)
        # Avoid division by zero
        dxt = np.where(dxt == 0, 1e-12, dxt)
        t -= (xt - xs) / dxt
        t = np.clip(t, 0.0, 1.0)

    ys = bezier(t, y0, y1, y2, y3)
    return np.column_stack([xs, ys])


import time
from threading import Timer
from typing import Any, Callable, Optional, Tuple, Dict

class Debouncer:
    """
    Leading-and-trailing debouncer/throttler.

    - First call goes out immediately (unless `force=False` and still in cool-down).
    - Further calls within `interval_sec` update the “pending” args,
      and exactly one trailing call fires at the end of that interval.
    - If you pass `force=True` to `call()`, it immediately sends
      (cancelling any pending trailing send), regardless of the cool-down.
    """
    def __init__(self, func: Callable[..., Any], interval_sec: float):
        self.func = func
        self.interval = interval_sec
        self._last_exec = 0.0                            # time.monotonic() of last send
        self._timer: Optional[Timer] = None              # threading.Timer for trailing call
        self._pending: Optional[Tuple[Tuple, Dict]] = None  # latest (args, kwargs) for trailing

    def _flush(self):
        """Internal: send the pending call at end of interval."""
        args, kwargs = self._pending  # type: ignore
        self.func(*args, **kwargs)
        self._last_exec = time.monotonic()
        self._pending = None
        self._timer = None

    def call(self, *args: Any, debounce_immediate: bool = False, **kwargs: Any) -> None:
        """
        Invoke the debounced function.

        :param force: if True, cancels any pending trailing call and
                      immediately invokes `func(*args, **kwargs)`.
        """
        now = time.monotonic()

        # --- force send immediately ---
        if debounce_immediate:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._pending = None
            self.func(*args, **kwargs)
            self._last_exec = now
            return

        elapsed = now - self._last_exec

        # --- if outside the interval window: send immediately ---
        if elapsed >= self.interval:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._pending = None
            self.func(*args, **kwargs)
            self._last_exec = now

        # --- otherwise, schedule/update trailing send ---
        else:
            # save latest args/kwargs
            self._pending = (args, kwargs)
            # only schedule one timer for the window’s end
            if not self._timer:
                delay = self.interval - elapsed
                self._timer = Timer(delay, self._flush)
                self._timer.start()

    __call__ = call  # so you can do: debouncer(arg) instead of debouncer.call(arg)
