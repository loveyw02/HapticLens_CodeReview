import time
import cupyx
import numpy as np
import cupy as cp
# from cupy_interp1d import interp1d as cupy_interp1d # needed some manual patch or something
import scipy

HAPTIC_SAMPLE_RATE = 8000

def first_order_finite_diff(signal: np.ndarray, fs: float) -> np.ndarray:
    # dt = 1 / fs  # Sampling period
    dt = 1
    velocity = np.diff(signal, prepend=signal[0]) / dt  # First-order finite difference
    return velocity

def second_order_finite_diff(signal: np.ndarray, fs: float) -> np.ndarray:
    # dt = 1 / fs
    dt = 1
    acceleration = (np.roll(signal, -1) - 2 * signal + np.roll(signal, 1)) / dt**2
    acceleration[0] = acceleration[1]  # Handle edge cases
    acceleration[-1] = acceleration[-2]
    return acceleration

def upsample_signal(signal: np.ndarray, t_orig: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    interpolator = scipy.interpolate.interp1d(t_orig, signal, kind='cubic', bounds_error=False, fill_value=(signal[0], signal[-1]), assume_sorted=True)
    return interpolator(t_new)

def create_hap_signal(signal_at_coords: np.ndarray, video_fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a haptic signal based on the input signal at the specified coordinates.
    The signal is resampled to a target sample rate and then modulated
    to create a haptic signal.
    Args:
        signal_at_coords (np.ndarray): The input signal at the specified coordinates. Expected to be [0, 1]
        video_fs (float): The sample rate of the input video.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The haptic signal, acceleration, acceleration magnitude, and resampled acceleration.
    """
    # todo: CONSIDER APPLYING A GAMMA CORRECTION x^(1/0.6) maybe (stevens power law for vibration at 250hz)
    start = time.time()
    duration = len(signal_at_coords) / video_fs
    t_orig = np.linspace(0, duration, len(signal_at_coords), endpoint=False)
    t_new = np.linspace(0, duration, int(duration * HAPTIC_SAMPLE_RATE), endpoint=False)

    signal_resampled = upsample_signal(signal_at_coords, t_orig, t_new) # ~33% func time (5ms)
    s_min = signal_resampled.min()
    s_range = signal_resampled.max() - s_min
    s_range = max(s_range, 0.05)  # Avoid amplifying extremely small signals
    signal_resampled_norm = (signal_resampled - s_min) / s_range

    accel = first_order_finite_diff(signal_at_coords, video_fs) # ~33% func time (5ms)
    accel_resampled = upsample_signal(accel, t_orig, t_new)

    # ~33% in generation (5ms)
    assert len(signal_resampled_norm) == len(accel_resampled), f"Signal and Accel resampled lengths do not match: {len(signal_resampled_norm)}, {len(accel_resampled)}"
    base_freq = 220 # A3
    freqs = base_freq + accel_resampled * 180
    delta_phases = 2 * np.pi * freqs / HAPTIC_SAMPLE_RATE
    phase_acc = np.cumsum(delta_phases) % (2 * np.pi)
    hap_signal = np.sin(phase_acc) * signal_resampled_norm

    # hap_signal: np.ndarray = hap_signal / np.max(np.abs(hap_signal)) * 0.9
    hap_signal = hap_signal.astype(np.float32)

    return hap_signal, signal_resampled_norm, accel, accel_resampled#, accel_resampled_mag

def first_order_finite_diff_cupy(signal: cp.ndarray, fs: float) -> cp.ndarray:
    # dt = 1 / fs  # Sampling period
    dt = 1
    velocity = cp.diff(signal, prepend=signal[0]) / dt  # First-order finite difference
    return velocity

# def upsample_signal_cupy(signal: cp.ndarray, t_orig: cp.ndarray, t_new: cp.ndarray) -> cp.ndarray:
#     interpolator = cupy_interp1d(t_orig, signal, kind='cubic', bounds_error=False, fill_value=(signal[0], signal[-1]), assume_sorted=True) # type: ignore
#     return interpolator(t_new)

# def create_hap_signal_cupy(signal_at_coords: cp.ndarray, video_fs: float) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
#     """
#     Create a haptic signal based on the input signal at the specified coordinates.
#     The signal is resampled to a target sample rate and then modulated
#     to create a haptic signal.
#     Args:
#         signal_at_coords (np.ndarray): The input signal at the specified coordinates. Expected to be [0, 1]
#         video_fs (float): The sample rate of the input video.
#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The haptic signal, acceleration, acceleration magnitude, and resampled acceleration.
#     """
#     # todo: CONSIDER APPLYING A GAMMA CORRECTION x^(1/0.6) maybe (stevens power law for vibration at 250hz)
#     start = time.time()
#     duration = len(signal_at_coords) / video_fs
#     t_orig = cp.linspace(0, duration, len(signal_at_coords), endpoint=False)
#     t_new = cp.linspace(0, duration, int(duration * HAPTIC_SAMPLE_RATE), endpoint=False)

#     signal_resampled = upsample_signal_cupy(signal_at_coords, t_orig, t_new) # ~33% func time (5ms)
#     s_min = signal_resampled.min()
#     s_range = signal_resampled.max() - s_min
#     s_range = max(s_range, 0.05)  # Avoid amplifying extremely small signals
#     signal_resampled_norm = (signal_resampled - s_min) / s_range

#     accel = first_order_finite_diff_cupy(signal_at_coords, video_fs) # ~33% func time (5ms)
#     accel_resampled = upsample_signal_cupy(accel, t_orig, t_new)

#     # ~33% in generation (5ms)
#     assert len(signal_resampled_norm) == len(accel_resampled), f"Signal and Accel resampled lengths do not match: {len(signal_resampled_norm)}, {len(accel_resampled)}"
#     base_freq = 220 # A3
#     freqs = base_freq + accel_resampled * 180
#     delta_phases = 2 * cp.pi * freqs / HAPTIC_SAMPLE_RATE
#     phase_acc = cp.cumsum(delta_phases) % (2 * cp.pi)
#     hap_signal = cp.sin(phase_acc) * signal_resampled_norm

#     # hap_signal: cp.ndarray = hap_signal / cp.max(cp.abs(hap_signal)) * 0.9
#     hap_signal = hap_signal.astype(cp.float32)

#     return hap_signal, signal_resampled_norm, accel, accel_resampled#, accel_resampled_mag

def compute_integral_cupy(frames: cp.ndarray, dtype=cp.float64) -> cp.ndarray:
    F, H, W = frames.shape
    x = frames.astype(dtype, copy=False)
    S = cp.zeros((F, H + 1, W + 1), dtype=dtype)
    S[:, 1:, 1:] = cp.cumsum(cp.cumsum(x, axis=1), axis=2)
    return S

def sample_box_mean_cupy(S: cp.ndarray, center: tuple[int, int], radius: int) -> cp.ndarray:
    F, Hp1, Wp1 = S.shape
    H, W = Hp1 - 1, Wp1 - 1

    x, y = center
    x = min(max(x, radius), W - radius)
    y = min(max(y, radius), H - radius)

    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = min(x + radius, W)
    y2 = min(y + radius, H)
    area = (x2 - x1) * (y2 - y1)

    A = S[:, y1, x1]
    B = S[:, y1, x2]
    C = S[:, y2, x1]
    D = S[:, y2, x2]
    rect_sum = D - B - C + A

    return rect_sum / area


def get_average_in_radius_cupy(frames: cp.ndarray, center: tuple[int, int], radius: int) -> cp.ndarray:
    _f, h, w = frames.shape
    x, y = center
    x = np.clip(x, radius, w - radius)
    y = np.clip(y, radius, h - radius)
    signal = frames[:, y - radius:y + radius, x - radius:x + radius]
    return cp.mean(signal, axis=(1, 2))

def get_average_in_radius(frames: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    _f, h, w = frames.shape
    x, y = center
    x = np.clip(x, radius, w - radius)
    y = np.clip(y, radius, h - radius)
    signal = frames[:, y - radius:y + radius, x - radius:x + radius]
    return np.mean(signal, axis=(1, 2))

def get_rms_in_radius(frames: np.ndarray, center: tuple[int, int], radius: int) -> np.ndarray:
    _f, h, w = frames.shape
    x, y = center
    x = np.clip(x, radius, w - radius)
    y = np.clip(y, radius, h - radius)
    signal = frames[:, y - radius:y + radius, x - radius:x + radius]
    return np.sqrt(np.mean(signal**2, axis=(1, 2)))

def extract_hap_signal(
    pds_mag: np.ndarray,
    extract_point: tuple[int, int],
    extract_radius: int,
    video_fs: float
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError("Not maintained, check gui.py")
    mag_norm = (pds_mag - np.min(pds_mag)) / (np.max(pds_mag) - np.min(pds_mag))

    signal_at_coords = get_average_in_radius(mag_norm, extract_point, extract_radius)
    # print(f"signal_at_coords min, max: {np.min(signal_at_coords), np.max(signal_at_coords)}")

    hap_signal, *_ = create_hap_signal(signal_at_coords, video_fs)

    return hap_signal, signal_at_coords
