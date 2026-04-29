
import cv2
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cnd

class SpatiotemporalSaliency:
    def __init__(self,
                 max_spatial_levels: int = 8,
                 max_center_level: int = 2,
                 max_d: int = 4,
                 temporal_buffer_len: int = 64,
                 use_padded_buffer: bool = True):
        self.max_spatial_levels = max_spatial_levels
        self.max_center_level = max_center_level # since it starts from 0 anyway
        self.max_d = max_d
        self.temporal_buffer_len = temporal_buffer_len
        self.use_padded_buffer = use_padded_buffer

    def build_spatial_pyramid(self, frame: np.ndarray) -> list:
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            base = frame.squeeze().astype(np.float32)
        else:
            base = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        pyr = [base]
        for _ in range(self.max_spatial_levels):
            pyr.append(cv2.pyrDown(pyr[-1])) # type: ignore
        return pyr

    def compute_spatial_saliency(self, frame: np.ndarray) -> np.ndarray:
        pyr = self.build_spatial_pyramid(frame)
        h, w = frame.shape[:2]
        maps = []
        for c in [2, 3, 4]:
            for d in [3, 4]:
                s = c + d
                if s < len(pyr):
                    C = pyr[c]
                    S = cv2.resize(pyr[s], (C.shape[1], C.shape[0]), interpolation=cv2.INTER_LINEAR)
                    if C.ndim == 2:
                        diff = np.abs(C - S)
                    else:
                        diff = np.linalg.norm(C - S, axis=2)
                    diff_resized = cv2.resize(diff, (w, h), interpolation=cv2.INTER_LINEAR)
                    maps.append(diff_resized)
        if not maps:
            return np.zeros((h, w), np.float32)
        sal = sum(maps)
        sal -= sal.min() # type: ignore
        if sal.max() > 0:
            sal /= sal.max()
        return sal.astype(np.float32)

    def compute_temporal_saliency(self, frames: list | np.ndarray) -> np.ndarray:
        # convert to LAB or keep single channel
        converted = []
        for f in frames:
            if f.ndim == 2 or (f.ndim == 3 and f.shape[2] == 1):
                converted.append(f.squeeze().astype(np.float32))
            else:
                converted.append(cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32))

        avg_pyr = {}
        max_level = self.max_center_level + self.max_d
        for n in range(max_level + 1):
            cnt = 2 ** n
            if len(converted) >= cnt:
                avg_pyr[n] = sum(converted[-cnt:]) / float(cnt)

        h, w = converted[-1].shape[:2]
        maps = []
        for c in range(self.max_center_level + 1):
            for d in [3, 4]:
                s = c + d
                if c in avg_pyr and s in avg_pyr:
                    C = avg_pyr[c]
                    S = avg_pyr[s]
                    if C.ndim == 2:
                        diff = np.abs(C - S)
                    else:
                        diff = np.linalg.norm(C - S, axis=2)
                    maps.append(diff)

        if not maps:
            return np.zeros((h, w), np.float32)
        sal = sum(maps)
        sal -= sal.min() # type: ignore
        if sal.max() > 0:
            sal /= sal.max()
        return sal.astype(np.float32)

    def compute_spatiotemporal_saliency_for_last_frame(self, buffer: list | np.ndarray) -> np.ndarray:
        spatial = self.compute_spatial_saliency(buffer[-1])
        temporal = self.compute_temporal_saliency(buffer)
        st = spatial * temporal
        st -= st.min()
        if st.max() > 0:
            st /= st.max()
        return st.astype(np.float32)

    def compute_spatiotemporal_saliency(self, all_frames: list) -> np.ndarray:
        # optional padding
        frames = all_frames
        af_padded = np.pad(np.array(all_frames), ((self.temporal_buffer_len, 0), (0,0), (0,0)), 'reflect')

        output = []
        for i in range(len(frames)):
            if self.use_padded_buffer:
                buffer = af_padded[i:i + self.temporal_buffer_len]
            else:
                buffer = all_frames[max(0, i - self.temporal_buffer_len):i + 1]
            output.append(self.compute_spatiotemporal_saliency_for_last_frame(buffer))
        return np.array(output)


def _srgb_to_linear(x: cp.ndarray) -> cp.ndarray:
    # x in [0,1]
    a = 0.055
    return cp.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def bgr_to_lab_cp(bgr_u8_or_f32: cp.ndarray) -> cp.ndarray:
    """
    Input:  HxWx3 BGR, uint8 [0,255] or float [0,255] or [0,1]
    Output: HxWx3 Lab float32 (L in [0,100], a/b roughly [-128,127])
    """
    x = bgr_u8_or_f32.astype(cp.float32)
    if x.max() > 1.5:
        x = x / 255.0

    # BGR -> RGB
    rgb = x[..., ::-1]

    # sRGB -> linear RGB
    rgb_lin = _srgb_to_linear(rgb)

    # linear RGB -> XYZ (D65)
    # Matrix for sRGB, D65
    M = cp.asarray([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=cp.float32)
    xyz = rgb_lin @ M.T

    # Normalize by reference white D65
    white = cp.asarray([0.95047, 1.00000, 1.08883], dtype=cp.float32)
    xyz_n = xyz / white

    # f(t) for Lab
    eps = 216 / 24389  # ~0.008856
    kappa = 24389 / 27 # ~903.3
    def f(t):
        return cp.where(t > eps, cp.cbrt(t), (kappa * t + 16) / 116)

    fxyz = f(xyz_n)
    fx, fy, fz = fxyz[..., 0], fxyz[..., 1], fxyz[..., 2]

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = cp.stack([L, a, b], axis=-1).astype(cp.float32)
    return lab

def pyr_down_cp(img: cp.ndarray) -> cp.ndarray:
    """
    pyrDown equivalent using only CuPy. OpenCV applies a 5x5 Gaussian
    (kernel 1/256 * [[1, 4, 6, 4, 1]^T x [1, 4, 6, 4, 1]]) and then
    decimates by two, with BORDER_REFLECT_101 padding. Using the
    separable kernel matches the CPU reference closely without leaving
    the GPU. <- thanks gpt
    """

    img = img.astype(cp.float32, copy=False)
    kernel = cp.asarray([1, 4, 6, 4, 1], dtype=cp.float32)

    def _blur(arr: cp.ndarray) -> cp.ndarray:
        tmp = cnd.convolve1d(arr, kernel, axis=0, mode="mirror")
        blurred = cnd.convolve1d(tmp, kernel, axis=1, mode="mirror")
        return blurred / 256.0

    if img.ndim == 2:
        blurred = _blur(img)
        return blurred[::2, ::2]
    else:
        blurred = _blur(img)
        return blurred[::2, ::2, :]

def resize_bilinear_cp(img: cp.ndarray, out_h: int, out_w: int) -> cp.ndarray:
    """
    Bilinear resize that mirrors OpenCV's coordinate mapping
    (align_corners=False, BORDER_REFLECT_101). Implemented with CuPy
    indexing so we stay on the GPU and get closer numerical parity with
    cv2.resize than ndimage.zoom.
    """

    img = img.astype(cp.float32, copy=False)
    in_h, in_w = img.shape[:2]

    if in_h == out_h and in_w == out_w:
        return img

    fy = in_h / out_h
    fx = in_w / out_w

    ys = (cp.arange(out_h, dtype=cp.float32) + 0.5) * fy - 0.5
    xs = (cp.arange(out_w, dtype=cp.float32) + 0.5) * fx - 0.5

    y0 = cp.floor(ys).astype(cp.int32)
    x0 = cp.floor(xs).astype(cp.int32)
    y1 = cp.clip(y0 + 1, 0, in_h - 1)
    x1 = cp.clip(x0 + 1, 0, in_w - 1)
    y0 = cp.clip(y0, 0, in_h - 1)
    x0 = cp.clip(x0, 0, in_w - 1)

    wy = ys - y0
    wx = xs - x0

    if img.ndim == 2:
        Ia = img[y0[:, None], x0]
        Ib = img[y0[:, None], x1]
        Ic = img[y1[:, None], x0]
        Id = img[y1[:, None], x1]
    else:
        Ia = img[y0[:, None, None], x0[:, None], :]
        Ib = img[y0[:, None, None], x1[:, None], :]
        Ic = img[y1[:, None, None], x0[:, None], :]
        Id = img[y1[:, None, None], x1[:, None], :]

    wy = wy[:, None]
    wx = wx[None, :]

    top = Ia * (1 - wx) + Ib * wx
    bottom = Ic * (1 - wx) + Id * wx
    out = top * (1 - wy) + bottom * wy
    return out.astype(cp.float32)


class SpatiotemporalSaliencyGPU(SpatiotemporalSaliency):
    def __init__(self,
                 max_spatial_levels: int = 8,
                 max_center_level: int = 2,
                 max_d: int = 4,
                 temporal_buffer_len: int = 64,
                 use_padded_buffer: bool = True):
        super().__init__(max_spatial_levels, max_center_level, max_d, temporal_buffer_len, use_padded_buffer)

    def build_spatial_pyramid(self, frame: np.ndarray) -> list[cp.ndarray]:
        gpu_frame = cp.asarray(frame)

        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            base = gpu_frame.squeeze().astype(cp.float32)
        else:
            base = bgr_to_lab_cp(gpu_frame)  # HxWx3 float32

        pyr = [base]
        for _ in range(self.max_spatial_levels):
            pyr.append(pyr_down_cp(pyr[-1]))
        return pyr

    def compute_spatial_saliency(self, frame: np.ndarray) -> cp.ndarray:
        pyr = self.build_spatial_pyramid(frame)
        h, w = frame.shape[:2]
        maps = []

        for c in [2, 3, 4]:
            for d in [3, 4]:
                s = c + d
                if s < len(pyr):
                    C = pyr[c]
                    S = pyr[s]

                    S_resized = resize_bilinear_cp(S, C.shape[0], C.shape[1])
                    diff = cp.abs(C - S_resized) if C.ndim == 2 else cp.linalg.norm(C - S_resized, axis=2)

                    diff_resized = resize_bilinear_cp(diff, h, w)
                    maps.append(diff_resized)

        if not maps:
            return cp.zeros((h, w), cp.float32)
        sal = sum(maps)
        sal -= sal.min() # type: ignore
        if sal.max() > 0:
            sal /= sal.max()
        return sal.astype(cp.float32)

    def compute_temporal_saliency(self, frames: list | np.ndarray) -> cp.ndarray:
        converted = []
        for f in frames:
            gf = cp.asarray(f)
            if f.ndim == 2 or (f.ndim == 3 and f.shape[2] == 1):
                converted.append(gf.squeeze().astype(cp.float32))
            else:
                converted.append(bgr_to_lab_cp(gf))

        avg_pyr = {}
        max_level = self.max_center_level + self.max_d
        for n in range(max_level + 1):
            cnt = 2 ** n
            if len(converted) >= cnt:
                avg_pyr[n] = sum(converted[-cnt:]) / float(cnt)

        h, w = converted[-1].shape[:2]
        maps = []
        for c in range(self.max_center_level + 1):
            for d in [3, 4]:
                s = c + d
                if c in avg_pyr and s in avg_pyr:
                    C = avg_pyr[c]
                    S = avg_pyr[s]
                    diff = cp.abs(C - S) if C.ndim == 2 else cp.linalg.norm(C - S, axis=2)
                    maps.append(diff)

        if not maps:
            return cp.zeros((h, w), cp.float32)
        sal = sum(maps)
        sal -= sal.min() # type: ignore
        if sal.max() > 0:
            sal /= sal.max()
        return sal.astype(cp.float32)

    def compute_spatiotemporal_saliency_for_last_frame(self, buffer: list | np.ndarray) -> np.ndarray:
        spatial = cp.asarray(self.compute_spatial_saliency(buffer[-1]))
        temporal = cp.asarray(self.compute_temporal_saliency(buffer))
        st = spatial * temporal
        st -= st.min()
        if st.max() > 0:
            st /= st.max()
        return cp.asnumpy(st.astype(cp.float32))

    def compute_spatiotemporal_saliency(self, all_frames: list) -> np.ndarray:
        # reuse CPU logic for buffer management
        return super().compute_spatiotemporal_saliency(all_frames)



if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Compute and display spatiotemporal saliency for a video')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scale factor for resizing frames')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark CPU vs GPU implementations')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for GPU acceleration')
    parser.add_argument('--compare-cpu-gpu', action='store_true', help='Compare CPU and GPU outputs on all frames')
    temporal_buffer_len = 64 # 2^6 (largest is level is 2+4)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame if scale factor is not 1.0
        if args.scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale_factor, fy=args.scale_factor, interpolation=cv2.INTER_LINEAR)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_frames.append(frame)

    print(f"Loaded {len(all_frames)} frames from {args.video} at {all_frames[0].shape[1]}x{all_frames[0].shape[0]}")

    def run_saliency(frames, use_cuda):
        ss = SpatiotemporalSaliency() if not use_cuda else SpatiotemporalSaliencyGPU()
        start_time = time.time()
        output = ss.compute_spatiotemporal_saliency(frames)
        end_time = time.time()
        elapsed_time = end_time - start_time
        framerate = len(frames) / elapsed_time if elapsed_time > 0 else 0
        return output, framerate, elapsed_time

    if args.compare_cpu_gpu:
        print("Comparing CPU and GPU implementations on all frames...")
        cpu_output, cpu_fps, cpu_time = run_saliency(all_frames, use_cuda=False)
        gpu_output, gpu_fps, gpu_time = run_saliency(all_frames, use_cuda=True)

        print(f"CPU: {cpu_fps:.2f} FPS, {cpu_time:.2f} seconds")
        print(f"GPU: {gpu_fps:.2f} FPS, {gpu_time:.2f} seconds")
        sse = np.sum((np.array(cpu_output) - np.array(gpu_output)) ** 2)
        assert sse < 1e-5, f"SSE between CPU and GPU outputs is too high: {sse:.4f}"

    elif args.benchmark:
        print(f"Benchmarking with {'CPU' if not args.use_cuda else 'GPU (cuda)'} implementation...")
        output, fps, elapsed_time = run_saliency(all_frames, use_cuda=args.use_cuda)
        print(f"{fps:.2f} FPS, {elapsed_time:.2f} seconds")
    else:
        use_cuda = args.use_cuda
        print(f"Using CUDA: {use_cuda}")
        while True:
            buffer = []
            start_time = time.time()  # Start time for framerate calculation
            ss = SpatiotemporalSaliency() if not use_cuda else SpatiotemporalSaliencyGPU()
            for i, frame in enumerate(all_frames):
                buffer.append(frame)
                # Maintain buffer size
                if len(buffer) > temporal_buffer_len:
                    buffer.pop(0)
                if buffer:
                    st_map = ss.compute_spatiotemporal_saliency_for_last_frame(buffer)
                    # Visualize overlay
                    sal_img = (st_map * 255).astype(np.uint8)
                    sal_bgr = cv2.cvtColor(sal_img, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Spatiotemporal Saliency', sal_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                end_time = time.time()  # End time for framerate calculation
                elapsed_time = end_time - start_time
                framerate = len(all_frames) / elapsed_time
                print(f"Framerate: {framerate:.2f} FPS")
                continue
            break
        cap.release()
        cv2.destroyAllWindows()

