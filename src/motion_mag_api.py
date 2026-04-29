import cv2
import numpy as np
import cupy as cp
from acceleration_based import AccelBased
from phase_based_processing import PhaseBased
from video_reader import get_video
from phase_utils import bandpass_filter, bgr2yiq, get_fft2_batch, yiq2rgb
from async_video_writer import AsyncFFmpegVideoWriter
from render_utils import draw_sparkline, render_extract_point
from steerable_pyramid import SteerablePyramid
from typing import Callable, Optional, Tuple, List
from typing import Any
from enum import Enum

NONE_MAT: Any = None
def load_video(
    colorspace: str,
    video_path: str,
    scale_factor: float,
    load_duration: Optional[float] = None,
    load_size: Optional[Tuple[int, int]] = None,
    max_duration: Optional[float] = None,
) -> Tuple[List[np.ndarray], float, Tuple[int, int], int, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    # get forward and inverse colorspace functions
    # inverse colorspace obtains frames back in BGR representation
    if colorspace == "luma1":
        colorspace_func = lambda x: bgr2yiq(x)[:, :, 0]
        inv_colorspace = lambda x: cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        inv_colorspace_rgb = lambda x: cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    elif colorspace in ("luma3", "yiq"):
        colorspace_func = bgr2yiq
        inv_colorspace = lambda x: cv2.cvtColor((yiq2rgb(x) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inv_colorspace_rgb = lambda x: (yiq2rgb(x) * 255).clip(0, 255).astype(np.uint8)

    elif colorspace == "gray":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        inv_colorspace = lambda x: cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        inv_colorspace_rgb = lambda x: cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    elif colorspace == "rgb":
        colorspace_func = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        inv_colorspace = lambda x: cv2.cvtColor((x * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        inv_colorspace_rgb = lambda x: (x * 255).clip(0, 255).astype(np.uint8)

    else:
        raise ValueError(f"Currently unsupported colorspace: {colorspace}")

    # get scaled video frames in proper colorspace and sample frequency fs
    frames, video_fs, (v_w, v_h) = get_video(video_path=video_path, scale_factor=scale_factor, colorspace_func=colorspace_func, load_size=load_size, max_duration=max_duration)

    if load_duration is not None:
        max_frames = int(np.floor(load_duration * video_fs))
        frames = frames[:max_frames]

    num_frames = len(frames)

    return frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func, inv_colorspace_rgb

def create_transfer_function(
    sample_frequency: float,
    video_fs: float,
    num_frames: int,
    freq_lo: float,
    freq_hi: float,
) -> cp.ndarray:
    # get sample frequency fs, use input sample freuqency if valid
    if sample_frequency > 0.0:
        fs = sample_frequency
        print(f"Sample Frequency overriden with input!: fs = {fs}, video_fs = {video_fs} \n")
    else:
        fs = video_fs
        # print(f"Detected Sample Frequency: fs = {fs} \n")

    # Get Bandpass Filter Transfer function
    transfer_function = bandpass_filter(freq_lo, freq_hi, fs, num_frames)
    return transfer_function

def create_tensors(
    frames: List[np.ndarray],
    ref_idx: int,
    pyramid_type: str,
    batch_size: int,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, SteerablePyramid]:
    # get reference frame info
    ref_frame = frames[ref_idx]
    h, w = ref_frame.shape[:2]

    # Get Complex Steerable Pyramid Object
    max_depth = int(np.floor(np.log2(np.min(np.array((w, h))))) - 2)
    if pyramid_type == "full_octave":
        csp = SteerablePyramid(depth=max_depth, orientations=4, filters_per_octave=1, twidth=1.0, complex_pyr=True)

    elif pyramid_type == "half_octave":
        csp = SteerablePyramid(depth=max_depth, orientations=8, filters_per_octave=2, twidth=0.75, complex_pyr=True)

    else:
        raise ValueError(f"Currently unsupported pyramid type: {pyramid_type}")
    # elif pyramid_type == "smooth_half_octave":
    #     csp = SuboctaveSP(depth=max_depth, orientations=8, filters_per_octave=2, cos_order=6, complex_pyr=True)

    # elif pyramid_type == "smooth_quarter_octave":
    #     csp = SuboctaveSP(depth=max_depth, orientations=8, filters_per_octave=4, cos_order=6, complex_pyr=True)

    # get Complex Steerable Pyramid Filters
    filters, crops, motion_dirs = csp.get_filters(h, w, cropped=False)
    filters_tensor = cp.array(filters, dtype=cp.float32)
    filter_dir_tensor = cp.array(motion_dirs, dtype=cp.float32)

    # if (filters_tensor.shape[0] % batch_size) != 0:
    #     print(f"WARNING! Selected Batch size: {batch_size} might not be compatible with the number of Filters: {filters_tensor.shape[0]}! \n")
    #     raise ValueError("Please select a batch size that is compatible with the number of filters")
    if ((filters_tensor.shape[0] - 2) % batch_size) != 0: # subtract low and high pass
        print(f"WARNING! Selected Batch size: {batch_size} might not be compatible with the number of Filters without low+high pass: {filters_tensor.shape[0]} - 2! \n")
        raise ValueError("Please select a batch size that is compatible with the number of filters")

    frames_tensor = cp.array(frames, dtype=cp.float32)

    return filters_tensor, filter_dir_tensor, frames_tensor, csp

class PhaseAlgorithm(Enum):
    LINEAR = "linear"
    ACCEL = "accel"

def get_phase_deltas_and_mag_video(
    sigma: float,
    video_fs: float,
    transfer_function: cp.ndarray,
    phase_mag: float,
    attenuate: bool,
    ref_idx: int,
    batch_size: int,
    colorspace: str,
    frames_tensor: cp.ndarray,
    filters_tensor: cp.ndarray,
    filter_dir_tensor: cp.ndarray,
    csp: SteerablePyramid,
    EPS: float,
    ontick: Optional[Callable[[float], None]] = None,
    phase_algo: PhaseAlgorithm = PhaseAlgorithm.LINEAR
) -> Tuple[np.ndarray, np.ndarray]:
    if phase_algo == PhaseAlgorithm.LINEAR:
        pb = PhaseBased(sigma, video_fs, transfer_function, phase_mag, attenuate, ref_idx, batch_size, EPS)
    elif phase_algo == PhaseAlgorithm.ACCEL:
        pb = AccelBased(sigma, video_fs, transfer_function, phase_mag, attenuate, ref_idx, batch_size, EPS)
    else:
        raise ValueError(f"Unsupported phase algorithm: {phase_algo}")

    phase_delta_dir_sum = None

    if colorspace == "yiq" or colorspace == "rgb":
        # process each channel individually
        result_video = cp.zeros_like(frames_tensor)
        for c in range(frames_tensor.shape[-1]):
            video_dft = get_fft2_batch(frames_tensor[:, :, :, c])
            result_video[:, :, :, c], _ = pb.process_single_channel(frames_tensor[:, :, :, c], filters_tensor, video_dft, filter_dir_tensor, csp)

    elif colorspace == "luma3":
        # process single Luma channel and add back to full color image
        result_video = frames_tensor.copy()
        video_dft = get_fft2_batch(frames_tensor[:, :, :, 0])
        result_video[:, :, :, 0], _ = pb.process_single_channel(frames_tensor[:, :, :, 0], filters_tensor, video_dft, filter_dir_tensor, csp)

    else:
        # process single channel
        video_dft = get_fft2_batch(frames_tensor)
        result_video, phase_delta_dir_sum = pb.process_single_channel(frames_tensor, filters_tensor, video_dft, filter_dir_tensor, csp, ontick=ontick)

    if phase_delta_dir_sum is None:
        raise ValueError("only single channel processing is supported currently") # could probably average across channels or smth

    result_video = result_video.get()
    phase_delta_dir_sum = phase_delta_dir_sum.get()

    return result_video, phase_delta_dir_sum

def generate_output_video(
    frames: List[np.ndarray],
    result_video: np.ndarray,
    og_w: int,
    og_h: int,
    num_frames: int,
    inv_colorspace: Callable[[np.ndarray], np.ndarray],
    # phase_delta_dir_sum: np.ndarray,
    pds_mag_all: np.ndarray,
    pds_ang_all: np.ndarray,
    extract_point: Tuple[int, int],
    extract_radius: int,
    signal_at_coords: np.ndarray,
    hap_signal: np.ndarray,
    video_save_path: str,
    video_codec: str,
    video_fs: float,
) -> None:
    vsep = np.zeros((og_h, 2, 3)).astype(np.uint8)
    bgr_null = np.zeros((og_h, og_w, 3)).astype(np.uint8)

    # stacked_frames = []
    out = None
    prev_frame_gpu = None
    optical_flow = None
    farneback = cv2.cuda_FarnebackOpticalFlow.create( # type: ignore
        numLevels=3,
        pyrScale=0.5,
        fastPyramids=False,
        winSize=15,
        numIters=3,
        polyN=5,
        polySigma=1.2,
        flags=0
    )

    for vid_idx in range(num_frames):

        # get BGR frames
        bgr_frame = inv_colorspace(frames[vid_idx])
        bgr_processed = inv_colorspace(result_video[vid_idx])
        # bgr_pda = inv_colorspace(phase_delta_avg[vid_idx])

        next_frame_cpu = (cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(bgr_processed, cv2.COLOR_BGR2GRAY))
        next_frame_gpu = (
            cv2.cuda_GpuMat(), # type: ignore
            cv2.cuda_GpuMat(), # type: ignore
        )
        next_frame_gpu[0].upload(next_frame_cpu[0])
        # next_frame_gpu[1].upload(next_frame_cpu[1])
        # flow = (
        #     cv2.cuda_GpuMat(), # type: ignore
        #     cv2.cuda_GpuMat(), # type: ignore
        # )
        flow = (
            cv2.cuda_GpuMat(next_frame_gpu[0].size(), cv2.CV_32FC2), # type: ignore
            cv2.cuda_GpuMat(next_frame_gpu[1].size(), cv2.CV_32FC2), # type: ignore
        )

        if prev_frame_gpu is not None:
            # optical_flow = (cv2.calcOpticalFlowFarneback(prev_frame[0], next_frame_cpu[0], None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0),
            #                 cv2.calcOpticalFlowFarneback(prev_frame[1], next_frame_cpu[1], None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0))
            farneback.calc(prev_frame_gpu[0], next_frame_gpu[0], flow[0])
            # farneback.calc(prev_frame_gpu[1], next_frame_gpu[1], flow[1])

            optical_flow = (
                np.zeros((og_h, og_w, 2), dtype=np.float32),
                np.zeros((og_h, og_w, 2), dtype=np.float32),
            )
            flow[0].download(optical_flow[0])
            # flow[1].download(optical_flow[1])

        prev_frame_gpu = next_frame_gpu


        # phase delta avg:
        # bgr_pda = cv2.cvtColor(cv2.normalize(phase_delta_avg[vid_idx], NONE_MAT, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLOR_GRAY2BGR)
        # result_phase:
        # bgr_pda = cv2.cvtColor(cv2.normalize(result_phase[vid_idx], NONE_MAT, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLOR_GRAY2BGR)
        # directional phase deltas:
        hsv_pda = np.zeros_like(bgr_frame)
        # pds_mag, pds_ang = cv2.cartToPolar(phase_delta_dir_sum[vid_idx][..., 0], phase_delta_dir_sum[vid_idx][..., 1])
        pds_mag, pds_ang = pds_mag_all[vid_idx], pds_ang_all[vid_idx]
        hsv_pda[..., 0] = pds_ang * 180 / np.pi / 2
        hsv_pda[..., 1] = 255
        hsv_pda[..., 2] = pds_mag * 255 #cv2.normalize(pds_mag, NONE_MAT, 0, 255, cv2.NORM_MINMAX)
        bgr_pda = cv2.cvtColor(hsv_pda, cv2.COLOR_HSV2BGR)

        # optical flow mag and ang as hsv:
        ofog_hsv = np.zeros_like(bgr_frame)
        # ofpr_hsv = np.zeros_like(bgr_frame)
        if optical_flow is not None:
            mag, ang = cv2.cartToPolar(optical_flow[0][..., 0], optical_flow[0][..., 1])
            ofog_hsv[..., 0] = ang * 180 / np.pi / 2
            ofog_hsv[..., 1] = 255
            ofog_hsv[..., 2] = mag * 255 #cv2.normalize(mag, NONE_MAT, 0, 255, cv2.NORM_MINMAX)
            # mag, ang = cv2.cartToPolar(optical_flow[1][..., 0], optical_flow[1][..., 1])
            # ofpr_hsv[..., 0] = ang * 180 / np.pi / 2
            # ofpr_hsv[..., 1] = 255
            # ofpr_hsv[..., 2] = cv2.normalize(mag, NONE_MAT, 0, 255, cv2.NORM_MINMAX)

        bgr_ofog = cv2.cvtColor(ofog_hsv, cv2.COLOR_HSV2BGR)
        # bgr_ofpr = cv2.cvtColor(ofpr_hsv, cv2.COLOR_HSV2BGR)
        render_extract_point(bgr_ofog, extract_point, extract_radius)
        # render_extract_point(bgr_ofpr, extract_point, extract_radius)

        vid_perc = vid_idx / (num_frames - 1)
        bgr_pdasparkline = draw_sparkline(signal_at_coords, vid_perc, (og_w, og_h), 0, 1)
        bgr_hapsparkline = draw_sparkline(hap_signal, vid_perc, (og_w, og_h), -1, 1, playback_head_line=True)

        bgr_pda = cv2.resize(bgr_pda, (og_w, og_h))

        render_extract_point(bgr_pda, extract_point, extract_radius)

        # resize to original shape
        bgr_frame = cv2.resize(bgr_frame, (og_w, og_h))
        bgr_processed = cv2.resize(bgr_processed, (og_w, og_h))

        render_extract_point(bgr_frame, extract_point, extract_radius)
        render_extract_point(bgr_processed, extract_point, extract_radius)

        top_vids = [bgr_frame, bgr_processed, bgr_pda]
        hsep = np.zeros((2, og_w * len(top_vids) + vsep.shape[1] * (len(top_vids) - 1), 3)).astype(np.uint8)
        bottom_row = [bgr_ofog, bgr_pdasparkline, bgr_hapsparkline]

        def insert_vseps(arr):
            out_row = []
            for i, img in enumerate(arr):
                out_row.append(img)
                if i < len(arr) - 1:
                    out_row.append(vsep)
            return out_row

        top_row = insert_vseps(top_vids)
        bottom_row = insert_vseps(bottom_row)

        # stack frames
        stacked = np.vstack((
            np.hstack(top_row),
            hsep,
            np.hstack(bottom_row),
        ))

        frame = stacked
        if out is None:
            print(f"Stacked Frame Shape: {frame.shape} \n")
            sh, sw, _ = frame.shape
            # out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*"VP80"), int(np.round(video_fs)), (sw, sh))
            # out = AsyncVideoWriterCV2(video_save_path, "VP80", int(np.round(video_fs)), (sw, sh), max_queue_size=0)
            out = AsyncFFmpegVideoWriter(video_save_path, video_codec, int(np.round(video_fs)), (sw, sh), max_queue_size=0)

        out.write(frame)

    if out: out.stop()
