"""Batch generate haptic waveforms from a directory of videos."""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Iterable, List, Sequence, Tuple

import cupy as cp
import cv2
import numpy as np
import scipy
from scipy.io import wavfile

from haptic import HAPTIC_SAMPLE_RATE, compute_integral_cupy, create_hap_signal, sample_box_mean_cupy
from motion_mag_api import PhaseAlgorithm, create_tensors, create_transfer_function, get_phase_deltas_and_mag_video, load_video
from saliency import SpatiotemporalSaliencyGPU
from phase_utils import bgr2yiq
from video_reader import DurationExceededError


logger = logging.getLogger(__name__)

DEFAULT_FREQ_LO = 1.0
DEFAULT_FREQ_HI = 11.0


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    max_size: Tuple[int, int]
    extraction_percentages: Sequence[float]
    algorithm: str
    preload_workers: int
    load_queue_size: int
    load_workers: int
    process_queue_size: int
    process_workers: int
    max_video_duration: float | None
    freq_hi_nyquist: bool


class GracefulExit(Exception):
    """Raised when a stop signal is received so workers can wind down."""


_stop_requested = False


def _handle_stop_signal(signum: int, _frame) -> None:  # type: ignore[override]
    global _stop_requested
    _stop_requested = True
    logger.warning("Received signal %s; stopping after current tasks finish.", signum)


# ----------------------------- Processing helpers -----------------------------


def bench_phase(
    frames: List[np.ndarray],
    video_fs: float,
    num_frames: int,
    batch_size: int,
    use_alt: str,
    freq_hi_nyquist: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Copy of the phase benchmarking routine used for production processing."""

    video_processing_method = "accel"
    pyramid_type: str = "half_octave"
    colorspace: str = "luma1"
    sample_frequency = -1  # -1 means use the video frame rate

    phase_mag: float = 8.0
    sigma: float = 0.0
    attenuate: bool = False
    ref_idx: int = 0
    freq_lo, freq_hi = resolve_phase_cutoffs(video_fs, freq_hi_nyquist)

    EPS = 1e-6

    # Don't manually convert colorspace here - let get_phase_deltas_and_mag_video handle it
    filters_tensor, filter_dir_tensor, frames_tensor, csp = create_tensors(frames, ref_idx, pyramid_type, batch_size)
    transfer_function = create_transfer_function(sample_frequency, video_fs, num_frames, freq_lo, freq_hi)

    phase_algo = PhaseAlgorithm.LINEAR if video_processing_method == "linear" else PhaseAlgorithm.ACCEL

    result_video, phase_delta_dir_sum = get_phase_deltas_and_mag_video(
        sigma,
        video_fs,
        transfer_function,
        phase_mag,
        attenuate,
        ref_idx,
        batch_size,
        colorspace,
        frames_tensor,
        filters_tensor,
        filter_dir_tensor,
        csp,
        EPS,
        phase_algo=phase_algo,
    )

    return 0.0, result_video, phase_delta_dir_sum


def bench_saliency(frames: List[np.ndarray], video_fs: float, num_frames: int) -> tuple[float, np.ndarray]:
    ss = SpatiotemporalSaliencyGPU()
    output = ss.compute_spatiotemporal_saliency(frames)
    if output is None:
        raise RuntimeError("Saliency computation returned None")
    return 0.0, output


def resolve_phase_cutoffs(video_fs: float, freq_hi_nyquist: bool) -> tuple[float, float]:
    freq_lo = DEFAULT_FREQ_LO
    if not freq_hi_nyquist:
        return freq_lo, DEFAULT_FREQ_HI

    # scipy.signal.firwin requires cutoff frequencies to stay strictly below Nyquist.
    freq_hi = float(np.nextafter(video_fs / 2.0, 0.0))
    if freq_hi <= freq_lo:
        raise ValueError(
            f"Cannot set freq_hi near Nyquist for video_fs={video_fs:.4f} Hz; "
            f"resolved freq_hi={freq_hi:.4f} must be greater than freq_lo={freq_lo:.4f}."
        )
    return freq_lo, freq_hi


def get_phase_mag_norm(
    frames: List[np.ndarray],
    video_fs: float,
    num_frames: int,
    batch_size: int = 4,
    use_alt: str = "",
    freq_hi_nyquist: bool = False,
) -> np.ndarray:
    _dur, _video, phase_delta_dir_sum = bench_phase(
        frames,
        video_fs,
        num_frames,
        batch_size=batch_size,
        use_alt=use_alt,
        freq_hi_nyquist=freq_hi_nyquist,
    )

    pd_r = phase_delta_dir_sum[..., 0]
    pd_i = phase_delta_dir_sum[..., 1]
    pds_mag_all, pds_ang_all = cv2.cartToPolar(pd_r, pd_i)
    mag_norm_all = (pds_mag_all - np.min(pds_mag_all)) / (np.max(pds_mag_all) - np.min(pds_mag_all) + 1e-8)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_mat = np.nan_to_num(mag_norm_all / pds_mag_all)

    pd_r_norm = pd_r * scale_mat
    pd_i_norm = pd_i * scale_mat

    _avg_r_norm = np.mean(pd_r_norm, axis=(1, 2))
    _avg_i_norm = np.mean(pd_i_norm, axis=(1, 2))
    _pds_ang_all_sub = pds_ang_all
    global_mag_avg = np.mean(mag_norm_all, axis=(1, 2))
    mag_norm_all_sub = np.clip(mag_norm_all - global_mag_avg[:, None, None], 0, None)

    return mag_norm_all_sub


def get_saliency_map(frames: List[np.ndarray], video_fs: float, num_frames: int) -> np.ndarray:
    _dur, saliency_map = bench_saliency(frames, video_fs, num_frames)
    return saliency_map


def get_md_S_phase(frames: List[np.ndarray], video_fs: float, num_frames: int, freq_hi_nyquist: bool = False) -> cp.ndarray:
    mag_norm_all_sub = get_phase_mag_norm(
        frames,
        video_fs,
        num_frames,
        batch_size=4,
        use_alt="",
        freq_hi_nyquist=freq_hi_nyquist,
    )
    return compute_integral_cupy(cp.array(mag_norm_all_sub))


def get_md_S_saliency(frames: List[np.ndarray], video_fs: float, num_frames: int) -> cp.ndarray:
    saliency_map = get_saliency_map(frames, video_fs, num_frames)
    return compute_integral_cupy(cp.array(saliency_map))

# ----------------------------- IO helpers -----------------------------


def discover_videos(input_dir: Path, extensions: Iterable[str] | None = None) -> list[Path]:
    if extensions is None:
        extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext_set = {ext.lower() for ext in extensions}
    return [p for p in input_dir.rglob("*") if p.suffix.lower() in ext_set and p.is_file()]


def _format_opts(max_size: Tuple[int, int], percent: float, algorithm: str) -> str:
    mw, mh = max_size
    return f"w{mw}h{mh}_pct{percent:g}_alg{algorithm}"


def _resolve_max_size(width: int | None, height: int | None) -> Tuple[int, int]:
    width_val = width or 0
    height_val = height or 0
    if width_val <= 0:
        width_val = 10**6
    if height_val <= 0:
        height_val = 10**6
    return width_val, height_val



def get_pending_outputs(video_path: Path, cfg: BatchConfig) -> list[tuple[float, Path]]:
    """Return pending (percent, target_path) outputs for this video. Does not perform file I/O."""
    relative_path = video_path.relative_to(cfg.input_dir)
    pending_outputs: list[tuple[float, Path]] = []
    for percent in cfg.extraction_percentages:
        opts = _format_opts(cfg.max_size, percent, cfg.algorithm)
        output_name = f"{video_path.stem}_{opts}.wav"
        target_path = cfg.output_dir / relative_path.parent / output_name
        if target_path.exists():
            continue
        pending_outputs.append((percent, target_path))
    return pending_outputs


def check_and_enqueue_for_loading(video_path: Path, cfg: BatchConfig, load_queue: Queue) -> None:
    """Check if video needs processing and enqueue for loading (I/O bound stage).

    This function blocks on queue.put when the buffer is full (backpressure).
    """
    if _stop_requested:
        return

    pending_outputs = get_pending_outputs(video_path, cfg)
    if not pending_outputs:
        logger.info("Skipping %s (all outputs already exist)", video_path)
        return

    # ensure output dir exists for this video
    relative_path = video_path.relative_to(cfg.input_dir)
    output_dir = cfg.output_dir / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enqueue for loading
    try:
        load_queue.put((video_path, pending_outputs), block=True)
        logger.debug("Enqueued %s for loading (load queue size now ~ %d)", video_path, load_queue.qsize())
    except Exception:
        logger.exception("Failed to enqueue %s for loading", video_path)


def load_and_enqueue_for_processing(load_queue: Queue, process_queue: Queue, cfg: BatchConfig) -> None:
    """Load video frames and enqueue for processing (CPU/memory bound stage)."""
    while True:
        try:
            item = load_queue.get()
        except Exception:
            logger.exception("Load queue get failed")
            break

        # sentinel -> exit
        if item is None:
            load_queue.task_done()
            break

        video_path, pending_outputs = item

        try:
            frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func, inv_colorspace_rgb = load_video(
                colorspace="luma1",
                video_path=str(video_path),
                scale_factor=1.0,
                load_size=cfg.max_size,
                max_duration=cfg.max_video_duration,
            )

            process_item = (video_path, pending_outputs, frames, video_fs, (v_w, v_h), num_frames)

            # Block if process queue is full to bound memory
            try:
                process_queue.put(process_item, block=True)
                logger.debug("Enqueued %s for processing (process queue size now ~ %d)", video_path, process_queue.qsize())
            except Exception:
                logger.exception("Failed to enqueue %s for processing", video_path)

        except DurationExceededError as e:
            logger.info("Skipping %s: %s", video_path, e)
        except GracefulExit:
            logger.warning("Graceful exit during video load for %s", video_path)
            load_queue.task_done()
            # put a sentinel back so other load workers can exit
            load_queue.put(None)
            break
        except Exception:
            logger.exception("Failed to load %s", video_path)
        finally:
            load_queue.task_done()


def process_loaded_item(item: tuple, cfg: BatchConfig) -> list[Path]:
    """Process a preloaded video item. Returns list of completed target paths."""
    (video_path, pending_outputs, frames, video_fs, (v_w, v_h), num_frames) = item

    if cfg.algorithm == "phase":
        md_S = get_md_S_phase(frames, video_fs, num_frames, freq_hi_nyquist=cfg.freq_hi_nyquist)
    elif cfg.algorithm == "saliency":
        md_S = get_md_S_saliency(frames, video_fs, num_frames)
    else:
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm}")

    center = (v_w // 2, v_h // 2)
    min_dim_half = min(v_w, v_h) / 2.0

    completed: list[Path] = []
    for percent, target_path in pending_outputs:
        if _stop_requested:
            raise GracefulExit()

        radius = max(1, int(min_dim_half * percent))
        signal_at_coords = sample_box_mean_cupy(md_S, center, radius)
        hap_signal, *_ = create_hap_signal(signal_at_coords.get(), video_fs)

        temp_path = target_path.with_suffix(target_path.suffix + ".temp")
        scipy.io.wavfile.write(temp_path, HAPTIC_SAMPLE_RATE, hap_signal)
        os.replace(temp_path, target_path)
        completed.append(target_path)
        logger.info("Wrote %s", target_path)

    return completed


def process_worker(queue: Queue, cfg: BatchConfig) -> None:
    """Worker loop that consumes preloaded items from the queue until a sentinel is received."""
    while True:
        try:
            item = queue.get()
        except Exception:
            logger.exception("Queue get failed")
            break

        # sentinel -> exit without putting it back; main will enqueue one sentinel per worker
        if item is None:
            queue.task_done()
            break

        try:
            completed = process_loaded_item(item, cfg)
            if completed:
                logger.debug("Completed %s", item[0])
        except GracefulExit:
            logger.warning("Graceful exit requested during processing; stopping worker")
            queue.task_done()
            # put a sentinel back so other workers can exit
            queue.put(None)
            break
        except Exception:
            logger.exception("Failed to process %s", item[0])
        finally:
            queue.task_done()


# ----------------------------- CLI and orchestration -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch haptic extraction from video datasets")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing input videos")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory for output wav files")
    parser.add_argument("--max-width", type=int, default=640, help="Maximum width for loaded videos (set <=0 to disable)")
    parser.add_argument("--max-height", type=int, default=480, help="Maximum height for loaded videos (set <=0 to disable)")
    parser.add_argument(
        "--extraction-percentages",
        type=float,
        nargs="+",
        default=[1.0],
        help="One or more region size percentages (relative to half the smaller frame dimension)",
    )
    parser.add_argument("--algorithm", choices=["phase", "saliency"], default="phase", help="Algorithm to use for extraction")
    parser.add_argument("--preload-workers", type=int, default=3, help="Number of preload worker threads (I/O bound)")
    parser.add_argument("--load-queue-size", type=int, help="Max videos queued for loading (defaults to load-workers * 2)")
    parser.add_argument("--load-workers", type=int, default=2, help="Number of video loading worker threads (CPU/memory bound)")
    parser.add_argument("--process-queue-size", type=int, help="Max loaded videos in memory waiting for processing (defaults to process-workers)")
    parser.add_argument("--process-workers", type=int, default=2, help="Number of GPU processing worker threads")
    parser.add_argument("--max-video-duration", type=float, default=None, help="Maximum duration in seconds to load from each video (default: None, load entire video)")
    parser.add_argument(
        "--freq-hi-nyquist",
        action="store_true",
        help="Set the phase high cutoff to just below fs/2 instead of the default 11 Hz",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGUSR1"):
        signals.append(signal.SIGUSR1) # type: ignore
    for sig in signals:
        signal.signal(sig, _handle_stop_signal)

    max_size = _resolve_max_size(args.max_width, args.max_height)
    cfg = BatchConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        max_size=max_size,
        extraction_percentages=tuple(args.extraction_percentages),
        algorithm=args.algorithm,
        process_workers=max(1, args.process_workers),
        load_workers=max(1, args.load_workers),
        load_queue_size=max(1, args.load_queue_size or (args.load_workers * 2)),
        preload_workers=max(1, args.preload_workers),
        process_queue_size=max(1, args.process_queue_size or args.process_workers),
        max_video_duration=args.max_video_duration,
        freq_hi_nyquist=args.freq_hi_nyquist,
    )

    videos = discover_videos(cfg.input_dir)
    if not videos:
        logger.warning("No videos found in %s", cfg.input_dir)
        return

    logger.info("Found %d videos to consider", len(videos))

    # 3-stage bounded pipeline:
    # - preload_executor (I/O): validates outputs and enqueues to load_queue
    # - load_executor (CPU/memory): decodes videos and enqueues to process_queue
    # - process_executor (GPU): processes frames and saves outputs
    load_queue = Queue(maxsize=cfg.load_queue_size)
    process_queue = Queue(maxsize=cfg.process_queue_size)

    preload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.preload_workers)
    load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.load_workers)
    process_executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.process_workers)

    # Start load workers (middle stage)
    load_futures = [load_executor.submit(load_and_enqueue_for_processing, load_queue, process_queue, cfg) for _ in range(cfg.load_workers)]

    # Start process workers (final stage)
    process_futures = [process_executor.submit(process_worker, process_queue, cfg) for _ in range(cfg.process_workers)]

    # Submit preload tasks (first stage)
    preload_futures = {preload_executor.submit(check_and_enqueue_for_loading, video, cfg, load_queue): video for video in videos}

    try:
        # Monitor preload tasks so we can catch errors early and support graceful exit
        for future in concurrent.futures.as_completed(preload_futures):
            video = preload_futures[future]
            try:
                future.result()
                logger.debug("Preloaded %s", video)
            except GracefulExit:
                logger.warning("Graceful exit requested during preload; cancelling remaining preload tasks")
                for f in preload_futures:
                    f.cancel()
                break
            except Exception:
                logger.exception("Failed to preload %s", video)

        # Once checking is done, signal load workers to exit
        for _ in range(cfg.load_workers):
            try:
                load_queue.put(None)
            except Exception:
                logger.exception("Failed to enqueue sentinel for load worker shutdown")

        # Wait for all load tasks to complete
        load_queue.join()
        concurrent.futures.wait(load_futures)

        # Signal process workers to exit
        for _ in range(cfg.process_workers):
            try:
                process_queue.put(None)
            except Exception:
                logger.exception("Failed to enqueue sentinel for process worker shutdown")

        # Wait for all process tasks to complete
        process_queue.join()
        concurrent.futures.wait(process_futures)

        logger.info("All processing complete")

    finally:
        logger.info("Shutting down executors...")
        preload_executor.shutdown(wait=True, cancel_futures=True)
        load_executor.shutdown(wait=True, cancel_futures=True)
        process_executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Executors shut down")


if __name__ == "__main__":
    main()
    logger.info("Exiting cleanly")
    sys.exit(0)
