"""Single-video phase-based motion magnification demo and export utility."""

import os
import sys
import re
import argparse
import time
import numpy as np
import cv2
import scipy

from render_utils import get_time_elapsed
from haptic import create_hap_signal, get_average_in_radius
from motion_mag_api import PhaseAlgorithm, create_tensors, create_transfer_function, generate_output_video, get_phase_deltas_and_mag_video, load_video
from phase_utils import *

## ==========================================================================================
## constants
EPS = 1e-6  # factor to avoid division by 0


## ==========================================================================================
## construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# Basic Args
ap.add_argument("-v", "--video_path", type=str, required=True, help="path to input video")
ap.add_argument("-a", "--phase_mag", type=float, default=25.0, required=True, help="Phase Magnification Factor")
ap.add_argument("-lo", "--freq_lo", type=float, required=True, help="Low Frequency cutoff for Temporal Filter")
ap.add_argument("-hi", "--freq_hi", type=float, required=True, help="High Frequency cutoff for Temporal Filter")
ap.add_argument(
    "-n",
    "--colorspace",
    type=str,
    default="luma1",
    choices={"luma1", "luma3", "gray", "yiq", "rgb"},
    help="Defines the Colorspace that the processing will take place in",
)

# Pyramid Args
ap.add_argument(
    "-p",
    "--pyramid_type",
    type=str,
    default="half_octave",
    choices={"full_octave", "half_octave", "smooth_half_octave", "smooth_quarter_octave"},
    help="Complex Steerable Pyramid Type",
)

# Phase Processing Args
ap.add_argument(
    "-s",
    "--sigma",
    type=float,
    default=0.0,
    help="Guassian Kernel Std Dev for amplitude weighted filtering, \n" "If 0, then amplitude weighted filtering will not be performed",
)
ap.add_argument("-t", "--attenuate", type=bool, default=False, help="Attenuates other frequencies if True")
ap.add_argument(
    "-fs",
    "--sample_frequency",
    type=float,
    default=-1.0,
    help="Video sample frequency, defaults to sample frequency from input "
    "video if input is less than or equal to zero. Video is "
    "reconstructed with detected sample frequency",
)

# Misc Args
ap.add_argument(
    "-r",
    "--reference_index",
    type=int,
    default=0,
    help="Reference Index for DC frame \
         (i.e. reference frame for phase changes)",
)
ap.add_argument("-c", "--scale_factor", type=float, default=1.0, help="Scales down image to rpeserve memory")
ap.add_argument("-b", "--batch_size", type=int, default=2, help="Batch size for CUDA parallelization")
ap.add_argument(
    "-d",
    "--save_directory",
    type=str,
    default="",
    help="Save directory for output video or GIF, if False outputs \
          are placed in the same location as the input video",
)

ap.add_argument("-ex", "--extractx", type=int, required=True, help="X coordinate to extract phase delta")
ap.add_argument("-ey", "--extracty", type=int, required=True, help="Y coordinate to extract phase delta")
ap.add_argument("-er", "--extract_radius", type=int, required=True, help="Radius of extraction area")
ap.add_argument("-m", "--method", type=str, default="linear", choices=["linear", "accel"], help="Phase Processing Method: linear or acceleration based")
# Add benchmark argument
ap.add_argument("-bm", "--benchmark", action="store_true", help="Benchmark the time taken by get_phase_deltas_and_mag_video")

## ==========================================================================================
## start main program

def main():
    ## Default use commandline args
    ## --> Comment this out to manually input args in script
    args = vars(ap.parse_args())

    # Optional: Pass arguments directly in script
    # --> Comment this out to receive args from commandline
    # args = vars(ap.parse_args(
    #     ["--video_path",       "videos/guitar.avi", # "videos/eye.avi", # "videos/crane_crop.avi",
    #      "--phase_mag",        "25.0", # "25.0",
    #      "--freq_lo",          "72", # "30", # "0.20",
    #      "--freq_hi",          "92", # "50", # "0.25",
    #      "--colorspace",       "luma3",
    #      "--pyramid_type",     "half_octave",
    #      "--sigma",            "2.0", # "5.0"
    #      "--attenuate",        "True", # "False",
    #      "--sample_frequency", "600", # "500", # "-1.0", # This is generally not needed
    #      "--reference_index",  "0",
    #      "--scale_factor",     "0.75", # "1.0"
    #      "--batch_size",       "4",
    #      "--save_directory",   "",
    #      "--save_gif",         "True"
    #      ]))

    # args = vars(ap.parse_args(
    #     ["--video_path",       "videos/crane_crop.avi",
    #      "--phase_mag",        "25.0",
    #      "--freq_lo",          "0.20",
    #      "--freq_hi",          "0.25",
    #      "--colorspace",       "luma3",
    #      "--pyramid_type",     "half_octave",
    #      "--sigma",            "5.0",
    #      "--attenuate",        "True", # "False",
    #      "--sample_frequency", "-1.0", # This is generally not needed
    #      "--reference_index",  "0",
    #      "--scale_factor",     "1.0",
    #      "--batch_size",       "4",
    #      "--save_directory",   "",
    #      "--save_gif",         "False"
    #      ]))

    ## Parse Args
    video_path = args["video_path"]
    phase_mag = args["phase_mag"]
    freq_lo = args["freq_lo"]
    freq_hi = args["freq_hi"]
    colorspace = args["colorspace"]
    pyramid_type = args["pyramid_type"]
    sigma = args["sigma"]
    attenuate = args["attenuate"]
    sample_frequency = args["sample_frequency"]
    ref_idx = args["reference_index"]
    scale_factor = args["scale_factor"]
    batch_size = args["batch_size"]
    save_directory = args["save_directory"]

    extract_point = (args["extractx"], args["extracty"])
    extract_radius = args["extract_radius"]
    video_processing_method = args["method"].lower()
    subtract_global_avg = False

    ## ======================================================================================
    ## start the clock once the args are received
    tic = cv2.getTickCount()

    ## ======================================================================================
    ## Process input filepaths
    if not os.path.exists(video_path):
        print(f"\nInput video path: {video_path} not found! exiting \n")
        sys.exit()

    if not save_directory:
        save_directory = os.path.dirname(video_path)
    elif not os.path.exists(save_directory):
        save_directory = os.path.dirname(video_path)
        print(f"\nSave Directory not found, " "using default input video directory instead \n")

    video_name = re.search("[\w -]*(?=\.\w*)", video_path).group() # type: ignore
    video_output = f"{video_name}_{video_processing_method}_{sigma}_{colorspace}_{freq_lo}to{freq_hi}hz_{attenuate}_{int(phase_mag)}x_OFOG.mp4"
    video_codec = "h264_nvenc"
    video_save_path = os.path.join(save_directory, video_output)

    print(f"\nProcessing {video_name} " f"and saving results to {video_save_path} \n")


    frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func, inv_colorspace_rgb = load_video(colorspace, video_path, scale_factor)
    toc_loaded_video = cv2.getTickCount()

    filters_tensor, filter_dir_tensor, frames_tensor, csp = create_tensors(frames, ref_idx, pyramid_type, batch_size)
    toc_filters_and_tensors = cv2.getTickCount()

    transfer_function = create_transfer_function(sample_frequency, video_fs, num_frames, freq_lo, freq_hi)


    phase_algo = PhaseAlgorithm.LINEAR if video_processing_method == "linear" else PhaseAlgorithm.ACCEL
    print(f"Using Phase Algorithm: {phase_algo.name} \n")

    benchmark_start = time.time()
    result_video, phase_delta_dir_sum = get_phase_deltas_and_mag_video(sigma, video_fs, transfer_function, phase_mag, attenuate, ref_idx, batch_size, colorspace, frames_tensor, filters_tensor, filter_dir_tensor, csp, EPS,
                                                                       ontick=lambda x: (print(f"Processing video... {x:.2%}")) and None, phase_algo=phase_algo)
    if args["benchmark"]:
        benchmark_time = time.time() - benchmark_start
        print(f"Time taken by get_phase_deltas_and_mag_video: {benchmark_time}")
        sys.exit()

    toc_finished_processing = cv2.getTickCount()

    pds_mag_all, pds_ang_all = cv2.cartToPolar(phase_delta_dir_sum[..., 0], phase_delta_dir_sum[..., 1])
    mag_norm_all = (pds_mag_all - np.min(pds_mag_all)) / (np.max(pds_mag_all) - np.min(pds_mag_all)) # this normalization isnt great because the extraction is sampled over an area and these averages are much lower than the single pixel peak. Not sure what else would work especially if we allow custom shapes for the extraction region.
    global_mag_avg = np.mean(mag_norm_all, axis=(1, 2)) # frame-wise avg magnitude
    mag_norm_all_sub = np.clip(mag_norm_all - global_mag_avg[:, None, None], 0, None) # subtract global avg


    signal_at_coords = get_average_in_radius(mag_norm_all_sub if subtract_global_avg else mag_norm_all, extract_point, extract_radius)
    hap_signal, signal_resampled_norm, accel, accel_resampled = create_hap_signal(signal_at_coords, video_fs)
    haptic_save_path = re.sub(r"\.[^.]*$", f"-{extract_point[0]}_{extract_point[1]}-{extract_radius}-pbm.wav", video_save_path)
    scipy.io.wavfile.write(haptic_save_path, 8000, hap_signal)

    # todo handle v_w, v_h and og_w, og_h
    generate_output_video(frames, result_video, v_w, v_h, num_frames, inv_colorspace,
                                  mag_norm_all_sub if subtract_global_avg else mag_norm_all, pds_ang_all,
                                  extract_point, extract_radius, signal_at_coords, hap_signal, video_save_path, video_codec, video_fs)
    toc_finished_generating_frames = cv2.getTickCount()
    print(f"Result video saved to: {video_save_path} \n")

    toc_final = cv2.getTickCount()

    print("Motion Magnification processing complete! \n")
    print(f"Video Loaded at: {get_time_elapsed(tic, toc_loaded_video)}")
    print(f"Filters and Video Tensors Created at: {get_time_elapsed(tic, toc_filters_and_tensors)}")
    print(f"Processing Finished at: {get_time_elapsed(tic, toc_finished_processing)}")
    print(f"All Frames Generated at: {get_time_elapsed(tic, toc_finished_generating_frames)}")
    print(f"Output finished at (HH:MM:SS): {get_time_elapsed(tic, toc_final)} \n")



if __name__ == "__main__":
    main()
