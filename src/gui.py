import ctypes
import gc
from pathlib import Path
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import scipy
from vedo import Plotter, Volume, Plane, Image, vtkclasses
import vedo
import vedo.shapes
from vedo.transformations import LinearTransform
from vedo.utils import vtk2numpy

from motion_mag_api import NONE_MAT, PhaseAlgorithm, create_tensors, create_transfer_function, generate_output_video, get_phase_deltas_and_mag_video, load_video
from render_utils import draw_sparkline, get_time_elapsed, render_extract_point, render_extract_point_rgb
from haptic import HAPTIC_SAMPLE_RATE, create_hap_signal, extract_hap_signal, get_average_in_radius, get_rms_in_radius
from gui_utils import Debouncer, angmag_to_rgba_vectorized, bgr_to_float_rgb, bgr_to_float_rgb_flat, cubic_bezier_points, generate_box_wireframe, update_from_gui
from saliency import SpatiotemporalSaliencyGPU
from enum import Enum

from quest_websocket import QuestWebSocketClient
from threading import Timer

from event_logger import EventLogger

COLOR_TEXT_LOWPRI = (175, 175, 175)

## constants
EPS = 1e-6  # factor to avoid division by 0

USER_STUDY_SAVE_DIR = Path("userstudy")
USER_STUDY_VIDEOS_AND_ALGORITHMS: List[Tuple[str, int] | str] = [
    ("videos/userstudy/training.mp4", 0),

    ("videos/userstudy/gun.mp4", 0),
    ("videos/userstudy/ace_combat.mp4", 0),
    ("videos/userstudy/archery.mp4", 0),
    ("videos/userstudy/drone.mp4", 0),
    ("videos/userstudy/f1.mp4", 0),
    "prompt survey",

    ("videos/userstudy/gun.mp4", 1),
    ("videos/userstudy/ace_combat.mp4", 1),
    ("videos/userstudy/archery.mp4", 1),
    ("videos/userstudy/drone.mp4", 1),
    ("videos/userstudy/f1.mp4", 1),
    "prompt survey",
    "prompt done"
]

@dataclass
class TrainingStep:
    text: str
    outlined_widgets: List[str]
    check_done: Callable[[], bool]

training_idx = 0
in_training = False
training_prev_extract_point: Optional[Tuple[int, int]] = None
training_prev_extract_radius: Optional[int] = None

training_steps: List[TrainingStep] = [
    TrainingStep(
        "Welcome to the Video To Haptics GUI!\nPlease make sure you have completed the background questionnaire on Qualtrics before proceeding.",
        outlined_widgets=[],
        check_done=lambda: True,
    ),
    TrainingStep(
        "You can click and drag to select a extraction location in the video. This is the location where the haptic signal will be generated from.",
        outlined_widgets=["frame_img"],
        check_done=lambda: training_prev_extract_point is not None and extract_point != training_prev_extract_point,
    ),
    TrainingStep(
        "You can use the scroll wheel to adjust the size of the extraction region.",
        outlined_widgets=["frame_img"],
        check_done=lambda: training_prev_extract_radius is not None and extract_radius != training_prev_extract_radius,
    ),
    TrainingStep(
        "Press the spacebar to loop the video playback.",
        outlined_widgets=["frame_img"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "The haptic signal generated from the current selected region will be visualized here.",
        outlined_widgets=["hapspark_img"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "Press the inside trigger on the controller to play the haptic signal.",
        outlined_widgets=["hapspark_img"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "You can use this slider to manually scrub through the video.",
        outlined_widgets=["vid_idx"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "(Advanced) To help with creating a signal, you can view the motion estimation here.\nThis pane can also be used to adjust the haptic extraction region (click+drag).",
        outlined_widgets=["pda_img"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "(Advanced) This 3D visualization shows the motion estimation over time.\nClick and drag to rotate the view. Scroll to zoom in/out.",
        outlined_widgets=["vedo_img"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "(Advanced) The un-amplified motion information used to generate the haptic signal is plotted here.",
        outlined_widgets=["extract_plot"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "To move to the next step, click the 'Submit Signal' button on the right of the top banner.",
        outlined_widgets=["submit_signal"],
        check_done=lambda: True,
    ),
    TrainingStep(
        "Thank you for completing the training steps!\n" \
        "For the rest of the design tasks please try to create the best signal possible for the video.\n" \
        "Click 'Next' to continue.",
        outlined_widgets=[],
        check_done=lambda: True,
    ),
]

def show_step_overlay(widget_tags):
    clear_overlay()

    overlay_color = (0, 0, 0, 150)
    border_color  = (50, 150, 255, 255)
    padding = 5
    win_size = tuple(map(float, dpg.get_item_rect_size(primary_window)))

    dpg.push_container_stack("training_overlay")
    # dpg.draw_rectangle((0,0), win_size, color=overlay_color, fill=overlay_color)
    # for widget_tag in widget_tags:
    #     min_xy = tuple(map(float, dpg.get_item_rect_min(widget_tag)))
    #     max_xy = tuple(map(float, dpg.get_item_rect_max(widget_tag)))
    #     dpg.draw_rectangle(min_xy, max_xy, color=(0,0,0,0), fill=(0,0,0,0))
    #     dpg.draw_rectangle(tuple(map(lambda x: x-4, min_xy)), tuple(map(lambda x: x+4, max_xy)), color=(50, 150, 255, 255), thickness=2)

    win_w, win_h = win_size
    if len(widget_tags) == 0:
        dpg.draw_rectangle((0, 0), (win_w, win_h), color=(0,0,0,0), fill=overlay_color)
    else:
        for tag in widget_tags:
            min_x, min_y = map(float, dpg.get_item_rect_min(tag))
            max_x, max_y = map(float, dpg.get_item_rect_max(tag))

            dpg.draw_rectangle((0, 0), (win_w, min_y), color=(0,0,0,0), fill=overlay_color)
            dpg.draw_rectangle((0, max_y), (win_w, win_h), color=(0,0,0,0), fill=overlay_color)
            dpg.draw_rectangle((0, min_y), (min_x, max_y), color=(0,0,0,0), fill=overlay_color)
            dpg.draw_rectangle((max_x, min_y), (win_w, max_y), color=(0,0,0,0), fill=overlay_color)

            dpg.draw_rectangle(
                (min_x - padding, min_y - padding),
                (max_x + padding, max_y + padding),
                color=border_color,
                thickness=3
            )

    dpg.pop_container_stack()
def clear_overlay():
    dpg.delete_item("training_overlay", children_only=True)

def on_training_next(back=False):
    global training_idx

    if back:
        training_idx = max(0, training_idx - 1)
    else:
        if training_idx >= len(training_steps):
            return
        step = training_steps[training_idx]
        if not step.check_done() and not dpg.is_key_down(dpg.mvKey_LControl):
            print(f"Training step {training_idx} not completed yet, cannot proceed.")
            dpg.set_value("training_status", "")
            dpg.set_frame_callback(dpg.get_frame_count() + 5, lambda: dpg.set_value("training_status", "Please try this action before continuing."))
            return
        training_idx += 1
    render_training_step()

def render_training_step():
    global training_idx, in_training, training_prev_extract_point, training_prev_extract_radius
    if dpg.does_item_exist("training_modal"):
        dpg.delete_item("training_modal")
    dpg.show_item("training_overlay_window")
    if training_idx >= len(training_steps):
        in_training = False
        clear_overlay()
        dpg.enable_item("submit_signal")
        dpg.hide_item("training_overlay_window")
        dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: submit_signal_callback("training_done", None, None))
        return

    step = training_steps[training_idx]
    show_step_overlay(step.outlined_widgets)

    vph = dpg.get_viewport_height()
    vpw = dpg.get_viewport_width()
    tmw = 420
    tmh = 200

    with dpg.window(label="Training", modal=False, no_close=True, pos=(1730, 550), width=tmw, height=tmh, tag="training_modal"):
        dpg.add_text(step.text, wrap=tmw-13)
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Back", callback=lambda: on_training_next(back=True))
            dpg.add_button(label="Next", callback=lambda: on_training_next(back=False))
            dpg.add_text(f"(Step {training_idx + 1}/{len(training_steps)})", color=COLOR_TEXT_LOWPRI)
        dpg.add_text("", tag="training_status")

def start_training():
    global training_idx, in_training, training_prev_extract_point, training_prev_extract_radius
    if in_training:
        return
    dpg.disable_item("submit_signal")
    training_prev_extract_point = extract_point
    training_prev_extract_radius = extract_radius
    training_idx = 0
    in_training = True
    render_training_step()

scale_factor: float = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75 # scale factor for video
sample_frequency: int = -1 # override video sample rate
pyramid_type: str = "half_octave"
colorspace: str = "luma1"

batch_size: int = 2 # batch size for cuda (not adj yet)

video_path: str = sys.argv[1] if len(sys.argv) > 1 else "videos/userstudy/training.mp4"
process_params_changed: bool = False
phase_mag: float = 8.0
sigma: float = 0.0
attenuate: bool = False
ref_idx: int = 0
freq_lo: float = 1.0
freq_hi: float = 11.0

extract_point: Tuple[int, int] = (0, 0)
extract_radius: int = 1
subtract_global_avg: bool = True
rms_instead_of_mean: bool = False
use_ang_extraction: bool = False

motion_vol: Optional[Volume] = None

class VideoProcessingMethod(Enum):
    LINEAR_PHASE = "Linear Phase Based"
    SPATIOTEMPORAL_SALIENCY = "Spatiotemporal Saliency"
    ACCELERATION_PHASE = "Acceleration Phase"

# video_processing_method = VideoProcessingMethod.PHASE_BASED_MOTION_MAGNIFICATION.value
video_processing_method: str = VideoProcessingMethod.ACCELERATION_PHASE.value

user_study_video_idx = 0
user_study_mode = sys.argv[4] if len(sys.argv) > 4 else None
user_study_participant_id = sys.argv[5] if len(sys.argv) > 5 else None
user_study_ab_flip = sys.argv[6] if len(sys.argv) > 6 else None
def vpm_ab_to_str(vpm: int) -> str:
    if user_study_ab_flip == "normal":
        return VideoProcessingMethod.ACCELERATION_PHASE.value if vpm == 0 else VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value
    elif user_study_ab_flip == "flip":
        return VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value if vpm == 0 else VideoProcessingMethod.ACCELERATION_PHASE.value
    else:
        raise ValueError(f"Invalid user_study_ab_flip value: {user_study_ab_flip}")
if user_study_mode is not None:
    if user_study_mode != "--user_study_pid":
        raise ValueError(f"Invalid argument: {user_study_mode}. Expected '--user_study_pid' or None.")
    elif user_study_participant_id is None:
        raise ValueError("User Study Participant ID must be provided.")
    elif user_study_ab_flip is None or user_study_ab_flip not in ("normal", "flip"):
        raise ValueError("User Study AB Flip must be provided (e.g. 'normal' or 'flip').")
    else:
        user_study_participant_save_dir = USER_STUDY_SAVE_DIR / Path(user_study_participant_id)
        user_study_participant_save_dir.mkdir(parents=False, exist_ok=True)
        next_task = USER_STUDY_VIDEOS_AND_ALGORITHMS[user_study_video_idx]
        if isinstance(next_task, str):
            raise ValueError(f"Unexpected task in user study: {next_task}. Expected a video path and algorithm index.")
        video_path, vpm_i = next_task
        video_processing_method = vpm_ab_to_str(vpm_i)
else:
    user_study_participant_save_dir = None


algo_survey_active = False
def show_algorithm_survey(algo_name: str):
    global algo_survey_active
    logger.log("Show Algorithm Survey", f"algo={algo_name}")
    algo_survey_active = True
    with dpg.window(label="Algorithm Survey Questions", modal=True, no_move=True, no_close=True, no_collapse=True, width=980, height=300) as survey_window:
        radio_buttons = []
        for qt in ["This algorithm was easy to work with overall.", "The behavior of this algorithm was predictable.", "I would use this algorithm again for a similar task."]:
            with dpg.collapsing_header(label=qt, default_open=True, bullet=True, leaf=True):
                rb = dpg.add_radio_button(
                    items=["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", "Somewhat Agree", "Agree", "Strongly Agree"],
                    default_value="Neutral",
                    horizontal=True
                )
                radio_buttons.append((qt, rb))
                dpg.bind_item_font(rb, smaller_font)
                dpg.add_spacer(height=10)

        dpg.add_spacer(height=10)

        def on_submit():
            global algo_survey_active
            responses = {qt: dpg.get_value(rb) for qt, rb in radio_buttons}
            print(f"Survey Responses for {algo_name}:", responses)
            logger.log("Algorithm Survey Responses", f"algo={algo_name}, responses={responses}")
            dpg.delete_item(survey_window)
            algo_survey_active = False
            dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: submit_signal_callback("show_algorithm_survey", None, None))

        dpg.add_button(label="Submit Responses", callback=on_submit)

last_submit_at: float = 0.0
def submit_signal_callback(sender, app_data, user_data):
    global user_study_video_idx, video_path, video_processing_method, process_params_changed, last_submit_at

    if sender != "show_algorithm_survey" and sender != "training_done":
        dpg.disable_item(sender)
        dpg.set_frame_callback(dpg.get_frame_count() + 60, lambda: dpg.enable_item(sender))
        if time.time() < last_submit_at + 20:
            print("[WARN] Submit button pressed too quickly, ignoring.")
            logger.log("WARN Submit Signal Ignored", f"video_path={video_path}")
            return
        last_submit_at = time.time()

        logger.log("Submit Signal", f"video_path={video_path}, video_processing_method={video_processing_method}, extract_point={extract_point}, extract_radius={extract_radius}, subtract_global_avg={subtract_global_avg}, rms_instead_of_mean={rms_instead_of_mean}, use_ang_extraction={use_ang_extraction}")
        next_vid, next_algo_int = USER_STUDY_VIDEOS_AND_ALGORITHMS[user_study_video_idx]
        assert next_algo_int in (0, 1), f"Unexpected algorithm index {next_algo_int} for user study video {next_vid}"
        if not user_study_participant_save_dir:
            print("No user study participant ID set, not saving haptic signal.")
        else:
            current_name = Path(video_path).stem
            out_file     = user_study_participant_save_dir / f"{current_name}__{next_algo_int}_{extract_radius}r_{extract_point[0]}x{extract_point[1]}y_{time.time()}.wav"
            # np.save(str(out_file), hap_signal)
            scipy.io.wavfile.write(out_file, 8000, hap_signal)
            print(f"saved hap_signal -> {out_file}")

    user_study_video_idx = user_study_video_idx + 1

    if user_study_video_idx >= len(USER_STUDY_VIDEOS_AND_ALGORITHMS):
        print("User study tasks completed. No more videos to process.")
        return

    next_task = USER_STUDY_VIDEOS_AND_ALGORITHMS[user_study_video_idx]
    if isinstance(next_task, str):
        # next task is a prompt
        if next_task == "prompt survey":
            # show_centered_modal("Please fill out the survey form to continue.", False)
            show_algorithm_survey(video_processing_method)

        elif next_task == "prompt done":
            logger.log("Signal Creation Completed", f"participant_id={user_study_participant_id}, ab_flip={user_study_ab_flip}")
            print("Signal creation tasks completed. Continue to signal rating task.")
            with dpg.window(modal=True, no_move=True, no_close=True, no_collapse=True, no_title_bar=True,
                    width=int(gui_el_width * 0.75), height=int(gui_el_height * 0.75),
                    pos=(dpg.get_viewport_width() // 2 - int(gui_el_width * 0.75) // 2, dpg.get_viewport_height() // 2 - int(gui_el_height * 0.75) // 2)) as centered_modal_window:
                wshost_domain = websocket_url.split("//")[1].split("/")[0]
                dpg.add_text(f"Signal creation tasks completed.\nThank you for your participation so far!\nPlease continue to signal rating task.\nhttp://localhost:8081/?videorating&wsshost={wshost_domain}")
            return
        else:
            raise ValueError(f"Unknown task: {next_task}")
    else:
        next_vid, next_algo_int = next_task

        video_path               = next_vid
        video_processing_method  = vpm_ab_to_str(next_algo_int)
        process_params_changed   = True

dpg.create_context()
logger = EventLogger(path=user_study_participant_save_dir / "gui_events.log" if user_study_participant_save_dir else "gui_events.log")
logger.log("Application Start", "")
if user_study_participant_id and user_study_participant_save_dir and user_study_ab_flip:
    logger.log("User Study Start", f"participant_id={user_study_participant_id}, ab_flip={user_study_ab_flip}")

with dpg.font_registry():
    default_font = dpg.add_font("fonts/RobotoMono-Regular.ttf", 20)  # Use any .ttf you want, 18 is the font size
    smaller_font = dpg.add_font("fonts/RobotoMono-Regular.ttf", 17)
    dpg.bind_font(default_font)

with dpg.theme() as default_theme:
    # with dpg.theme_component(dpg.mvInfLineSeries): doesnt work
    #     dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 5.0)
    with dpg.theme_component(dpg.mvPlot):
        dpg.add_theme_style(dpg.mvPlotStyleVar_LabelPadding, 2, 2, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 4, 6, category=dpg.mvThemeCat_Plots)
        dpg.add_theme_style(dpg.mvPlotStyleVar_LegendInnerPadding, 3, 3, category=dpg.mvThemeCat_Plots)
dpg.bind_theme(default_theme)

primary_window = "primary_window"
with dpg.window(label="Primary Window", tag=primary_window):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Load Video", callback=lambda: dpg.show_item("load_dialog"))
        loading_text_el = dpg.add_text("Loading...", color=COLOR_TEXT_LOWPRI, tag="loading_text")
        dpg.add_button(label="Submit Signal", callback=submit_signal_callback, tag="submit_signal")

dpg.create_viewport(title='HapticLens - Vibration from Video', width=2300, height=1150)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.render_dearpygui_frame()
dpg.maximize_viewport()
dpg.render_dearpygui_frame()
dpg.render_dearpygui_frame()
dpg.render_dearpygui_frame()
dpg.set_primary_window(primary_window, True)

dpg.render_dearpygui_frame() # first render

def update_loading_text(text: str, dont_log: bool = False):
    dpg.set_value(loading_text_el, text)
    if not dont_log:
        print(f"loading: {text}")
    dpg.render_dearpygui_frame()

tic = cv2.getTickCount()

old_video_path = None
def get_or_load_video(colorspace: str, video_path: str, scale_factor: float) -> Tuple[List[np.ndarray], float, Tuple[int, int], int, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    global old_video_path, frames, video_fs, v_w, v_h, num_frames, inv_colorspace, colorspace_func, inv_colorspace_rgb
    global frames_rgb
    if video_path != old_video_path:
        frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func, inv_colorspace_rgb = load_video(colorspace, video_path, scale_factor)
        # for i in range(len(frames)):
        #     noise_level = 0.1
        #     noise = np.random.rand(*frames[i].shape)
        #     frames[i] = (1 - noise_level) * frames[i] + noise_level * noise
        frames_rgb = [inv_colorspace_rgb(frame) for frame in frames]
        old_video_path = video_path
    return frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func

last_phase_algo: Optional[PhaseAlgorithm] = None
def reprocess_video(tic: int, loading_text: str):
    global result_video, pds_ang_all, pds_ang_all_sub, mag_norm_all, mag_norm_all_sub, pd_r_norm, pd_i_norm, avg_r_norm, avg_i_norm
    global result_video_rgb, last_phase_algo

    transfer_function = create_transfer_function(sample_frequency, video_fs, num_frames, freq_lo, freq_hi)

    print(f"Performing Phase Based Motion Magnification \n")
    phase_algo = PhaseAlgorithm.LINEAR if video_processing_method == VideoProcessingMethod.LINEAR_PHASE.value else PhaseAlgorithm.ACCEL
    last_phase_algo = phase_algo
    if user_study_participant_save_dir and video_processing_method == VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value: # can skip processing
        result_video = np.zeros((num_frames, v_h, v_w), dtype=np.float32)
        phase_delta_dir_sum = np.zeros((num_frames, v_h, v_w, 2), dtype=np.float32)
    else:
        result_video, phase_delta_dir_sum = get_phase_deltas_and_mag_video(sigma, video_fs, transfer_function, phase_mag, attenuate, ref_idx, batch_size, colorspace, frames_tensor, filters_tensor, filter_dir_tensor, csp, EPS,
                                                                       ontick=lambda x: (update_loading_text(loading_text + f" Processing video... {x:.2%}", dont_log=True)) and None, phase_algo=phase_algo)
    result_video_rgb = [inv_colorspace_rgb(frame) for frame in result_video]
    toc_finished_processing = cv2.getTickCount()

    pd_r = phase_delta_dir_sum[..., 0]
    pd_i = phase_delta_dir_sum[..., 1]
    pds_mag_all, pds_ang_all = cv2.cartToPolar(pd_r, pd_i)
    mag_norm_all = (pds_mag_all - np.min(pds_mag_all)) / (np.max(pds_mag_all) - np.min(pds_mag_all)) # this normalization isnt great because the extraction is sampled over an area and these averages are much lower than the single pixel peak. Not sure what else would work especially if we allow custom shapes for the extraction region.

    with np.errstate(divide='ignore', invalid='ignore'):
        scale_mat = np.nan_to_num(mag_norm_all / pds_mag_all)

    pd_r_norm = pd_r * scale_mat
    pd_i_norm = pd_i * scale_mat

    avg_r_norm = np.mean(pd_r_norm, axis=(1, 2)) # type: ignore
    avg_i_norm = np.mean(pd_i_norm, axis=(1, 2)) # type: ignore
    # pds_mag_all_sub, pds_ang_all_sub = cv2.cartToPolar(pd_r - avg_r[:, None, None], pd_i - avg_i[:, None, None]) # subbing the avg in cartesian basically doesnt do anything
    # mag_norm_all_sub = (pds_mag_all_sub - np.min(pds_mag_all_sub)) / (np.max(pds_mag_all_sub) - np.min(pds_mag_all_sub))
    pds_ang_all_sub = pds_ang_all
    global_mag_avg = np.mean(mag_norm_all, axis=(1, 2)) # frame-wise avg magnitude
    mag_norm_all_sub = np.clip(mag_norm_all - global_mag_avg[:, None, None], 0, None) # subtract global avg

    if motion_vol:
        vtk_scalars = motion_vol.dataset.GetPointData().GetScalars()
        vtk_volume_complex = vtk2numpy(vtk_scalars)
        vtk_volume_complex[:, 0] = pds_ang_all.ravel(order="F")
        vtk_volume_complex[:, 1] = mag_norm_all.ravel(order="F")
        vtk_scalars.Modified()

    loading_text += f" Video processed at {get_time_elapsed(tic, toc_finished_processing)}."
    update_loading_text(loading_text)
    logger.log("Video Processed", f"{video_path}, {video_processing_method}, phase_algo={phase_algo}, sigma={sigma}, attenuate={attenuate}, ref_idx={ref_idx}, freq_lo={freq_lo}, freq_hi={freq_hi}")

def load_and_process_video(video_path: str):
    global old_video_path
    global extract_point, extract_radius, frames, video_fs, v_w, v_h, num_frames, inv_colorspace, colorspace_func
    global filters_tensor, filter_dir_tensor, frames_tensor, csp
    global stsaliency

    print(f"Loading video {video_path} at {get_time_elapsed(tic, cv2.getTickCount())}.")
    loading_text = f"processing at {get_time_elapsed(tic, cv2.getTickCount())}."
    update_loading_text(loading_text + " Loading video...", dont_log=True)

    frames, video_fs, (v_w, v_h), num_frames, inv_colorspace, colorspace_func = get_or_load_video(colorspace, video_path, scale_factor)
    toc_loaded_video = cv2.getTickCount()
    old_video_path = video_path

    extract_point = (v_w // 2, v_h // 2)
    extract_radius = int(np.ceil(min(v_w, v_h) // 15))

    loading_text += f" Video loaded at {get_time_elapsed(tic, toc_loaded_video)}."
    update_loading_text(loading_text + " Creating tensors...")

    filters_tensor, filter_dir_tensor, frames_tensor, csp = create_tensors(frames, ref_idx, pyramid_type, batch_size)
    toc_filters_and_tensors = cv2.getTickCount()

    loading_text += f" Tensors created at {get_time_elapsed(tic, toc_filters_and_tensors)}."
    update_loading_text(loading_text + " Processing video...")

    reprocess_video(tic, loading_text)
    loading_text = dpg.get_value(loading_text_el)

    if user_study_participant_save_dir and video_processing_method != VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value: # can skip stsaliency
        print("User study mode, skipping STSaliency computation.")
        stsaliency = np.zeros((num_frames, v_h, v_w), dtype=np.float32)
    else:
        update_loading_text(loading_text + " Computing STSal...")
        ss = SpatiotemporalSaliencyGPU(use_padded_buffer=True)
        stsaliency = ss.compute_spatiotemporal_saliency(frames)
        toc_saliency = cv2.getTickCount()
        loading_text += f" STSal computed at {get_time_elapsed(tic, toc_saliency)}."
        update_loading_text(loading_text)

    logger.log("Video Loaded", f"{video_path}, {video_processing_method}")


load_and_process_video(video_path)

# Dynamic vars
tic_last_render = tic_last_render_nodpg = cv2.getTickCount()
render_params_changed = True
extract_params_changed = True
vid_idx = 0
playback_start_time: Optional[float] = None
no_loop = False
x_axis = None

gl_tex_reg = None
def setup_textures():
    TEXREG = "texture_registry"
    global gl_tex_reg, frame_rgb_flat, processed_rgb_flat, pda_rgb_flat, hapspark_rgb_flat, vedo_rgb_flat

    calc_gui_sizes()

    for tag in ("frame_tex", "processed_tex", "pda_tex", "hapspark_tex", "vedo_tex"):
        if dpg.does_item_exist(tag):
            try:
                dpg.delete_item(tag)
                dpg.remove_alias(tag)
            except Exception as e:
                print(f"[WARN] Error deleting texture {tag}: {e}")
    frame_rgb_flat = np.zeros((v_h * v_w * 3), dtype=np.float32)
    processed_rgb_flat = np.zeros_like(frame_rgb_flat)
    pda_rgb_flat = np.zeros_like(frame_rgb_flat)
    hapspark_rgb_flat = np.zeros((hapspark_width * hapspark_height * 3), dtype=np.float32)
    vedo_rgb_flat = np.zeros((gui_el_width * gui_el_height * 3), dtype=np.float32)

    if gl_tex_reg is None:
        gl_tex_reg = dpg.add_texture_registry(show=False, tag=TEXREG)

    dpg.add_raw_texture(width=v_w,            height=v_h,              default_value=frame_rgb_flat, format=dpg.mvFormat_Float_rgb, tag="frame_tex", parent=gl_tex_reg) #type: ignore
    dpg.add_raw_texture(width=v_w,            height=v_h,              default_value=processed_rgb_flat, format=dpg.mvFormat_Float_rgb, tag="processed_tex", parent=gl_tex_reg)  #type: ignore
    dpg.add_raw_texture(width=v_w,            height=v_h,              default_value=pda_rgb_flat, format=dpg.mvFormat_Float_rgb, tag="pda_tex", parent=gl_tex_reg) #type: ignore
    dpg.add_raw_texture(width=hapspark_width, height=hapspark_height,  default_value=hapspark_rgb_flat, format=dpg.mvFormat_Float_rgb, tag="hapspark_tex", parent=gl_tex_reg) #type: ignore
    dpg.add_raw_texture(width=gui_el_width,   height=gui_el_height,    default_value=vedo_rgb_flat, format=dpg.mvFormat_Float_rgb, tag="vedo_tex", parent=gl_tex_reg) #type: ignore
    update_gui_sizes()

def calc_gui_sizes():
    global gui_el_width, gui_el_height, hapspark_width, hapspark_height
    MIN_WIDTH = 350
    MAX_WIDTH = 550
    MIN_HEIGHT = 250
    MAX_HEIGHT = 600

    SR_MIN_WIDTH = 450
    SR_MAX_WIDTH = 650 # 750
    SR_MIN_HEIGHT = 350
    SR_MAX_HEIGHT = 450

    def get_new_dim(orig_w, orig_h, min_w, max_w, min_h, max_h, maintain_aspect=True, try_alt_widths=None):
        ratio = orig_w / orig_h
        if not maintain_aspect:
            base_width = orig_w
            for alt_w in (try_alt_widths or []):
                if min_w <= alt_w <= max_w:
                    base_width = alt_w
                    break
            new_w = max(min_w, min(base_width, max_w))
            new_h = max(min_h, min(new_w/ratio, max_h))
            return int(new_w), int(new_h)
        s_min = max(min_w/orig_w, min_h/orig_h)
        s_max = min(max_w/orig_w, max_h/orig_h)
        if s_min <= s_max:
            scale = 1 if s_min <= 1 <= s_max else (s_min if 1 < s_min else s_max)
        else:
            scale = s_min

        new_w, new_h = orig_w*scale, orig_h*scale
        return int(new_w), int(new_h)

    orig_w, orig_h = v_w, v_h
    gui_el_width, gui_el_height = get_new_dim(orig_w, orig_h, MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT, maintain_aspect=True, try_alt_widths=None)
    hapspark_width, hapspark_height = get_new_dim(gui_el_width, gui_el_height, SR_MIN_WIDTH, SR_MAX_WIDTH, SR_MIN_HEIGHT, SR_MAX_HEIGHT, maintain_aspect=False, try_alt_widths=[(gui_el_width*4 +8*3 -8*2)/3])

def update_gui_sizes(nocalc=False):
    global gui_el_width, gui_el_height, hapspark_width, hapspark_height
    if not nocalc:
        calc_gui_sizes()

    print(f"Setting GUI sizes to {gui_el_width}x{gui_el_height} sr=({hapspark_width}x{hapspark_height}) for {v_w}x{v_h} video")
    if dpg.does_item_exist("frame_img"):
        for tag, tex in {
            "frame_img":"frame_tex",
            "processed_img":"processed_tex",
            "pda_img":"pda_tex",
            "vedo_img":"vedo_tex"
        }.items():
            dpg.configure_item(tag, texture_tag=tex, width=gui_el_width, height=gui_el_height) # todo: it might be worth doing a gamma preserving resize for the pda_img (eg resize in linear space, conv back to srgb)
        dpg.configure_item("hapspark_img", texture_tag="hapspark_tex", width=hapspark_width, height=hapspark_height)

    if dpg.does_item_exist("controls_window"):
        dpg.configure_item("controls_window", width=hapspark_width, height=hapspark_height+30)
    if dpg.does_item_exist("extract_plot"):
        dpg.configure_item("extract_plot", width=hapspark_width, height=hapspark_height+30)

    if dpg.does_item_exist("vid_idx"):
        dpg.configure_item("vid_idx", width=hapspark_width * 2 + 8 * 1)
        # dpg.configure_item("vid_idx", width=int(hapspark_width * 0.75)) # for screenshot

setup_textures()

with dpg.group(horizontal=True, parent=primary_window):
    with dpg.group():
        dpg.add_text("Original Frame")
        dpg.add_image("frame_tex", tag="frame_img")
    with dpg.group():
        pda_text = dpg.add_text("Phase Deltas (Value=Mag, Hue=Angle)")
        dpg.add_image("pda_tex", tag="pda_img")
    tdvolume_group = dpg.add_group()
    dpg.add_text("3D Volume View", parent=tdvolume_group)
    dpg.add_image("vedo_tex", tag="vedo_img", parent=tdvolume_group)
    magframe_group = dpg.add_group()
    dpg.add_text("Motion Magnified Frame (for reference)", parent=magframe_group)
    dpg.add_image("processed_tex", tag="processed_img", parent=magframe_group)

MOUSE_TRACK_IMGS = ["frame_img", "pda_img", "processed_img", "vedo_img"]
EXTRACT_IMGS = ["frame_img", "pda_img", "processed_img"]
THREED_IMGS = ["vedo_img"]
mouse_down_in_image = None
def image_click_callback(sender, app_data, user_data):
    global extract_point, extract_params_changed, mouse_down_in_image
    # print(f"image_click_callback: {user_data} with {mouse_down_in_image}")
    if user_data == "release" or dpg.is_item_shown("load_dialog") or algo_survey_active:
        mouse_down_in_image = None
        send_pcm_signal_debounce(hap_signal, debounce_immediate=True)
        return
    elif user_data == "down" and mouse_down_in_image is not None:
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

    for tag in MOUSE_TRACK_IMGS:
        if dpg.does_item_exist(tag) and dpg.is_item_shown(tag):
            min_x, min_y = dpg.get_item_rect_min(tag)
            max_x, max_y = dpg.get_item_rect_max(tag)

            if min_x <= mouse_x <= max_x and min_y <= mouse_y <= max_y:
                if user_data == "down":
                    mouse_down_in_image = tag
                elif user_data == "drag":
                    if mouse_down_in_image != tag:
                        return

                if tag in EXTRACT_IMGS:
                    rel_x = int((mouse_x - min_x) / (max_x - min_x) * v_w)
                    rel_y = int((mouse_y - min_y) / (max_y - min_y) * v_h)

                    if extract_point != (rel_x, rel_y):
                        extract_point = (rel_x, rel_y)
                        extract_params_changed = True
                        logger.log("Extract Point Changed", f"radius={extract_radius}, position={extract_point}")

                break
            else:
                if user_data == "down":
                    mouse_down_in_image = "NONIMAGE"
def image_wheel_callback(sender, app_data, user_data):
    global extract_radius, extract_params_changed

    if dpg.is_item_shown("load_dialog") or algo_survey_active:
        return

    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    for tag in MOUSE_TRACK_IMGS:
        if dpg.does_item_exist(tag) and dpg.is_item_shown(tag):
            min_x, min_y = dpg.get_item_rect_min(tag)
            max_x, max_y = dpg.get_item_rect_max(tag)

            if min_x <= mouse_x <= max_x and min_y <= mouse_y <= max_y:
                if tag in EXTRACT_IMGS:
                    extract_radius += app_data * 5
                    extract_radius = int(np.clip(extract_radius, 1, min(v_w, v_h)//2))
                    extract_params_changed = True
                    logger.log("Extract Point Changed", f"radius={extract_radius}, position={extract_point}")

                if tag in THREED_IMGS:
                    plt.zoom(1 + app_data * 0.2)
                    vedo_state.needs_update = True
                break

@dataclass
class VedoSceneState:
    last_pos: Optional[Tuple[float, float]] = None
    needs_update: bool = False
    sub_vol: bool = False
vedo_state = VedoSceneState()

def threed_view_move_callback(sender, app_data, user_data):
    if user_data == "release":
        vedo_state.last_pos = None
        return
    if mouse_down_in_image in THREED_IMGS:
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        if vedo_state.last_pos:
            dx = mouse_x - vedo_state.last_pos[0]
            dy = mouse_y - vedo_state.last_pos[1]
            plt.azimuth(dx * -0.15)
            plt.elevation(dy * 0.15)
            vedo_state.needs_update = True
        vedo_state.last_pos = (mouse_x, mouse_y)
    else:
        vedo_state.last_pos = None

with dpg.handler_registry():
    # dpg.add_mouse_click_handler(callback=image_click_callback, user_data="click")
    dpg.add_mouse_drag_handler(callback=image_click_callback, user_data="drag")
    dpg.add_mouse_down_handler(callback=image_click_callback, user_data="down")
    dpg.add_mouse_release_handler(callback=image_click_callback, user_data="release")
    dpg.add_mouse_wheel_handler(callback=image_wheel_callback)
    dpg.add_key_press_handler(dpg.mvKey_Period, callback=lambda s, a, u: update_global_render_param(s, np.clip(vid_idx + 1, 0, num_frames-1).item(), "vid_idx"))
    dpg.add_key_press_handler(dpg.mvKey_Comma, callback=lambda s, a, u: update_global_render_param(s, np.clip(vid_idx - 1, 0, num_frames-1).item(), "vid_idx"))
    for playpause_key in (dpg.mvKey_Spacebar, dpg.mvKey_K):
        dpg.add_key_press_handler(playpause_key, callback=lambda s, a, u: update_global_render_param(s, time.time() if playback_start_time is None else None, "playback_start_time"))
    def save_video_callback(sender, app_data, user_data):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            print("Saving video...")
            video_save_path = "gui_output.mp4"
            video_codec = "h264_nvenc"
            generate_output_video(frames, result_video, v_w, v_h, num_frames, inv_colorspace,
                                  mag_norm_all_sub if subtract_global_avg else mag_norm_all, pds_ang_all_sub if subtract_global_avg else pds_ang_all,
                                  extract_point, extract_radius, signal_at_coords, hap_signal, video_save_path, video_codec, video_fs)

    dpg.add_key_press_handler(dpg.mvKey_S, callback=save_video_callback)
    def handle_esc(sender, app_data, user_data):
        global training_idx
        if in_training:
            training_idx = len(training_steps)
            render_training_step()
    dpg.add_key_press_handler(dpg.mvKey_Escape, callback=handle_esc)
    dpg.add_mouse_drag_handler(callback=threed_view_move_callback, user_data="drag")
    dpg.add_mouse_release_handler(callback=threed_view_move_callback, user_data="release")

def update_param_logger(sender, app_data, user_data, old_value):
    """Callback to log parameter changes from the GUI."""
    match user_data:
        case "playback_start_time":
            event = "Playback Started from UI" if app_data is not None else "Playback Stopped from UI"
            logger.log(event, "")
        case "vid_idx":
            event_type = "Video Scrub (slider)" if sender == "vid_idx" else "Video Scrub (keyboard)"
            logger.log(event_type, f"{old_value}->{app_data}")
        case "extract_radius":
            logger.log("Extract Point Changed", f"radius={extract_radius}, position={extract_point}")
        case "extract_point":
            logger.log("Extract Point Changed", f"radius={extract_radius}, position={extract_point}")

def update_global_render_param(sender, app_data, user_data):
    global render_params_changed
    old_value = globals().get(user_data)
    globals()[user_data] = app_data
    render_params_changed = True
    update_param_logger(sender, app_data, user_data, old_value)

def update_global_extract_param(sender, app_data, user_data):
    global extract_params_changed
    old_value = globals().get(user_data)
    globals()[user_data] = app_data
    extract_params_changed = True
    update_param_logger(sender, app_data, user_data, old_value)

def update_global_process_param(sender, app_data, user_data):
    global process_params_changed
    old_value = globals().get(user_data)
    globals()[user_data] = app_data
    process_params_changed = True
    update_param_logger(sender, app_data, user_data, old_value)

def load_new_video_callback(sender, app_data, user_data):
    # new_path = app_data.get("file_path_name") # doesnt resolve file extension correctly
    new_path = next(iter(app_data.get("selections", {}).values()), None)
    if not new_path:
        return
    print("Loading new video:", new_path)
    update_global_process_param("load_dialog", new_path, "video_path")
    # reload_video(new_path) cannot be run from callback

def proc_type_radio_callback(sender, app_data, user_data):
    global video_processing_method, process_params_changed
    if app_data == video_processing_method:
        return
    video_processing_method = app_data
    process_params_changed = True
    logger.log("Algorithm Changed", f"{video_processing_method}")

with dpg.group(horizontal=True, parent=primary_window) as gconout:
    with dpg.theme() as controls_theme:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (25, 25, 25), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Header, (62, 62, 62), category=dpg.mvThemeCat_Core)
    dpg.bind_item_theme(gconout, controls_theme)
    with dpg.file_dialog(directory_selector=False, show=False, callback=load_new_video_callback, tag="load_dialog", default_path="videos/userstudy/"):
        dpg.add_file_extension(".*",   custom_text="All Files (.*)")
        dpg.add_file_extension(".mp4", custom_text="MP4 Files (.mp4)")
        dpg.add_file_extension(".avi", custom_text="AVI Files (.avi)")

    with dpg.child_window(label="controls", tag="controls_window", width=v_w, height=v_h+30, border=False) as controls_window:
        with dpg.collapsing_header(label="Configuration", default_open=True):
            with dpg.group(horizontal=True):
                dpg.add_text("x, y:")
                dpg.add_input_intx(tag="extract_point", default_value=extract_point, size=2, min_value=0, max_value=max(v_w, v_h), callback=lambda s, a, u: update_global_extract_param(s, a[:2], u), user_data="extract_point")
            with dpg.group(horizontal=True):
                dpg.add_text("Radius:")
                dpg.add_input_int(tag="extract_radius", default_value=extract_radius, min_value=1, max_value=min(v_w, v_h)//2, min_clamped=True, max_clamped=True, callback=update_global_extract_param, user_data="extract_radius")
            with dpg.group(horizontal=True) as subtract_global_avg_group:
                dpg.add_text("Subtract Global Avg Motion:")
                dpg.add_checkbox(tag="subtract_global_avg", default_value=subtract_global_avg, callback=update_global_extract_param, user_data="subtract_global_avg")
            with dpg.group(horizontal=True):
                dpg.add_text("RMS instead of mean:")
                dpg.add_checkbox(tag="rms_instead_of_mean", default_value=rms_instead_of_mean, callback=update_global_extract_param, user_data="rms_instead_of_mean")
            with dpg.group(horizontal=True):
                dpg.add_text("Use angle avg extraction:")
                dpg.add_checkbox(tag="use_ang_extraction", default_value=use_ang_extraction, callback=update_global_extract_param, user_data="use_ang_extraction")
        with dpg.collapsing_header(label="Processing Params (Requires Reprocessing)", default_open=True) as process_params_header:
            with dpg.tooltip(process_params_header):
                dpg.add_text("These affect the magnified video output, but appear to be less important for haptic signal generation.")
            with dpg.group(horizontal=True):
                dpg.add_text("freq_lo:")
                dpg.add_input_float(tag="freq_lo", default_value=freq_lo, min_value=0.01, max_value=video_fs//2 - 0.1, min_clamped=True, max_clamped=True, callback=update_global_process_param, user_data="freq_lo", on_enter=True)
            with dpg.group(horizontal=True):
                dpg.add_text("freq_hi:")
                dpg.add_input_float(tag="freq_hi", default_value=freq_hi, min_value=0.01, max_value=video_fs//2 - 0.1, min_clamped=True, max_clamped=True, callback=update_global_process_param, user_data="freq_hi", on_enter=True)
        with dpg.collapsing_header(label="Vision Algorithm", default_open=True) as video_processing_group:
            dpg.add_radio_button(
                [method.value for method in VideoProcessingMethod],
                default_value=video_processing_method,
                tag="video_processing_method",
                callback=proc_type_radio_callback,
                user_data="video_processing_method"
            )
            if user_study_participant_save_dir:
                dpg.hide_item(video_processing_group)  # hide in user study mode


    with dpg.plot(label="Extracted Magnitude Signals", tag="extract_plot", height=v_h+30, width=v_w) as plot:
        dpg.add_plot_legend()

        x_axis = dpg.add_plot_axis(dpg.mvXAxis)
        dpg.set_axis_limits(x_axis, -0.5, num_frames)

        with dpg.plot_axis(dpg.mvYAxis) as y_axis:
            null_x: list[float] = np.arange(0, num_frames, dtype=float).tolist()
            null_y: list[float] = np.zeros(num_frames, dtype=float).tolist()
            dpg.add_line_series(null_x, null_y, tag="extracted_mean_delta_mag", label="Amplitude")
            dpg.add_line_series(null_x, null_y, tag="signal_resampled_norm", label="resamp+norm", show=False)
            # dpg.add_line_series(null_x, null_y, tag="accel", label="accel")
            # dpg.add_line_series(null_x, null_y, tag="accel_mag", label="accel_mag")
            dpg.add_line_series(null_x, null_y, tag="accel_resampled", label="Frequency")
            # dpg.add_line_series(null_x, null_y, tag="accel_resampled_mag", label="accel_resampled_mag")
            dpg.add_inf_line_series([vid_idx], tag="playback_head")
            dpg.set_axis_limits_constraints(y_axis, -0.1, 1.1)
            dpg.set_axis_limits(y_axis, -0.1, 1.0)
            # dpg.set_axis_limits_auto(y_axis)
            # dpg.reset_axis_limits_constraints(y_axis)


    with dpg.group():
        dpg.add_text("Generated Haptic Signal (Normalized)")
        dpg.add_image("hapspark_tex", tag="hapspark_img")

with dpg.group(horizontal=True, parent=primary_window):
    # dpg.add_spacer(width=v_w)
    slider_w = v_w * 2 + 8 * 1
    dpg.add_slider_int(tag="vid_idx", width=slider_w, min_value=0, max_value=num_frames-1, clamped=True, default_value=vid_idx, callback=update_global_render_param, user_data="vid_idx")
    dpg.add_text("Frame @ 0.00s", tag="frame_num")#, color=COLOR_TEXT_LOWPRI)
    dpg.add_button(label="Play/Pause", tag="play_pause", callback=lambda s, a, u: update_global_render_param(s, time.time() if playback_start_time is None else None, "playback_start_time"))


# dpg.show_style_editor()
# dpg.show_metrics()

dpg.add_spacer(height=10, parent=primary_window)
with dpg.group(horizontal=True, parent=primary_window):
    dpg.add_text(f"render tics: 0", tag="render_tic", color=COLOR_TEXT_LOWPRI, parent=primary_window)
    dpg.add_spacer(width=10)
    dpg.add_text("no quests connected...", tag="quest_status")
    dpg.add_loading_indicator(tag="quest_signal_not_acked_spinner", style=1, radius=1.1)
    dpg.hide_item("quest_signal_not_acked_spinner")  # Initially hidden

signal_at_coords = np.zeros(num_frames, dtype=float)
hap_signal = np.zeros(int(num_frames * (HAPTIC_SAMPLE_RATE / video_fs)), dtype=float)

def setup_vedo_scene():
    global volume_frame_spacing, motion_vol, saliency_vol, final_frame_plane, final_frame_bg
    global plt, vtk_volume_complex, vtk_sal, extract_point_bb_lines, frame_marker, objects

    volume_frame_spacing = max(1, (min(v_h, v_w) / 2) / num_frames)

    motion_vol = Volume(None, dims=(num_frames, v_h, v_w))
    motion_vol.mode(0)
    motion_vol.jittering(False)
    motion_vol.properties.SetInterpolationTypeToNearest()
    motion_vol.properties.IndependentComponentsOff()
    motion_vol.color([
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 1, 1),
        (0, 0, 1),
        (1, 0, 1),
        (1, 0, 0)
    ], vmin=0, vmax=2*np.pi)
    # motion_vol.alpha([(x, x**3) for x in np.linspace(0, 1, 10)], vmin=0, vmax=1)
    motion_vol.alpha(cubic_bezier_points(0.85, 0.0, 0.77, 1.0, n=10), vmin=0, vmax=1)
    motion_vol.dataset.AllocateScalars(vtkclasses.VTK_FLOAT, 2)
    motion_vol.dataset.SetSpacing(volume_frame_spacing, 1, 1)
    vtk_volume_complex = vtk2numpy(motion_vol.dataset.GetPointData().GetScalars())
    vtk_volume_complex[:, 0] = pds_ang_all.ravel(order="F")
    vtk_volume_complex[:, 1] = mag_norm_all.ravel(order="F")
    motion_vol.actor.SetUserTransform(LinearTransform().scale([1, -1, 1]).translate([0, v_h, 0]).T)

    saliency_vol = Volume(None, dims=(num_frames, v_h, v_w))
    saliency_vol.mode(0)
    saliency_vol.jittering(False)
    saliency_vol.properties.SetInterpolationTypeToNearest()
    saliency_vol.color(["black", "white"], vmin=0, vmax=1)
    saliency_vol.alpha([(x, x**3) for x in np.linspace(0, 1, 10)], vmin=0, vmax=1)
    saliency_vol.dataset.AllocateScalars(vtkclasses.VTK_FLOAT, 1)
    saliency_vol.dataset.SetSpacing(volume_frame_spacing, 1, 1)
    vtk_sal = vtk2numpy(saliency_vol.dataset.GetPointData().GetScalars())
    vtk_sal[:] = stsaliency.ravel(order='F')
    saliency_vol.actor.SetUserTransform(LinearTransform().scale([1, -1, 1]).translate([0, v_h, 0]).T)

    final_frame_bg = Plane(pos=[(num_frames*volume_frame_spacing)+16, v_h/2, v_w/2], normal=[1, 0, 0], s=(v_w, v_h), alpha=0.8, c="black")
    final_frame_plane = Plane(pos=[(num_frames*volume_frame_spacing)+15, v_h/2, v_w/2], normal=[1, 0, 0], s=(v_w, v_h), alpha=0.3)
    final_frame_plane.texture(Image(np.fliplr(frames_rgb[-1]), channels=3))

    if 'plt' in globals():
        try:
            plt.close() # type: ignore
        except Exception:
            pass
    plt = Plotter(offscreen=True, axes=0, size=(gui_el_width, gui_el_height), bg="black") # type: ignore

    extract_point_bb_lines = vedo.shapes.Lines(
        start_pts=generate_box_wireframe([0, -1, -1], [1, 1, 1]),
        c="red",
        lw=3,
    )

    frame_marker = vedo.shapes.Line(
        p0=[
            [0, 15, -10],
            [0, 0, -10],
            [0, 0, v_w + 10],
            [0, 15, v_w + 10],
        ],
        lw=3,
        c="blue",
    )

    objects = [frame_marker, extract_point_bb_lines, motion_vol, saliency_vol, final_frame_plane, final_frame_bg]
    plt.show(*objects, interactive=False)
    assert plt.camera is not None, "Camera not initialized"
    plt.azimuth(-80)
    plt.elevation(20)


def cleanup_vedo_scene():
    """Release vedo OpenGL resources before loading another video."""
    if 'plt' in globals():
        try:
            plt.close()  # type: ignore
        except Exception:
            pass
        del globals()['plt']
    for name in (
        'motion_vol', 'saliency_vol', 'final_frame_plane',
        'final_frame_bg', 'objects', 'extract_point_bb_lines',
        'frame_marker', 'vtk_volume_complex', 'vtk_sal'
    ):
        if name in globals():
            del globals()[name]
    gc.collect()


def reload_video(path: str):
    global tic
    tic = cv2.getTickCount()
    cleanup_vedo_scene()  # untested
    # cleanup_torch_state()  # untested

    global video_path, vid_idx, playback_start_time, motion_vol
    video_path = path
    vid_idx = 0
    playback_start_time = None

    motion_vol = None
    load_and_process_video(video_path)
    loading_text = dpg.get_value(loading_text_el)
    print(f"Reloading UI for {video_path} at {get_time_elapsed(tic, cv2.getTickCount())}.")
    update_loading_text(loading_text + " Recreating Textures...")
    setup_textures()
    print(f"Textures recreated at {get_time_elapsed(tic, cv2.getTickCount())}.")
    update_loading_text(loading_text + " Recreating 3D Scene...")
    setup_vedo_scene()
    print(f"3D Scene recreated at {get_time_elapsed(tic, cv2.getTickCount())}.")
    update_loading_text(loading_text + " Reinitializing GUI...")
    dpg.configure_item("vid_idx", max_value=num_frames-1, default_value=0)
    if x_axis is not None:
        dpg.set_axis_limits(x_axis, -0.5, num_frames)
    null_x = np.arange(0, num_frames, dtype=float).tolist()
    null_y = np.zeros(num_frames, dtype=float).tolist()
    dpg.set_value("extracted_mean_delta_mag", [null_x, null_y])
    dpg.set_value("signal_resampled_norm", [null_x, null_y])
    dpg.set_value("accel_resampled", [null_x, null_y])
    dpg.set_value("playback_head", [[0.0]])
    global signal_at_coords, hap_signal
    signal_at_coords = np.zeros(num_frames, dtype=float)
    hap_signal = np.zeros(int(num_frames * (HAPTIC_SAMPLE_RATE / video_fs)), dtype=float)
    global render_params_changed, extract_params_changed, process_params_changed
    render_params_changed = extract_params_changed = process_params_changed = True
    vedo_state.needs_update = True
    dpg.set_value("video_processing_method", video_processing_method) # update radio button state, in case it was changed
    update_loading_text(loading_text + " GUI Ready at " + get_time_elapsed(tic, cv2.getTickCount()))
    logger.log("Video Reloaded, UI Ready", f"{video_path}, {video_processing_method}")

setup_vedo_scene()

update_gui_sizes()

with dpg.window(tag="centered_modal", modal=True, no_move=True, no_close=True, no_collapse=True, no_title_bar=True,
                    width=int(gui_el_width * 0.75), height=int(gui_el_height * 0.75),
                    pos=(dpg.get_viewport_width() // 2 - int(gui_el_width * 0.75) // 2, dpg.get_viewport_height() // 2 - int(gui_el_height * 0.75) // 2)) as centered_modal_window:
    centered_modal_text = dpg.add_text("")
    centered_modal_loading_spinner = dpg.add_loading_indicator(style=0, radius=min(gui_el_width, gui_el_height) * 0.03)
dpg.hide_item(centered_modal_window)

def show_loading_modal(text, spinner=True, immediate=True):
    dpg.show_item(centered_modal_window)
    dpg.set_value(centered_modal_text, text)
    if spinner:
        dpg.show_item(centered_modal_loading_spinner)
    else:
        dpg.hide_item(centered_modal_loading_spinner)
    if immediate:
        dpg.render_dearpygui_frame()
def hide_loading_modal():
    dpg.hide_item(centered_modal_window)


# Quest websocket --------------------------------------------------------------
websocket_url = sys.argv[3] if len(sys.argv) > 3 else "ws://localhost:8081/ws"
quest_ws = QuestWebSocketClient(websocket_url, ws_log=lambda msg: print(f"[WS] {msg}"))
quest_device_info = None

def _quest_ws_message(msg: dict) -> None:
    global playback_start_time, no_loop, vid_idx, quest_device_info, render_params_changed
    cmd = msg.get("cmd")
    if cmd == "starting_playback":
        playback_start_time = time.time()
        no_loop = True
        logger.log("Playback Started From Controller", "")
    elif cmd == "stopping_playback":
        playback_start_time = None
        no_loop = False
        vid_idx = 0
        render_params_changed = True
        logger.log("Playback Stopped From Controller", "")
    elif cmd == "ack_haptic_signal":
        dpg.hide_item("quest_signal_not_acked_spinner")
    elif "systemId" in msg:
        quest_device_info = msg
        update_device_status()
    else:
        if quest_ws.ws_log: quest_ws.ws_log(f"[WARN] Unknown message cmd => {cmd}")

def update_device_status(forvis=False) -> None:
    if quest_device_info:
        l = quest_device_info.get("haptic_sample_rate", {}).get("left", 0)
        r = quest_device_info.get("haptic_sample_rate", {}).get("right", 0)
        txt = f"{quest_device_info.get('systemName', 'Quest')} L:{l}Hz R:{r}Hz"
    else:
        txt = "Quest: not connected"
    if forvis:
        txt = f"Quest Controllers L:2000Hz R:2000Hz"
    dpg.set_value("quest_status", txt)
update_device_status(forvis=True)

send_pcm_signal_debounce = Debouncer(quest_ws.send_pcm_signal, interval_sec=2.0)

quest_ws.register_on_message(_quest_ws_message)


# dpg.add_draw_layer(tag="training_overlay", parent=primary_window)
with dpg.window(tag="training_overlay_window", pos=(0,0), width=dpg.get_viewport_width(), height=dpg.get_viewport_height(),
                no_title_bar=True, no_close=True, no_resize=True, no_background=True, no_move=True, no_scrollbar=True, no_collapse=True, no_scroll_with_mouse=True, no_bring_to_front_on_focus=False):
    with dpg.drawlist(width=dpg.get_viewport_width(), height=dpg.get_viewport_height(), tag="training_overlay_drawlist"):
        dpg.add_draw_layer(tag="training_overlay")

def resize_training_overlay():
    dpg.configure_item("training_overlay_window", width=dpg.get_viewport_width(), height=dpg.get_viewport_height())
    dpg.configure_item("training_overlay_drawlist", width=dpg.get_viewport_width(), height=dpg.get_viewport_height())
    if in_training:
        render_training_step()
dpg.set_viewport_resize_callback(lambda: resize_training_overlay())
dpg.set_frame_callback(dpg.get_frame_count() + 1, lambda: dpg.configure_item("training_overlay_window", no_bring_to_front_on_focus=True))
dpg.set_frame_callback(dpg.get_frame_count() + 2, lambda: dpg.hide_item("training_overlay_window"))

print("UI Ready")
logger.log("UI Ready", f"Video: {video_path}, Processing Method: {video_processing_method}")
if user_study_participant_save_dir and "training.mp4" in video_path:
    dpg.set_frame_callback(dpg.get_frame_count() + 3, lambda: start_training())
# Render loop
while dpg.is_dearpygui_running():
    if playback_start_time is not None:
        elapsed_time = time.time() - playback_start_time
        next_frame_idx = int(elapsed_time * video_fs)
        if next_frame_idx >= num_frames:
            if no_loop:
                playback_start_time = None # play once
            else:
                playback_start_time = time.time() # loop
            vid_idx = 0
        else:
            vid_idx = next_frame_idx
        render_params_changed = True

    dpg.set_item_label("play_pause", "Play" if playback_start_time is None else "Pause")

    dpg.set_value("vid_idx", vid_idx)
    dpg.set_value("extract_point", extract_point)
    dpg.set_value("extract_radius", extract_radius)
    dpg.set_value("subtract_global_avg", subtract_global_avg)
    dpg.set_value("freq_lo", freq_lo)
    dpg.set_value("freq_hi", freq_hi)


    frame_time_sec = vid_idx * np.float64(1.0) / video_fs
    vid_perc = vid_idx / (num_frames - 1)
    # dpg.set_item_label("vid_idx", f"Frame @ {frame_time_sec:.2f}s ({vid_perc:.2%})")
    dpg.set_value("frame_num", f"Frame @ {frame_time_sec:5.2f}s ({vid_perc:6.2%})")

    toc_inputs = cv2.getTickCount()

    if render_params_changed or extract_params_changed or process_params_changed or vedo_state.needs_update:

        if process_params_changed and video_path != old_video_path:
            show_loading_modal("Loading next video, please wait...")
            reload_video(video_path)
        elif process_params_changed and video_processing_method != VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value:
            # print(f"last_phase_algo: {last_phase_algo}, video_processing_method: {video_processing_method}") reprocessing wasnt working for a sec but couldnt reproduce...
            if ((last_phase_algo == PhaseAlgorithm.ACCEL and video_processing_method != VideoProcessingMethod.ACCELERATION_PHASE.value) or
                (last_phase_algo == PhaseAlgorithm.LINEAR and video_processing_method != VideoProcessingMethod.LINEAR_PHASE.value)):
                show_loading_modal("Reprocessing video, please wait...")
                reprocess_video(cv2.getTickCount(), "Reprocessing for new params...")
        hide_loading_modal()

        if extract_params_changed or process_params_changed:
            motion_data_matrix = None
            get_in_radius = get_rms_in_radius if rms_instead_of_mean else get_average_in_radius #perf todo: switch to sample_box_mean_cupy
            if video_processing_method == VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value:
                motion_data_matrix = stsaliency
                signal_at_coords = get_in_radius(motion_data_matrix, extract_point, extract_radius)
            else:
                if use_ang_extraction:
                    assert rms_instead_of_mean is False, "RMS extraction is not supported with angle-based processing methods."
                    avg_i_at_coords = get_in_radius(pd_i_norm, extract_point, extract_radius)
                    avg_r_at_coords = get_in_radius(pd_r_norm, extract_point, extract_radius)
                    if subtract_global_avg:
                        avg_i_at_coords -= avg_i_norm
                        avg_r_at_coords -= avg_r_norm
                    signal_at_coords = np.sqrt(avg_i_at_coords**2 + avg_r_at_coords**2)
                else:
                    motion_data_matrix = mag_norm_all_sub if subtract_global_avg else mag_norm_all
                    signal_at_coords = get_in_radius(motion_data_matrix, extract_point, extract_radius)
            toc_get_avg = cv2.getTickCount()

            dpg.set_value("extracted_mean_delta_mag", [np.arange(0, num_frames).tolist(), signal_at_coords.tolist()])
            hap_signal, signal_resampled_norm, accel, accel_resampled = create_hap_signal(signal_at_coords, video_fs)
            # print(f"Created haptic signal with {len(hap_signal)} samples at {HAPTIC_SAMPLE_RATE}Hz, min={hap_signal.min():.3f}, max={hap_signal.max():.3f}, mean={hap_signal.mean():.3f}, std={hap_signal.std():.3f}")
            send_pcm_signal_debounce(hap_signal)
            dpg.show_item("quest_signal_not_acked_spinner")  # Show spinner while signal is being sent
            toc_create_hap_signal = cv2.getTickCount()
            dpg.set_value("signal_resampled_norm", [np.arange(0, num_frames, video_fs/HAPTIC_SAMPLE_RATE).tolist(), signal_resampled_norm.tolist()])
            # dpg.set_value("accel", [np.arange(0, num_frames).tolist(), accel.tolist()])
            dpg.set_value("accel_resampled", [np.arange(0, num_frames, video_fs/HAPTIC_SAMPLE_RATE).tolist(), accel_resampled.tolist()])
            # print(f"toc_get_avg: {get_time_elapsed(toc_inputs, toc_get_avg)}, toc_create_hap_signal: {get_time_elapsed(toc_get_avg, toc_create_hap_signal)}")

            # ep_lines.transform.reset()
            extract_point_bb_lines.scale([num_frames*volume_frame_spacing+15*3, extract_radius, extract_radius], reset=True)
            extract_point_bb_lines.pos(x=-15, y=v_h-extract_point[1], z=extract_point[0])

        toc_hap_signal = cv2.getTickCount()

        if render_params_changed or extract_params_changed or process_params_changed:
            dpg.set_value("playback_head", [[float(vid_idx)]])

            rgb_frame = frames_rgb[vid_idx].copy()
            rgb_processed = result_video_rgb[vid_idx].copy()

            if video_processing_method == VideoProcessingMethod.SPATIOTEMPORAL_SALIENCY.value:
                dpg.set_value(pda_text, "Spatiotemporal Saliency")
                dpg.show_item(tdvolume_group)
                dpg.hide_item(magframe_group)
                dpg.hide_item(subtract_global_avg_group)
                dpg.hide_item(process_params_header)
                saliency_vol.actor.SetVisibility(1)
                motion_vol.actor.SetVisibility(0) # type: ignore
                rgb_pda = cv2.cvtColor((stsaliency[vid_idx] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                dpg.set_value(pda_text, "Phase (Value=Mag, Hue=Angle)")
                dpg.show_item(magframe_group)
                dpg.show_item(tdvolume_group)
                if user_study_participant_id is not None:
                    dpg.hide_item(subtract_global_avg_group) # hide in user study mode
                else:
                    dpg.show_item(subtract_global_avg_group)
                if video_processing_method == VideoProcessingMethod.ACCELERATION_PHASE.value:
                    dpg.hide_item(process_params_header)
                else:
                    dpg.show_item(process_params_header)
                saliency_vol.actor.SetVisibility(0)
                motion_vol.actor.SetVisibility(1) # type: ignore

                hsv_pda = np.zeros_like(rgb_frame)
                # pds_mag, pds_ang = cv2.cartToPolar(phase_delta_dir_sum[vid_idx][..., 0], phase_delta_dir_sum[vid_idx][..., 1])
                pds_mag = (mag_norm_all_sub if subtract_global_avg else mag_norm_all)[vid_idx]
                pds_ang = (pds_ang_all_sub if subtract_global_avg else pds_ang_all)[vid_idx]
                hsv_pda[..., 0] = pds_ang * 180 / np.pi / 2
                hsv_pda[..., 1] = 255
                hsv_pda[..., 2] = pds_mag * 255
                cv2.cvtColor(hsv_pda, cv2.COLOR_HSV2RGB, hsv_pda)
                rgb_pda = hsv_pda

            toc_bgrs = cv2.getTickCount()

            # bgr_pdasparkline = draw_sparkline(signal_at_coords, vid_perc, (hapspark_width, hapspark_height), 0, 1)
            bgr_hapsparkline = draw_sparkline(hap_signal, vid_perc, (hapspark_width, hapspark_height), -1, 1, playback_head_line=True)

            toc_sparklines = cv2.getTickCount()

            render_extract_point_rgb(rgb_frame, extract_point, extract_radius)
            render_extract_point_rgb(rgb_processed, extract_point, extract_radius)
            render_extract_point_rgb(rgb_pda, extract_point, extract_radius)

            toc_renderepoint = cv2.getTickCount()

            np.copyto(frame_rgb_flat, (rgb_frame.astype(np.float32) / 255.0).ravel())
            np.copyto(processed_rgb_flat, (rgb_processed.astype(np.float32) / 255.0).ravel())
            np.copyto(pda_rgb_flat, (rgb_pda.astype(np.float32) / 255.0).ravel())
            bgr_to_float_rgb_flat(bgr_hapsparkline, hapspark_rgb_flat)

            toc_update_tex = cv2.getTickCount()
        else:
            toc_bgrs = toc_hap_signal
            toc_sparklines = toc_hap_signal
            toc_renderepoint = toc_hap_signal
            toc_update_tex = toc_hap_signal

        if vedo_state.needs_update or extract_params_changed or process_params_changed or render_params_changed:
            if extract_params_changed:
                if subtract_global_avg != vedo_state.sub_vol:
                    vtk_volume_complex[:, 0] = (pds_ang_all_sub if subtract_global_avg else pds_ang_all).ravel(order="F")
                    vtk_volume_complex[:, 1] = (mag_norm_all_sub if subtract_global_avg else mag_norm_all).ravel(order="F")
                    vedo_state.sub_vol = subtract_global_avg
                    motion_vol.dataset.GetPointData().GetScalars().Modified() # type: ignore
            frame_marker.pos(x=vid_idx*volume_frame_spacing, y=0, z=0)
            screenshot = plt.screenshot(asarray=True)
            np.copyto(vedo_rgb_flat, (screenshot.astype(np.float32) / 255.0).ravel())

            toc_rendervedo = cv2.getTickCount()
        else:
            toc_rendervedo = toc_update_tex


        render_params_changed = False
        process_params_changed = False
        extract_params_changed = False
        vedo_state.needs_update = False
    else:
        toc_hap_signal = toc_inputs
        toc_bgrs = toc_inputs
        toc_sparklines = toc_inputs
        toc_renderepoint = toc_inputs
        toc_rendervedo = toc_inputs
        toc_update_tex = toc_inputs


    tics_elapsed = cv2.getTickCount() - tic_last_render_nodpg
    ms_elapsed = tics_elapsed / cv2.getTickFrequency() * 1000

    fromtic = tic_last_render_nodpg # or tic_last_render, but render_dearpygui_frame includes wait
    toc_data = [
        ("inp", toc_inputs),
        ("hap", toc_hap_signal),
        ("bgr", toc_bgrs),
        ("spr", toc_sparklines),
        ("rect", toc_renderepoint),
        ("tex", toc_update_tex),
        ("vedo", toc_rendervedo)
    ]
    toc_names, tocs = zip(*toc_data)
    percents = np.diff(tocs, prepend=fromtic) / tics_elapsed
    perc_str = ", ".join([f"{name} {perc:.2f}" for name, perc in zip(toc_names, percents)])
    dpg.set_value("render_tic", f"elapsed: {ms_elapsed:6.2f} ms. RT%: {perc_str}")

    tic_last_render = cv2.getTickCount()
    dpg.render_dearpygui_frame()
    tic_last_render_nodpg = cv2.getTickCount()

logger.log("Application Exit", "")
logger.close()
dpg.destroy_context()
