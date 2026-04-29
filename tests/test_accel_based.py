import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from motion_mag_api import create_transfer_function, create_tensors, get_phase_deltas_and_mag_video, PhaseAlgorithm

def generate_test_frames(num_frames=20, h=32, w=32, amp=5):
    """Generate a simple grayscale video of a square moving sinusoidally."""
    frames = []
    for i in range(num_frames):
        frame = np.zeros((h, w), dtype=np.float32)
        x = int(w / 2 + amp * np.sin(2 * np.pi * i / num_frames))
        y = h // 2
        frame[y - 2 : y + 2, x - 2 : x + 2] = 1.0
        frames.append(frame)
    return frames


def save_expected_output(path=os.path.join(os.path.dirname(__file__), "data", "expected_accel_output.npz")):
    """Utility function to (re)generate the expected output file."""
    fs = 30.0
    frames = generate_test_frames()
    filters_tensor, filter_dir_tensor, frames_tensor, csp = create_tensors(
        frames, ref_idx=0, pyramid_type="half_octave", batch_size=4
    )
    transfer_function = create_transfer_function(
        sample_frequency=-1,
        video_fs=fs,
        num_frames=len(frames),
        freq_lo=0.5,
        freq_hi=2.0,
    )
    result_video, phase_delta_dir_sum = get_phase_deltas_and_mag_video(
        0.0,
        fs,
        transfer_function,
        5.0,
        False,
        0,
        4,
        "gray",
        frames_tensor,
        filters_tensor,
        filter_dir_tensor,
        csp,
        1e-6,
        phase_algo=PhaseAlgorithm.ACCEL,
    )
    np.savez(path, result_video=result_video, phase_delta_dir_sum=phase_delta_dir_sum)


def test_accel_based_regression():
    fs = 30.0
    frames = generate_test_frames()
    filters_tensor, filter_dir_tensor, frames_tensor, csp = create_tensors(
        frames, ref_idx=0, pyramid_type="half_octave", batch_size=4
    )
    transfer_function = create_transfer_function(
        sample_frequency=-1,
        video_fs=fs,
        num_frames=len(frames),
        freq_lo=0.5,
        freq_hi=2.0,
    )
    result_video, phase_delta_dir_sum = get_phase_deltas_and_mag_video(
        0.0,
        fs,
        transfer_function,
        5.0,
        False,
        0,
        4,
        "gray",
        frames_tensor,
        filters_tensor,
        filter_dir_tensor,
        csp,
        1e-6,
        phase_algo=PhaseAlgorithm.ACCEL,
    )

    expected = np.load(os.path.join(os.path.dirname(__file__), "data", "expected_accel_output.npz"))
    exp_video = expected["result_video"]
    exp_accel = expected["phase_delta_dir_sum"]

    assert result_video.shape == exp_video.shape
    assert phase_delta_dir_sum.shape == exp_accel.shape

    diff_video = np.abs(result_video - exp_video).mean()
    diff_accel = np.abs(phase_delta_dir_sum - exp_accel).mean()

    print(f"Video difference: {diff_video}")
    print(f"Acceleration difference: {diff_accel}")

    assert diff_video < 0.0002
    assert diff_accel < 0.03

    assert np.any(result_video != 0), "Result video is entirely zeros"
    assert np.any(phase_delta_dir_sum != 0), "Acceleration output is entirely zeros"

if __name__ == "__main__":
    test_accel_based_regression()
    print("Test passed successfully.")
