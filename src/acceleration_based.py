import math
import time
from typing import Callable, Optional, Tuple
import numpy as np
import cv2
import cupy as cp
from pyramid_utils import build_level, build_level_batch, recon_level_batch
from steerable_pyramid import SteerablePyramid

ON_TICK_DUR = 0.250 # seconds

class AccelBased:

    def __init__(
        self,
        sigma,
        video_fs,
        transfer_function,
        phase_mag,
        attenuate,
        ref_idx,
        batch_size,
        eps=1e-6,
        phase_bound_perc: float = 0.1,
    ):
        """
        sigma - std dev of Amplitude Weighted Phase Blurring
                (use 0 for no blurring)
        transfer_function - Frequency Domain Bandpass Filter
                            Transfer Function (array)
        phase_mag - Phase Magnification/Amplification factor
        attenuate - determines whether to attenuate other frequencies
        ref_idx - index of reference frame to compare local phase
                  changes to (DC frame)
        batch_size - batch size for parallelization
        eps - offset to avoid division by zero
        """
        self.sigma = sigma
        self.video_fs = video_fs
        self.transfer_function = transfer_function
        self.phase_mag = phase_mag
        self.attenuate = attenuate
        self.ref_idx = ref_idx
        self.batch_size = batch_size
        self.eps = eps
        self.phase_bound_perc = phase_bound_perc
        if self.phase_bound_perc < 0.0 or self.phase_bound_perc > 1.0:
            raise ValueError(f"Phase bound percentage must be between 0.0 and 1.0, got {self.phase_bound_perc}")

        self.gauss_kernel = self.get_gauss_kernel()

    def get_gauss_kernel(self):
        """Obtains Gaussian Kernel for Amplitude weighted Blurring
        Inputs: None
        Outputs:
            gauss_kernel
        """
        # ensure ksize is odd or the filtering will take too long
        # see warning in: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        ksize = int(max(3, cp.ceil(4 * self.sigma) - 1))
        if (ksize % 2) != 1:
            ksize += 1

        # get Gaussian Blur Kernel for reference only
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        gauss_kernel = cp.array(gk @ gk.T, dtype=cp.float32).reshape(1, 1, ksize, ksize)

        return gauss_kernel

    def shift_correction(self, finer_phase_diff, coarser_phase_diff, pyr_scale_factor, perc_of_limitation, num_levels, current_level): # Phase-Based Frame Interpolation for Video Sec. 3.2 (Eq. 10 + 11)
        if coarser_phase_diff is not None:
            phi = cp.arctan2(cp.sin(finer_phase_diff - pyr_scale_factor * coarser_phase_diff), cp.cos(finer_phase_diff - pyr_scale_factor * coarser_phase_diff))
            confidence_based = cp.abs(phi) > cp.pi / 2.0
            finer_phase_diff = cp.where(confidence_based, pyr_scale_factor * coarser_phase_diff, finer_phase_diff)

        if perc_of_limitation > 0.0:
            phase_limit = perc_of_limitation * cp.pi * pyr_scale_factor ** (num_levels - current_level) # Eq. 11
            to_bound = cp.abs(finer_phase_diff) > phase_limit
            finer_phase_diff = cp.where(to_bound, pyr_scale_factor * coarser_phase_diff if coarser_phase_diff is not None else 0, finer_phase_diff)

        return finer_phase_diff



    def phase_difference(self, curr_phase, prev_phase): # Phase-Based Frame Interpolation for Video Eq. 8
        return cp.arctan2(cp.sin(prev_phase - curr_phase), cp.cos(prev_phase - curr_phase))

    def diff_of_gauss(self, frames_phase_buffer, k): # Video Acceleration Magnification 3.3. Temporal Acceleration Filtering
        # frames_phase_buffer: [T, B, H, W]
        # k: [K, 1, 1, 1] (1D DoG filter along time, pre-shaped)

        accel = cp.sum(frames_phase_buffer * k, axis=0)   # [B,H,W]
        return accel


    def phase_unwrap(self, shifted, original, cutoff=cp.pi):
        """
        shifted:   array_like of shape (B,H,W), shift-corrected Δphase
        original:  same shape, raw Δphase
        returns:   same shape, unwrapped Δphase
        """
        dp    = original - shifted # Incremental phase variations
        dps   = ( (dp + cp.pi) % (2*cp.pi) ) - cp.pi # Equivalent phase variations in [-pi,pi)

        mask = (dps == -cp.pi) & (dp > 0) # Preserve variation sign for pi vs. -pi
        dps = cp.where(mask, cp.pi, dps)

        dp_corr = dps - dp # Incremental phase corrections

        dp_corr = cp.where(cp.abs(dp) < cutoff, 0, dp_corr) # Ignore correction when incr. variation is < CUTOFF

        return original + dp_corr # Integrate corrections and add to P to produce smoothed phase values


    def process_single_channel(self, frames_tensor, filters_tensor, video_dft, filter_dir_tensor, csp: SteerablePyramid, ontick: Optional[Callable[[float], None]] = None) -> Tuple[cp.ndarray, cp.ndarray]:
        """Applies Phase Based Processing in the Frequency Domain
        for single channel frames
        Inputs:
            frames_tensor - tensor of frames to process
            filters_tensor - tensor of Complex Steerable Filter components
            video_dft - tensor of DFT video frames
            filter_dir_tensor - unit vectors of orientation for each filter
            csp - Complex Steerable Pyramid object
            ontick - optional callback function for progress updates
        Outputs:
            result_video - tensor of reconstructed video frames with amplified motion
            phase_deltas_avg - tensor of average phase deltas across all filters
        """
        num_frames, _, _ = frames_tensor.shape
        num_filters, h, w = filters_tensor.shape

        # print("frames_tensor shape:", frames_tensor.shape)
        # print("filters_tensor shape:", filters_tensor.shape)
        # print("filter_dir_tensor shape:", filter_dir_tensor.shape)

        # allocate tensors for processing
        recon_dft = cp.zeros((num_frames, h, w), dtype=cp.complex64)
        phase_accels = cp.zeros((self.batch_size, num_frames, h, w), dtype=cp.float32) # phase acceleration for each batch
        phase_accels_sum = cp.zeros((num_frames, h, w, 2), dtype=cp.float32) # sum of phase accelerations for all batches

        tic = time.perf_counter()

        num_filters_per_level = csp.orientations
        prev_level_phases_diff = cp.zeros((num_filters_per_level, num_frames, h, w), dtype=cp.float32)  # shape: [B, F, H, W]
        assert self.batch_size <= num_filters_per_level, f"Batch size {self.batch_size} exceeds number of filters per level {num_filters_per_level}, requires changing level_idx calculations"
        assert (num_filters - 2) % self.batch_size == 0, f"Number of filters {num_filters} minus 2 (for lo and hi pass) must be divisible by batch size {self.batch_size}"

        # DoG filter params
        motion_freq = 4 # hz, estimated motion frequency
        time_interval = 1/4 * 1 / motion_freq  # in sec. one quarter of sine wave.
        frame_interval = math.ceil(self.video_fs * time_interval)  # in frames
        window_size = 2 * frame_interval  # in frames
        # Size of DOG kernel. In DOG kernel, we want the peak and two bottom value match our first method. So the length of DOG is set twice as the original window.
        norder = (window_size * 2)

        # dog_kernel = make_temporal_kernel(self.video_fs, time_interval, mode='DOG', sigma1_ratio=0.3, sigma2_ratio=5.0, device=self.device)  # [1, 1, K]
        dog_kernel = make_temporal_kernel(self.video_fs, time_interval, mode='DOG', sigma1_ratio=0.5, sigma2_ratio=2.0)  # [1, 1, K]
        dog_kernel = dog_kernel.reshape(-1, 1, 1, 1) # [K, 1, 1, 1]

        for filt_i_lo in range(1, num_filters - 1, self.batch_size):
            if tic + ON_TICK_DUR < time.perf_counter() and ontick:
                ontick(filt_i_lo / num_filters * 0.90)
                tic = time.perf_counter()

            # get batch indices
            idx1 = filt_i_lo
            idx2 = filt_i_lo + self.batch_size

            level_idx1 = (filt_i_lo - 1) % num_filters_per_level
            level_idx2 = level_idx1 + self.batch_size
            current_level = (filt_i_lo - 1) // num_filters_per_level + 1

            # get current filter batch
            filter_batch = filters_tensor[idx1:idx2]       # shape: [B, H, W]
            if idx1 - self.batch_size < 0:
                prev_filter_batch = None
            else:
                prev_filter_batch = filters_tensor[idx1 - self.batch_size:idx2 - self.batch_size]
            motion_dirs_batch = filter_dir_tensor[idx1:idx2]

            ## get reference frame pyramid and phase (DC)
            ref_pyr = build_level_batch(video_dft[self.ref_idx, :, :].reshape(1, h, w), filter_batch)  # shape: [B, H, W], complex
            ref_phase = cp.angle(ref_pyr)  # shape: [B, H, W]

            ## Get Phase Deltas for each frame
            # prev_frames_phase = torch.zeros((norder + 1, self.batch_size, h, w), dtype=torch.float32, device=self.device)  # shape: [norder + 1, B, H, W]
            frames_phase_buffer = cp.tile(ref_phase, (norder + 1, 1, 1, 1)) # shape: [norder + 1, B, H, W] filled with phase from reference frame

            # symmetric padding instead of repeating
            # frames_phase_buffer = torch.zeros((norder + 1, self.batch_size, h, w), dtype=torch.float32, device=self.device)  # shape: [norder + 1, B, H, W]
            # for i in range(norder + 1):
            #     frame_idx = min(num_frames - 1, norder - i)
            #     frame_idx = min(frame_idx, 2) # artificially repeat after X frames
            #     frames_phase_buffer[i] = torch.angle(build_level_batch(video_dft[frame_idx, :, :].unsqueeze(0), filter_batch))  # shape: [B, H, W]

            for vid_idx in range(num_frames):
                curr_pyr = build_level_batch(video_dft[vid_idx, :, :].reshape(1, h, w), filter_batch)  # [B, H, W], complex

                prev_level_phase_orig = prev_level_phases_diff[level_idx1:level_idx2, vid_idx, :, :]
                # add curr_phase to the end of prev_frames_phase and drop the first frame
                frames_phase_buffer = cp.roll(frames_phase_buffer, shift=-1, axis=0) # switching this roll to a ring buffer does not improve performance at all. most rolls are from the ffts (fftshift)
                frames_phase_buffer[-1] = cp.angle(curr_pyr)

                fac = 1.5
                direct_phase_diff = frames_phase_buffer[-1] - frames_phase_buffer[-2]  # shape: [B, H, W]
                mask_pos = direct_phase_diff > fac * cp.pi
                mask_neg = direct_phase_diff < -fac * cp.pi
                frames_phase_buffer[-1] += cp.where(mask_pos, -2 * cp.pi, 0) + cp.where(mask_neg, 2 * cp.pi, 0)

                phase_conv = self.diff_of_gauss(frames_phase_buffer, dog_kernel)  # shape: [norder + 1, B, H, W]

                prev_level_phase = self.shift_correction(finer_phase_diff=prev_level_phase_orig, coarser_phase_diff=phase_conv, pyr_scale_factor=csp.pyr_scale_factor, perc_of_limitation=self.phase_bound_perc, num_levels=csp.num_filts, current_level=current_level)

                unwrapped_phase_diff = self.phase_unwrap(shifted=prev_level_phase, original=prev_level_phase_orig, cutoff=2) # 2 in motionamp.m from Video Acceleration Magnification
                phase_accels[:, vid_idx, :, :] = unwrapped_phase_diff

                prev_level_phases_diff[level_idx1:level_idx2, vid_idx, :, :] = phase_conv


            # mult by motion dirs and add to sum
            phase_accels_sum += cp.einsum('bfyx,bd->fyxd', phase_accels.real, motion_dirs_batch)

            ## Apply Motion Magnifications
            def recon_helper(filter_batch):
                for vid_idx in range(num_frames):

                    vid_dft = video_dft[vid_idx, :, :].reshape(1, h, w)
                    curr_pyr = build_level_batch(vid_dft, filter_batch)
                    delta = phase_accels[:, vid_idx, :, :]

                    ## Perform Amplitude Weighted Blurring
                    if self.sigma != 0:
                        amplitude_weight = cp.abs(curr_pyr) + self.eps

                        # Torch Functional Approach for convolutional filtering
                        weight = cp.convolve(amplitude_weight, self.gauss_kernel, mode='same')
                        delta  = cp.convolve(amplitude_weight * delta, self.gauss_kernel, mode='same') / weight

                    ## Modify phase variation
                    modifed_phase = delta * self.phase_mag

                    ## Attenuate other frequencies by scaling current magnitude
                    ## by normalized reference phase. This removed all phase
                    ## changes except the bandpass filtered phases
                    if self.attenuate:
                        curr_pyr = cp.abs(curr_pyr) * (ref_pyr / cp.abs(ref_pyr))

                    ## apply modified phase to current level pyramid decomposition
                    # if modified_phase = 0, then no change!
                    curr_pyr = curr_pyr * cp.exp(1.0j * modifed_phase)

                    ## accumulate reconstructed levels
                    recon_dft[vid_idx, :, :] += cp.sum(recon_level_batch(curr_pyr, filter_batch), axis=0)

            if prev_filter_batch is not None:
                recon_helper(prev_filter_batch)
            if idx1 > num_filters - 1 - self.batch_size: #last batch
                for vid_idx in range(num_frames):
                    prev_level_phase_orig = prev_level_phases_diff[level_idx1:level_idx2, vid_idx, :, :]
                    prev_level_phase = self.shift_correction(finer_phase_diff=prev_level_phase_orig, coarser_phase_diff=None, pyr_scale_factor=csp.pyr_scale_factor, perc_of_limitation=self.phase_bound_perc, num_levels=csp.num_filts, current_level=current_level)

                    unwrapped_phase_diff = self.phase_unwrap(shifted=prev_level_phase, original=prev_level_phase_orig, cutoff=2) # 2 in motionamp.m from Video Acceleration Magnification
                    phase_accels[:, vid_idx, :, :] = unwrapped_phase_diff
                recon_helper(filter_batch)

        # getting average over all filters
        # phase_deltas_sum /= num_filters

        ## add unchanged Low Pass Component for contrast
        # adding hipass seems to cause bad artifacts and leaving
        # it out doesn't seem to impact the overall quality

        # hipass = filters_tensor[0]
        lopass = filters_tensor[-1]

        ## add back lo and hi pass components
        for vid_idx in range(num_frames):
            # spatial_sal = compute_spatial_saliency(frames_tensor[vid_idx, :, :].cpu().numpy())
            # phase_accels_sum[vid_idx, :, :, 0] *= torch.tensor(spatial_sal, device=self.device)

            if tic + ON_TICK_DUR < time.perf_counter() and ontick:
                ontick(vid_idx / num_frames * 0.10 + 0.90)
                tic = time.perf_counter()

            # Get Pyramid Decompositions for Hi and Lo Pass Filters
            # curr_pyr_hi = build_level(video_dft[vid_idx, :, :], hipass)
            curr_pyr_lo = build_level(video_dft[vid_idx, :, :], lopass)

            # dft_hi = torch.fft.fftshift(torch.fft.fft2(curr_pyr_hi))
            dft_lo = cp.fft.fftshift(cp.fft.fft2(curr_pyr_lo))

            # accumulate reconstructed hi and lo components
            # recon_dft[vid_idx, :, :] += dft_hi*hipass
            recon_dft[vid_idx, :, :] += dft_lo * lopass

        ## Get Inverse DFT
        result = cp.fft.ifft2(cp.fft.ifftshift(recon_dft, axes=(1, 2)), axes=(1, 2))
        result_video = result.real

        return result_video, phase_accels_sum


def make_temporal_kernel(frame_rate, time_interval, mode='DOG', sigma1_ratio=0.5, sigma2_ratio=2.0) -> cp.ndarray:
    """
    Create a temporal kernel for video processing.

    Args:
        frame_rate (float): Frame rate of the video.
        time_interval (float): Time interval in seconds.
        mode (str): 'DOG' (Difference of Gaussians) or 'INT' (Interpolation kernel).

    Returns:
        torch.Tensor: 1D temporal kernel of shape [1, 1, K]
    """
    frame_interval = int(math.ceil(frame_rate * time_interval))
    window_size = 2 * frame_interval
    signal_len = 2 * window_size
    sigma = frame_interval / 2
    x = cp.linspace(-signal_len / 2, signal_len / 2, num=signal_len + 1)

    if mode.upper() == 'DOG':
        sigma1 = sigma * sigma1_ratio
        sigma2 = sigma * sigma2_ratio

        gauss1 = cp.exp(-cp.power(x, 2) / (2 * sigma1 ** 2))
        gauss1 /= gauss1.sum()

        gauss2 = cp.exp(-cp.power(x, 2) / (2 * sigma2 ** 2))
        gauss2 /= gauss2.sum()

        dog = gauss1 - gauss2
        kernel = dog / cp.abs(dog).sum()

    elif mode.upper() == 'INT':
        kernel = cp.zeros_like(x)
        kernel[frame_interval] = 0.5
        kernel[2 * frame_interval] = -1.0
        kernel[3 * frame_interval] = 0.5
        kernel = -kernel / cp.abs(kernel).sum()

    else:
        raise ValueError("Unsupported mode. Use 'DOG' or 'INT'.")

    return kernel.reshape(1, 1, -1)
