"""Phase-based magnification processing."""

import time
from typing import Callable, Optional, Tuple
import numpy as np
import cv2
import cupy as cp
from pyramid_utils import build_level, build_level_batch, recon_level_batch

ON_TICK_DUR = 0.250 # seconds

class PhaseBased:

    def __init__(
        self,
        sigma,
        video_fs,
        transfer_function,
        phase_mag,
        attenuate,
        ref_idx,
        batch_size,
        device,
        eps=1e-6,
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

        self.gauss_kernel = self.get_gauss_kernel()

    def get_gauss_kernel(self):
        """Obtains Gaussian Kernel for Amplitude weighted Blurring
        Inputs: None
        Outputs:
            gauss_kernel
        """
        ksize = int(max(3, cp.ceil(4 * self.sigma) - 1))
        if (ksize % 2) != 1:
            ksize += 1

        # get Gaussian Blur Kernel for reference only
        gk = cv2.getGaussianKernel(ksize=ksize, sigma=self.sigma)
        gauss_kernel = cp.array(gk @ gk.T, dtype=cp.float32).reshape(1, 1, ksize, ksize)

        return gauss_kernel

    def process_single_channel(self, frames_tensor, filters_tensor, video_dft, filter_dir_tensor, csp, ontick: Optional[Callable[[float], None]] = None) -> Tuple[cp.ndarray, cp.ndarray]:
        """Applies Phase Based Processing in the Frequency Domain
        for single channel frames
        Inputs:
            frames_tensor - tesnor of frames to process
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
        phase_deltas = cp.zeros((self.batch_size, num_frames, h, w), dtype=cp.complex64) # phase deltas for each batch
        phase_deltas_sum = cp.zeros((num_frames, h, w, 2), dtype=cp.float32) # sum of phase deltas for all batches

        tic = time.perf_counter()

        for level in range(1, num_filters - 1, self.batch_size):
            if tic + ON_TICK_DUR < time.perf_counter() and ontick:
                ontick(level / num_filters * 0.90)
                tic = time.perf_counter()

            # get batch indices
            idx1 = level
            idx2 = level + self.batch_size

            # get current filter batch
            filter_batch = filters_tensor[idx1:idx2]
            motion_dirs_batch = filter_dir_tensor[idx1:idx2]

            ## get reference frame pyramid and phase (DC)
            ref_pyr = build_level_batch(video_dft[self.ref_idx, :, :].reshape(1, h, w), filter_batch)
            ref_phase = cp.angle(ref_pyr)

            ## Get Phase Deltas for each frame
            for vid_idx in range(num_frames):
                curr_pyr = build_level_batch(video_dft[vid_idx, :, :].reshape(1, h, w), filter_batch)

                # unwrapped phase delta
                _delta = cp.angle(curr_pyr) - ref_phase

                # get phase delta wrapped to [-pi, pi]
                phase_deltas[:, vid_idx, :, :] = ((cp.pi + _delta) % (2 * cp.pi)) - cp.pi

            ## Temporally Filter the phase deltas
            # Filter in Frequency Domain across frames and convert back to phase space
            phase_deltas = cp.fft.ifft(self.transfer_function * cp.fft.fft(phase_deltas, axis=1), axis=1).real
            # phase_deltas_sum += phase_deltas.sum(dim=0).real
            # mult by motion dirs and add to sum
            phase_deltas_sum += cp.einsum('bfyx,bd->fyxd', phase_deltas.real, motion_dirs_batch)

            ## Apply Motion Magnifications
            for vid_idx in range(num_frames):

                vid_dft = video_dft[vid_idx, :, :].reshape(1, h, w)
                curr_pyr = build_level_batch(vid_dft, filter_batch)
                delta = phase_deltas[:, vid_idx, :, :]

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
                ## changes except the banpdass filtered phases
                if self.attenuate:
                    curr_pyr = cp.abs(curr_pyr) * (ref_pyr / cp.abs(ref_pyr))

                ## apply modified phase to current level pyramid decomposition
                # if modified_phase = 0, then no change!
                curr_pyr *= cp.exp(1.0j * modifed_phase)

                ## accumulate reconstruced levels
                recon_dft[vid_idx, :, :] += cp.sum(recon_level_batch(curr_pyr, filter_batch), axis=0)


        # getting average over all filters
        # phase_deltas_sum /= num_filters

        ## add unchanged Low Pass Component for contrast
        # adding hipass seems to cause bad artifacts and leaving
        # it out doesn't seem to impact the overall quality

        # hipass = filters_tensor[0]
        lopass = filters_tensor[-1]

        ## add back lo and hi pass components
        for vid_idx in range(num_frames):
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

        ## Get Inverse DFT and remove from CUDA if applicable
        result = cp.fft.ifft2(cp.fft.ifftshift(recon_dft, axes=(1, 2)), axes=(1, 2))
        result_video = result.real

        return result_video, phase_deltas_sum
