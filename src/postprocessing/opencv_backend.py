"""
src/postprocessing/opencv_backend.py
=====================================
CPU-based postprocessing using OpenCV and NumPy.

Takes the model's per-pixel class probabilities and produces
a background-blurred version of the original frame where the
target object (cat) stays sharp.

Data path:
    GPU tensor (probabilities from model)
        → .cpu() move to CPU RAM
        → argmax → binary mask (h, w)
        → resize mask to full resolution (OpenCV/CPU)
        → gaussian blur mask edges (OpenCV/CPU)
        → gaussian blur full frame (OpenCV/CPU)
        → blend sharp + blurred using mask (NumPy/CPU)
        → list of (H, W, 3) uint8 RGB arrays → encoder

Everything here is CPU. The GPU→CPU transfer happens once
at the start (.cpu() call on probabilities tensor).
This is what cvcuda_backend.py eliminates entirely.
"""

from typing import List

import cv2
import numpy as np
import torch

from src.postprocessing.base import BasePostprocessor


class OpenCVPostprocessor(BasePostprocessor):
    """
    CPU postprocessor. Implements BasePostprocessor contract.

    Usage:
        post = OpenCVPostprocessor(
            output_layout=encoder.input_layout,  # "HWC"
            gpu_output=encoder.gpu_input,         # False
        )
        frames = post(probabilities, orig_tensor, resized_tensor, class_index)
    """

    # Blur kernel sizes — must be odd numbers
    # Larger = stronger blur effect
    MASK_BLUR_KERNEL  = (21, 21)   # softens mask edges so blend looks natural
    FRAME_BLUR_KERNEL = (51, 51)   # background blur strength

    @property
    def name(self) -> str:
        return "opencv (CPU)"

    def __call__(
        self,
        probabilities:  torch.Tensor,   # (B, num_classes, h, w) on GPU
        orig_tensor:    torch.Tensor,   # (B, C, H, W) float32 on CPU
        resized_tensor: torch.Tensor,   # (B, C, h, w) float32 on CPU
        class_index:    int,
    ) -> List[np.ndarray]:
        """
        Returns list of (H, W, 3) uint8 RGB arrays — one per frame.
        """

        # ------------------------------------------------------------------
        # Move model output from GPU → CPU
        # This is the transfer we are benchmarking against cvcuda_backend
        # which keeps everything on GPU and never calls .cpu()
        # ------------------------------------------------------------------
        probs = probabilities.detach().cpu()   # (B, num_classes, h, w)
        orig  = orig_tensor.cpu()              # (B, C, H, W)

        B = probs.shape[0]
        result_frames = []

        for i in range(B):

            # --------------------------------------------------------------
            # Step 1: Extract binary mask from probabilities
            #
            # probs[i] is (num_classes, h, w)
            # argmax along dim=0 gives (h, w) where each pixel value
            # is the predicted class index
            # We then create a binary mask: 1.0 where class == target
            # --------------------------------------------------------------
            pred_class = probs[i].argmax(dim=0).numpy()              # (h, w)
            mask_small = (pred_class == class_index).astype(np.float32)  # 0 or 1

            # --------------------------------------------------------------
            # Step 2: Get original frame as uint8 numpy (H, W, 3)
            #
            # orig[i] is (C, H, W) float32 [0, 1]
            # permute → (H, W, C), scale → uint8
            # --------------------------------------------------------------
            orig_np = orig[i].permute(1, 2, 0).numpy()               # (H, W, 3)
            orig_np = (orig_np * 255).clip(0, 255).astype(np.uint8)
            H, W    = orig_np.shape[:2]

            # --------------------------------------------------------------
            # Step 3: Resize mask from inference size → full resolution
            #
            # INTER_LINEAR gives smooth edges during upscaling.
            # Result is float32 in [0, 1] range (not just 0 or 1 anymore
            # because interpolation blends between them — this is intentional,
            # it creates a soft edge for the blend in Step 5)
            # --------------------------------------------------------------
            mask_full = cv2.resize(
                mask_small, (W, H),
                interpolation=cv2.INTER_LINEAR
            )                                                          # (H, W)

            # --------------------------------------------------------------
            # Step 4: Soften mask edges with Gaussian blur
            #
            # Without this the object boundary looks like a hard cutout.
            # After blur, pixels near the edge get partial values (0.3, 0.7)
            # which blend naturally in Step 5.
            # --------------------------------------------------------------
            mask_full = cv2.GaussianBlur(
                mask_full, self.MASK_BLUR_KERNEL, sigmaX=0
            )
            mask_full = mask_full[:, :, np.newaxis]                   # (H, W, 1)

            # --------------------------------------------------------------
            # Step 5: Blur the entire original frame
            # --------------------------------------------------------------
            blurred = cv2.GaussianBlur(
                orig_np, self.FRAME_BLUR_KERNEL, sigmaX=0
            )

            # --------------------------------------------------------------
            # Step 6: Blend sharp + blurred using soft mask
            #
            # mask=1.0 → take sharp original (object pixels)
            # mask=0.0 → take blurred version (background pixels)
            # mask=0.5 → 50/50 blend (edge pixels — looks natural)
            #
            # Formula: result = sharp * mask + blurred * (1 - mask)
            # --------------------------------------------------------------
            orig_f   = orig_np.astype(np.float32)
            blurred_f = blurred.astype(np.float32)
            blended  = orig_f * mask_full + blurred_f * (1.0 - mask_full)
            blended  = blended.clip(0, 255).astype(np.uint8)

            result_frames.append(blended)

        return result_frames