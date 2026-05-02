"""
src/postprocessing/cvcuda_backend.py
=====================================
GPU-based postprocessing using CV-CUDA.

Performs the exact same operations as opencv_backend.py but
everything stays on the GPU. No tensor ever moves to CPU RAM.

Compare to opencv_backend.py data path:
  OPENCV:  GPU (probs) → CPU → argmax → resize mask → blur → blend → CPU list
  CVCUDA:  GPU (probs) → argmax → resize mask → blur → blend → GPU → CPU list

The only CPU operation remaining is the final conversion to numpy
for the encoder — which is unavoidable since our BatchEncoder is
CPU-based (OpenCV VideoWriter). When you add NVENC later, even
this last step goes away.

Operations performed (all GPU):
  1. argmax on probability tensor        — PyTorch GPU
  2. cast mask to float                  — PyTorch GPU  
  3. resize mask to full resolution      — CV-CUDA GPU kernel
  4. gaussian blur mask edges            — CV-CUDA GPU kernel
  5. gaussian blur full frame            — CV-CUDA GPU kernel
  6. blend sharp + blurred using mask    — PyTorch GPU
  7. convert to uint8                    — PyTorch GPU
  8. move to CPU for encoder             — one final transfer
"""

from typing import List

import cv2
import numpy as np
import torch
import cvcuda

from src.postprocessing.base import BasePostprocessor


class CVCUDAPostprocessor(BasePostprocessor):
    """
    GPU postprocessor using CV-CUDA. Implements BasePostprocessor contract.

    Interface is identical to OpenCVPostprocessor — same inputs, same
    outputs. The pipeline runner cannot tell the difference.
    """

    # Blur kernel sizes — must be odd, match opencv_backend for fair comparison
    MASK_BLUR_KERNEL  = (21, 21)
    FRAME_BLUR_KERNEL = (51, 51)

    @property
    def name(self) -> str:
        return "cvcuda (GPU)"

    def __call__(
        self,
        probabilities:  torch.Tensor,   # (B, num_classes, h, w) on GPU
        orig_tensor:    torch.Tensor,   # (B, C, H, W) float32 on GPU
        resized_tensor: torch.Tensor,   # (B, C, h, w) float32 on GPU
        class_index:    int,
    ) -> List[np.ndarray]:
        """
        Returns list of (H, W, 3) uint8 RGB numpy arrays — one per frame.
        All heavy work happens on GPU. Only the final .cpu().numpy()
        touches CPU — required by our OpenCV encoder.
        """

        B, C, H, W = orig_tensor.shape
        _, _, h, w = probabilities.shape

        # ------------------------------------------------------------------
        # Step 1: Extract binary mask from probabilities — PyTorch GPU
        #
        # probabilities is (B, num_classes, h, w) on GPU
        # argmax along dim=1 → (B, h, w) — predicted class per pixel
        # Compare to class_index → (B, h, w) bool mask
        # Cast to float32 so CV-CUDA can process it
        # ------------------------------------------------------------------
        pred_class = probabilities.argmax(dim=1)                # (B, h, w) int64 GPU
        mask = (pred_class == class_index).float()              # (B, h, w) float32 GPU

        # ------------------------------------------------------------------
        # Step 2: Add channel dim and prepare for CV-CUDA
        #
        # CV-CUDA expects NHWC layout.
        # mask is (B, h, w) → unsqueeze → (B, h, w, 1) → NHWC float32
        # ------------------------------------------------------------------
        mask_nhwc = mask.unsqueeze(-1).contiguous()             # (B, h, w, 1)
        cvcuda_mask = cvcuda.as_tensor(mask_nhwc, "NHWC")

        # ------------------------------------------------------------------
        # Step 3: Resize mask from inference size → full resolution
        #
        # Goes from (B, h, w, 1) → (B, H, W, 1)
        # INTER_LINEAR gives smooth interpolated values at edges (0.0-1.0)
        # which creates natural-looking blend boundaries
        # ------------------------------------------------------------------
        cvcuda_mask_full = cvcuda.resize(
            cvcuda_mask,
            (B, H, W, 1),
            cvcuda.Interp.LINEAR,
        )

        # ------------------------------------------------------------------
        # Step 4: Gaussian blur the mask edges — CV-CUDA GPU kernel
        #
        # Softens the hard 0/1 boundary so the blend looks natural.
        # kernel_size must be (width, height) tuple of odd ints.
        # sigma=0 tells OpenCV/CV-CUDA to compute sigma from kernel size.
        # ------------------------------------------------------------------
        cvcuda_mask_blurred = cvcuda.gaussian(
            cvcuda_mask_full,
            kernel_size=self.MASK_BLUR_KERNEL,
            sigma=(0.0, 0.0),
            border=cvcuda.Border.REFLECT,
        )

        # ------------------------------------------------------------------
        # Step 5: Prepare original frame for CV-CUDA blurring
        #
        # orig_tensor is (B, C, H, W) float32 [0,1] on GPU — NCHW
        # CV-CUDA needs NHWC — permute and make contiguous
        # Scale [0,1] → [0,255] uint8 for gaussian blur kernel
        # (CV-CUDA gaussian works on uint8 efficiently)
        # ------------------------------------------------------------------
        orig_nhwc = orig_tensor.permute(0, 2, 3, 1).contiguous()   # (B,H,W,3) float
        orig_uint8 = (orig_nhwc * 255).clamp(0, 255).byte()        # (B,H,W,3) uint8
        cvcuda_orig = cvcuda.as_tensor(orig_uint8, "NHWC")

        # ------------------------------------------------------------------
        # Step 6: Gaussian blur the full frame — CV-CUDA GPU kernel
        #
        # This blurs the entire frame. We then use the mask to decide
        # which pixels keep the sharp original vs the blurred version.
        # ------------------------------------------------------------------
        cvcuda_blurred = cvcuda.gaussian(
            cvcuda_orig,
            kernel_size=self.FRAME_BLUR_KERNEL,
            sigma=(0.0, 0.0),
            border=cvcuda.Border.REFLECT,
        )

        # ------------------------------------------------------------------
        # Step 7: Convert CV-CUDA results back to PyTorch GPU tensors
        #
        # Use torch.as_tensor(..., device='cuda') pattern confirmed working
        # with cvcuda 0.16.0 ExternalBuffer
        # ------------------------------------------------------------------
        mask_full_torch = torch.as_tensor(
            cvcuda_mask_blurred.cuda(), device="cuda"
        ).float()                                               # (B, H, W, 1)

        orig_torch = torch.as_tensor(
            cvcuda_orig.cuda(), device="cuda"
        ).float()                                               # (B, H, W, 3)

        blurred_torch = torch.as_tensor(
            cvcuda_blurred.cuda(), device="cuda"
        ).float()                                               # (B, H, W, 3)

        # ------------------------------------------------------------------
        # Step 8: Blend sharp + blurred using soft mask — PyTorch GPU
        #
        # mask=1.0 → sharp original (object)
        # mask=0.0 → blurred background
        # mask=0.x → smooth blend at edges
        # ------------------------------------------------------------------
        blended = (
            orig_torch * mask_full_torch
            + blurred_torch * (1.0 - mask_full_torch)
        ).clamp(0, 255).byte()                                  # (B, H, W, 3) uint8 GPU

        # ------------------------------------------------------------------
        # Step 9: Move to CPU for encoder — unavoidable with CPU encoder
        #
        # This is ONE transfer per batch — much better than opencv_backend
        # which transferred probabilities, orig, and intermediate results.
        # With NVENC encoder this .cpu() call goes away entirely.
        # ------------------------------------------------------------------
        blended_cpu = blended.cpu().numpy()                     # (B, H, W, 3) uint8 CPU

        # Split batch into list of individual frames
        return [blended_cpu[i] for i in range(B)]