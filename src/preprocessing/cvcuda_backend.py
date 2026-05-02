"""
src/preprocessing/cvcuda_backend.py
=====================================
GPU-based preprocessing using CV-CUDA.

Performs the exact same operations as opencv_backend.py but
every operation stays on the GPU. No data ever moves to CPU RAM.

Data path (compare to opencv_backend.py):
    CPU RAM (raw frames from decoder)
        → upload to GPU once (torch.as_tensor + .cuda())
        → resize    (CV-CUDA GPU kernel)
        → normalize (CV-CUDA GPU kernel)
        → three tensors returned — all on GPU
        → model reads directly from GPU memory
        → postprocessing reads directly from GPU memory

The only CPU→GPU transfer is the initial upload of raw frames.
Everything after that is GPU-to-GPU.

CV-CUDA tensor format:
    CV-CUDA uses NHWC layout internally (batch, height, width, channels)
    PyTorch models expect NCHW (batch, channels, height, width)
    We convert at the end with .permute(0, 3, 1, 2)
    This permute is a zero-copy view — no data is moved.
"""

from typing import List, Tuple

import numpy as np
import torch
import cvcuda

from src.preprocessing.base import BasePreprocessor


class CVCUDAPreprocessor(BasePreprocessor):
    """
    GPU preprocessor using CV-CUDA. Implements BasePreprocessor contract.

    The interface is identical to OpenCVPreprocessor — same inputs,
    same outputs, same three tensors. The pipeline runner cannot tell
    the difference. Only the location of the data changes (GPU vs CPU).

    Usage:
        pre = CVCUDAPreprocessor(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        orig, resized, normalized = pre(frames, out_size=(224, 224))
        # All three tensors are on GPU
    """

    # def __init__(
    #     self,
    #     mean: List[float] = (0.485, 0.456, 0.406),
    #     std:  List[float] = (0.229, 0.224, 0.225),
    # ):
    #     # Store mean and std as GPU tensors shaped (1, 1, 1, 3)
    #     # NHWC layout — broadcasts over (B, H, W, C) cvcuda tensors
    #     self._mean = torch.tensor(mean, dtype=torch.float32).cuda()
    #     self._std  = torch.tensor(std,  dtype=torch.float32).cuda()

    def __init__(
    self,
    mean: List[float] = (0.485, 0.456, 0.406),
    std:  List[float] = (0.229, 0.224, 0.225),
):
        # Store as plain lists — move to GPU lazily on first __call__
        # because CUDA context may not be initialized at __init__ time
        self._mean_list = list(mean)
        self._std_list  = list(std)
        self._mean: torch.Tensor | None = None
        self._std:  torch.Tensor | None = None
    
    def _ensure_on_gpu(self, device: torch.device):
        """Move mean/std to GPU on first call — safe because CUDA is ready by then."""
        if self._mean is None:
            self._mean = torch.tensor(
                self._mean_list, dtype=torch.float32, device=device
            )
            self._std = torch.tensor(
                self._std_list, dtype=torch.float32, device=device
            )

    @property
    def name(self) -> str:
        return "cvcuda (GPU)"

    # def __call__(
    #     self,
    #     frames:   List[np.ndarray],
    #     out_size: Tuple[int, int],
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Parameters
    #     ----------
    #     frames   : list of (H, W, 3) uint8 RGB numpy arrays from decoder
    #     out_size : (width, height) to resize to

    #     Returns
    #     -------
    #     orig_tensor       : (B, C, H, W) float32 GPU tensor
    #     resized_tensor    : (B, C, h, w) float32 GPU tensor
    #     normalized_tensor : (B, C, h, w) float32 GPU tensor

    #     All tensors on CUDA — no CPU round trip after initial upload.
    #     """
    #     out_w, out_h = out_size
    #     B = len(frames)

    #     # ------------------------------------------------------------------
    #     # Step 1: Upload raw frames from CPU → GPU
    #     #
    #     # This is the ONE and ONLY CPU→GPU transfer in this backend.
    #     # We stack all frames into a single contiguous numpy array first
    #     # (one transfer is faster than B separate transfers).
    #     #
    #     # np.stack → (B, H, W, 3) uint8
    #     # torch.as_tensor shares memory with numpy (zero copy on CPU side)
    #     # .cuda() does the actual DMA transfer to GPU
    #     # ------------------------------------------------------------------
    #     batch_np  = np.stack(frames, axis=0)                    # (B, H, W, 3)
    #     batch_gpu = torch.as_tensor(batch_np).cuda()            # (B, H, W, 3) uint8 on GPU

    #     # ------------------------------------------------------------------
    #     # Step 2: Wrap GPU tensor as a CV-CUDA tensor
    #     #
    #     # cvcuda.as_tensor() is zero-copy — it creates a CV-CUDA view
    #     # of the existing PyTorch GPU memory. No data is duplicated.
    #     # "NHWC" tells CV-CUDA the memory layout of our tensor.
    #     # ------------------------------------------------------------------
    #     cvcuda_input = cvcuda.as_tensor(batch_gpu, "NHWC")

    #     # ------------------------------------------------------------------
    #     # Step 3: Keep a copy of original frames at full resolution
    #     #
    #     # Convert uint8 [0,255] → float32 [0,1] on GPU using PyTorch.
    #     # CV-CUDA does not have a direct uint8→float normalize operator
    #     # so we use PyTorch for this scalar operation (stays on GPU).
    #     # ------------------------------------------------------------------
    #     orig_gpu = batch_gpu.float() / 255.0                    # (B, H, W, 3) float32

    #     # ------------------------------------------------------------------
    #     # Step 4: Resize using CV-CUDA GPU kernel
    #     #
    #     # cvcuda.resize() runs a CUDA kernel — much faster than
    #     # cv2.resize() which runs on CPU.
    #     #
    #     # out_size for cvcuda is (B, out_h, out_w, C) — note height first
    #     # cvcuda.Interp.LINEAR = bilinear interpolation (same as OpenCV)
    #     # ------------------------------------------------------------------
    #     resized_cvcuda = cvcuda.resize(
    #         cvcuda_input,
    #         (B, out_h, out_w, 3),
    #         cvcuda.Interp.LINEAR,
    #     )

    #     # ------------------------------------------------------------------
    #     # Step 5: Convert resized CV-CUDA tensor back to PyTorch tensor
    #     #
    #     # torch.as_tensor(cvcuda_tensor) is zero-copy — same GPU memory.
    #     # Result is (B, H, W, C) uint8 — convert to float32 [0,1].
    #     # ------------------------------------------------------------------
    #     # resized_gpu = torch.as_tensor(resized_cvcuda.cuda()).float() / 255.0
    #     # ------------------------------------------------------------------
    #     # Step 5: Convert resized CV-CUDA tensor back to PyTorch tensor
    #     #
    #     # In CV-CUDA 0.16.0 we access the underlying cuda tensor directly
    #     # via .cuda() on the cvcuda tensor which returns a torch tensor
    #     # ------------------------------------------------------------------
    #     # resized_torch = resized_cvcuda.cuda()                   # PyTorch tensor on GPU
    #     # resized_gpu   = resized_torch.float() / 255.0
    #     # ------------------------------------------------------------------
    #     # Step 5: Convert resized CV-CUDA tensor back to PyTorch tensor
    #     #
    #     # CV-CUDA 0.16.0 returns ExternalBuffer from operations.
    #     # We convert via numpy on GPU using dlpack — zero copy bridge
    #     # between CV-CUDA and PyTorch that works across buffer types.
    #     # ------------------------------------------------------------------
    #     import torch.utils.dlpack as torchdlpack
    #     resized_gpu = torchdlpack.from_dlpack(
    #         resized_cvcuda.__dlpack__()
    #     ).float() / 255.0

    #     # ------------------------------------------------------------------
    #     # Step 6: Normalize using PyTorch on GPU
    #     #
    #     # Same formula as torchvision.transforms.Normalize:
    #     #   output = (input - mean) / std
    #     #
    #     # self._mean and self._std are (3,) GPU tensors.
    #     # They broadcast over (B, H, W, 3) automatically.
    #     # Everything stays on GPU — no .cpu() call anywhere.
    #     # ------------------------------------------------------------------
    #     # normalized_gpu = (resized_gpu - self._mean) / self._std # (B, H, W, 3)
    #     self._ensure_on_gpu(resized_gpu.device)
    #     normalized_gpu = (resized_gpu - self._mean) / self._std

    #     # ------------------------------------------------------------------
    #     # Step 7: Convert NHWC → NCHW for PyTorch model
    #     #
    #     # CV-CUDA works in NHWC (batch, height, width, channels).
    #     # PyTorch models expect NCHW (batch, channels, height, width).
    #     # .permute() is a zero-copy view — just changes stride metadata.
    #     # .contiguous() ensures memory is laid out for efficient model access.
    #     # ------------------------------------------------------------------
    #     orig_tensor       = orig_gpu.permute(0, 3, 1, 2).contiguous()
    #     resized_tensor    = resized_gpu.permute(0, 3, 1, 2).contiguous()
    #     normalized_tensor = normalized_gpu.permute(0, 3, 1, 2).contiguous()

    #     return orig_tensor, resized_tensor, normalized_tensor
    def __call__(
        self,
        frames:   List[np.ndarray],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        out_w, out_h = out_size
        B = len(frames)

        # ------------------------------------------------------------------
        # Step 1: Upload raw frames CPU → GPU as one contiguous tensor
        # ------------------------------------------------------------------
        batch_np  = np.stack(frames, axis=0)              # (B, H, W, 3) uint8
        batch_gpu = torch.as_tensor(
            batch_np, device="cuda"
        ).contiguous()                                     # (B, H, W, 3) uint8 GPU

        # ------------------------------------------------------------------
        # Step 2: Keep original — convert uint8→float32 via cvcuda.convertto
        # scale=1/255 does the [0,255]→[0,1] conversion inside the GPU kernel
        # ------------------------------------------------------------------
        cvcuda_orig = cvcuda.as_tensor(batch_gpu, "NHWC")
        cvcuda_orig_f32 = cvcuda.convertto(cvcuda_orig, np.float32, scale=1/255)

        # ------------------------------------------------------------------
        # Step 3: Resize using CV-CUDA GPU kernel
        # ------------------------------------------------------------------
        cvcuda_resized = cvcuda.resize(
            cvcuda_orig,
            (B, out_h, out_w, 3),
            cvcuda.Interp.LINEAR,
        )

        # ------------------------------------------------------------------
        # Step 4: Convert resized uint8 → float32 via CV-CUDA
        # ------------------------------------------------------------------
        cvcuda_resized_f32 = cvcuda.convertto(
            cvcuda_resized, np.float32, scale=1/255
        )

        # ------------------------------------------------------------------
        # Step 5: Normalize using CV-CUDA normalize operator
        # Requires mean and std as cvcuda tensors shaped (1,1,1,3) NHWC
        # ------------------------------------------------------------------
        self._ensure_on_gpu(torch.device("cuda"))

        mean_t = self._mean.reshape(1, 1, 1, 3).contiguous()
        std_t  = self._std.reshape(1, 1, 1, 3).contiguous()

        cvcuda_mean = cvcuda.as_tensor(mean_t, "NHWC")
        cvcuda_std  = cvcuda.as_tensor(std_t,  "NHWC")

        cvcuda_normalized = cvcuda.normalize(
            cvcuda_resized_f32,
            base=cvcuda_mean,
            scale=cvcuda_std,
            flags=cvcuda.NormalizeFlags.SCALE_IS_STDDEV,
        )

        # ------------------------------------------------------------------
        # Step 6: Reformat NHWC → NCHW for PyTorch model
        # cvcuda.reformat is zero-copy — just changes stride metadata
        # ------------------------------------------------------------------
        cvcuda_orig_nchw = cvcuda.reformat(cvcuda_orig_f32,   "NCHW")
        cvcuda_res_nchw  = cvcuda.reformat(cvcuda_resized_f32, "NCHW")
        cvcuda_norm_nchw = cvcuda.reformat(cvcuda_normalized,  "NCHW")

        # # ------------------------------------------------------------------
        # # Step 7: Wrap as PyTorch tensors — torch.as_tensor on cvcuda.Tensor
        # # In 0.16.0 we pass the .cuda() result which returns a torch tensor
        # # ------------------------------------------------------------------
        # orig_tensor       = torch.as_tensor(cvcuda_orig_nchw.cuda()).float()
        # resized_tensor    = torch.as_tensor(cvcuda_res_nchw.cuda()).float()
        # normalized_tensor = torch.as_tensor(cvcuda_norm_nchw.cuda()).float()

        # return orig_tensor, resized_tensor, normalized_tensor
        # ------------------------------------------------------------------
        # Step 7: Convert cvcuda ExternalBuffer → PyTorch CUDA tensor
        #
        # cvcuda Tensor.cuda() returns an ExternalBuffer, not a torch tensor.
        # torch.as_tensor(ext_buf, device='cuda') reads the buffer via
        # __cuda_array_interface__ and wraps it as a CUDA tensor zero-copy.
        # Without device='cuda' it falls back to CPU — always pass it.
        # ------------------------------------------------------------------
        orig_tensor = torch.as_tensor(
            cvcuda_orig_nchw.cuda(), device="cuda"
        ).float()
        resized_tensor = torch.as_tensor(
            cvcuda_res_nchw.cuda(), device="cuda"
        ).float()
        normalized_tensor = torch.as_tensor(
            cvcuda_norm_nchw.cuda(), device="cuda"
        ).float()

        return orig_tensor, resized_tensor, normalized_tensor