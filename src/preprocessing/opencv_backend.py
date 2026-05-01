"""
src/preprocessing/opencv_backend.py
====================================
CPU-based preprocessing using OpenCV and NumPy.

This is the BASELINE backend. Every operation runs on the CPU.
Frames stay in CPU RAM the entire time.

The equivalent torchvision pipeline would be:
    transforms.Compose([
        transforms.Resize(out_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

The difference is that torchvision works on one PIL image at a time.
This works on a batch of raw numpy frames directly, which is more
efficient for video pipelines where frames are already numpy arrays
coming out of the decoder.

Data path:
    CPU RAM (raw frames)
        → resize (CPU/OpenCV)
        → normalize (CPU/NumPy)
        → stack into batch tensor (CPU/PyTorch)
        → [pipeline moves tensor to GPU for inference]
        → [postprocessing brings result back to CPU]
"""

from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.preprocessing.base import BasePreprocessor


class OpenCVPreprocessor(BasePreprocessor):
    """
    CPU preprocessor. Implements BasePreprocessor contract.

    Usage:
        pre = OpenCVPreprocessor(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        orig, resized, normalized = pre(frames, out_size=(224, 224))
    """

    def __init__(
        self,
        mean: List[float] = (0.485, 0.456, 0.406),
        std:  List[float] = (0.229, 0.224, 0.225),
    ):
        # Store as float32 numpy arrays shaped (1, 1, 3) so NumPy
        # broadcasts correctly over (H, W, 3) frames without loops
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std  = np.array(std,  dtype=np.float32).reshape(1, 1, 3)

    @property
    def name(self) -> str:
        return "opencv (CPU)"

    def __call__(
        self,
        frames:   List[np.ndarray],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        frames   : list of (H, W, 3) uint8 RGB numpy arrays
        out_size : (width, height) — note OpenCV resize takes (W, H)

        Returns
        -------
        orig_tensor       : (B, C, H, W) float32 CPU tensor
        resized_tensor    : (B, C, h, w) float32 CPU tensor
        normalized_tensor : (B, C, h, w) float32 CPU tensor
        """
        out_w, out_h = out_size

        orig_list       = []
        resized_list    = []
        normalized_list = []

        for frame in frames:
            # ----------------------------------------------------------
            # 1. Keep original at full resolution for compositing later
            #    Convert uint8 → float32 [0, 1] range
            # ----------------------------------------------------------
            orig_f32 = frame.astype(np.float32) / 255.0
            orig_list.append(orig_f32)

            # ----------------------------------------------------------
            # 2. Resize to model input size
            #    INTER_LINEAR = bilinear interpolation, good balance of
            #    speed and quality for downscaling 4K → 224x224
            # ----------------------------------------------------------
            resized = cv2.resize(
                frame, (out_w, out_h),
                interpolation=cv2.INTER_LINEAR
            )
            resized_f32 = resized.astype(np.float32) / 255.0
            resized_list.append(resized_f32)

            # ----------------------------------------------------------
            # 3. Normalize: (pixel - mean) / std
            #    Same operation as torchvision.transforms.Normalize
            #    self._mean and self._std broadcast over (H, W, 3)
            # ----------------------------------------------------------
            normalized = (resized_f32 - self._mean) / self._std
            normalized_list.append(normalized)

        # ----------------------------------------------------------
        # Stack individual frames into batch arrays: (B, H, W, C)
        # then permute to PyTorch's expected (B, C, H, W) format
        # np.stack is faster than torch.stack on CPU numpy arrays
        # ----------------------------------------------------------
        def to_bchw(arrays: List[np.ndarray]) -> torch.Tensor:
            stacked = np.stack(arrays, axis=0)          # (B, H, W, C)
            tensor  = torch.from_numpy(stacked)         # still (B, H, W, C)
            return tensor.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        orig_tensor       = to_bchw(orig_list)
        resized_tensor    = to_bchw(resized_list)
        normalized_tensor = to_bchw(normalized_list)

        # All three tensors live on CPU at this point.
        # The pipeline runner will call .to(device) on normalized_tensor
        # before passing it to the model. orig and resized stay on CPU
        # because postprocessing (OpenCV) needs them there.
        return orig_tensor, resized_tensor, normalized_tensor