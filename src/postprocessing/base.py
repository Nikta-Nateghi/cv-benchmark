"""
src/postprocessing/base.py
==========================
Abstract base class that defines the postprocessing contract.

Postprocessing takes the model's output and turns it back into
a viewable frame — specifically it blurs the background and
keeps the detected object (cat) sharp.

It needs three inputs from preprocessing plus the model output:

  probabilities   — what the model thinks each pixel is
  orig_tensor     — the original full-resolution frame
                    (to composite the final result onto)
  resized_tensor  — the frame at inference size
                    (intermediate step for clean mask upscaling)
  class_index     — which class index means "our target object"

Why resized_tensor is needed for mask upscaling:
  The model outputs a mask at 224x224.
  The original frame is 3840x2160.
  Going directly 224→3840 in one step on a 4K frame produces
  blocky artifacts at object edges because the jump is too large.
  Going 224→224 (align with resized) → 3840x2160 (with smoothing)
  gives much cleaner edges.

Output:
  A list of numpy arrays (H, W, 3) uint8 RGB — one per frame.
  The encoder converts these to BGR when writing to disk.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch


class BasePostprocessor(ABC):
    """
    All postprocessors must inherit from this and implement __call__.

    The pipeline runner holds a reference typed as BasePostprocessor
    so it works identically with either backend.
    """

    def __init__(self, output_layout: str, gpu_output: bool):
        """
        Parameters
        ----------
        output_layout : "HWC" or "CHW"
            Format the encoder expects.
            "HWC" = (Height, Width, Channels) — standard numpy/OpenCV format.
            "CHW" = (Channels, Height, Width) — PyTorch tensor format.
            Our OpenCV encoder needs "HWC".
            An NVENC encoder would need "CHW" GPU tensors.

        gpu_output : bool
            If False — return CPU numpy arrays (OpenCV encoder needs this).
            If True  — return GPU tensors (NVENC encoder can consume directly,
                       skipping the GPU→CPU copy entirely).
            Read from encoder.gpu_input so they stay in sync automatically.
        """
        self.output_layout = output_layout
        self.gpu_output    = gpu_output

    @abstractmethod
    def __call__(
        self,
        probabilities:  torch.Tensor,
        orig_tensor:    torch.Tensor,
        resized_tensor: torch.Tensor,
        class_index:    int,
    ) -> List[np.ndarray]:
        """
        Blend model predictions back onto original frames.

        Parameters
        ----------
        probabilities  : (B, num_classes, h, w) float32 — model output, on GPU
        orig_tensor    : (B, C, H, W) float32           — full resolution frames
        resized_tensor : (B, C, h, w) float32           — inference-size frames
        class_index    : int — which channel in probabilities is our target class

        Returns
        -------
        List of (H, W, 3) uint8 RGB numpy arrays, one per frame in batch.
        """
        ...

    @property
    def name(self) -> str:
        """Human readable backend name for logging."""
        return self.__class__.__name__