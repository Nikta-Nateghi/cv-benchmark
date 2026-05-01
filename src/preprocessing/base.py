"""
src/preprocessing/base.py
=========================
Abstract base class that defines the preprocessing contract.

Both opencv_backend.py and cvcuda_backend.py MUST implement this
interface. The pipeline runner only knows about this base class —
it never imports a backend directly. This is what lets you swap
backends with zero changes to pipeline code.

What preprocessing must do:
  Input  — list of raw numpy frames (H, W, 3) uint8 RGB
  Output — three tensors:

  orig_tensor       (B, C, H, W)  float32
    The original frames at full resolution.
    Used in postprocessing to composite the blur back onto.

  resized_tensor    (B, C, h, w)  float32
    Frames resized to inference_size but NOT normalized.
    Used in postprocessing to scale the mask back up correctly.

  normalized_tensor (B, C, h, w)  float32
    Frames resized AND normalized (zero mean, unit std).
    This is the ONLY tensor the model ever sees.

Why three tensors instead of one:
  The model needs normalized_tensor.
  Postprocessing needs orig_tensor to know the original colors.
  Postprocessing needs resized_tensor as an intermediate for
  mask upscaling — going straight from inference_size to full
  resolution in one step produces artifacts on 4K frames.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch


class BasePreprocessor(ABC):
    """
    All preprocessors must inherit from this class and implement __call__.

    The pipeline runner types its preprocessor as BasePreprocessor,
    so it works identically whether it gets an OpenCVPreprocessor
    or a CVCUDAPreprocessor at runtime.
    """

    @abstractmethod
    def __call__(
        self,
        frames: List[np.ndarray],
        out_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a batch of raw frames into model-ready tensors.

        Parameters
        ----------
        frames   : list of numpy arrays (H, W, 3) uint8 RGB
        out_size : (width, height) to resize frames to

        Returns
        -------
        orig_tensor       : (B, C, H, W) float32  — original resolution
        resized_tensor    : (B, C, h, w) float32  — resized, not normalized
        normalized_tensor : (B, C, h, w) float32  — resized + normalized
        """
        ...

    @property
    def name(self) -> str:
        """Human readable backend name for logging."""
        return self.__class__.__name__