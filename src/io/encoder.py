"""
src/io/encoder.py
=================
Writes processed frames to an output video file using OpenCV (CPU).

Why CPU is fine here:
  Same reasoning as decoder — encoding is not what we are benchmarking.
  If you add DeepStream/NVENC later, add deepstream_encoder.py here.

Input format expected:
  Frames as numpy arrays (H, W, 3) uint8 RGB.
  The encoder converts RGB → BGR internally before writing,
  so every upstream module works in RGB consistently.

Two attributes are read by Postprocessing to know how to
format its output:
  .input_layout  — "HWC" means (Height, Width, Channels) numpy array
  .gpu_input     — False means encoder needs CPU numpy, not a GPU tensor
                   When you add NVENC this becomes True and postprocessing
                   skips the GPU→CPU copy automatically.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.io.decoder import Batch


class BatchEncoder:
    """
    Writes batches of RGB frames to an mp4 file.

    Usage:
        encoder = BatchEncoder(
            fname="data/output/result_opencv.mp4",
            fps=25.0,
            width=3840,
            height=2160,
        )
        encoder.start()
        encoder(batch)   # batch.frame is list of numpy (H,W,3) uint8 RGB
        encoder.join()
    """

    # These two are read by Postprocessing.__init__() to know
    # what format to return frames in — do not rename them
    input_layout: str = "HWC"    # Height x Width x Channels
    gpu_input:    bool = False    # needs CPU numpy arrays, not GPU tensors

    def __init__(self, fname: str, fps: float, width: int, height: int):
        self.fname  = str(Path(fname))
        self.fps    = fps
        self.width  = width
        self.height = height
        self._writer: cv2.VideoWriter | None = None
        self._frames_written = 0

        # Make sure output directory exists
        Path(fname).parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """
        Open the VideoWriter. Call before the pipeline loop.
        Separated from __init__ so decoder can be inspected first
        and width/height confirmed before committing to a file.
        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.fname, fourcc, self.fps, (self.width, self.height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"OpenCV could not open VideoWriter for: {self.fname}\n"
                f"Check that the output directory exists and is writable."
            )
        print(
            f"[Encoder] {Path(self.fname).name} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.1f}fps"
        )

    def __call__(self, batch: "Batch"):
        """
        Write all frames in a batch to the output file.

        Accepts frames as:
          - numpy array (H, W, 3) uint8 RGB  ← normal path
          - torch tensor (C, H, W) float     ← converts automatically
        """
        if self._writer is None:
            raise RuntimeError("BatchEncoder.start() must be called before writing.")

        for frame in batch.frame:
            # Handle torch tensor input — convert to numpy first
            if hasattr(frame, "cpu"):
                frame = frame.cpu().numpy()

            # Handle (C, H, W) → (H, W, C)
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = frame.transpose(1, 2, 0)

            # Ensure uint8
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            # RGB → BGR for OpenCV writer
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._writer.write(frame_bgr)
            self._frames_written += 1

    def join(self):
        """
        Flush and close the video file.
        Call after the pipeline loop — not calling this will produce
        a corrupt or empty output file.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"[Encoder] Wrote {self._frames_written} frames → {self.fname}")