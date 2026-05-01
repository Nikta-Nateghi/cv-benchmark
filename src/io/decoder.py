"""
src/io/decoder.py
=================
Reads video files frame by frame using OpenCV (CPU).

Why this is CPU-based and that is fine:
  For our benchmark we are comparing preprocessing backends (CPU vs GPU).
  The decoder is not what we are measuring so CPU is acceptable here.
  If you later add DeepStream, you would add deepstream_decoder.py
  in this same folder — nothing else in the project changes.

Output format:
  Each batch is a Batch object containing:
    .frame  — list of numpy arrays (H, W, 3) uint8 RGB
    .idx    — integer batch counter starting at 0
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Batch:
    """
    Container that travels through the pipeline stages.
    Each stage reads from it and/or writes back to it.

    frame starts as a list of raw numpy arrays from the decoder.
    By the time it reaches the encoder it contains processed frames.
    """
    frame: List[np.ndarray]   # list of (H, W, 3) uint8 RGB arrays
    idx:   int                # batch number — used for progress reporting


class BatchDecoder:
    """
    Opens a video file and yields batches of RGB frames.

    Usage:
        decoder = BatchDecoder("data/input/sample.mp4", batch_size=4)
        decoder.start()
        while True:
            batch = decoder()
            if batch is None:
                break
            # batch.frame is a list of numpy arrays
        decoder.join()
    """

    def __init__(self, fname: str, batch_size: int):
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path.resolve()}")

        self.fname      = str(path)
        self.batch_size = batch_size
        self._cap:  Optional[cv2.VideoCapture] = None
        self._idx   = 0

        # Read metadata without keeping capture open
        # (start() opens it properly)
        cap = cv2.VideoCapture(self.fname)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open: {self.fname}")

        self.fps        = cap.get(cv2.CAP_PROP_FPS)
        self.width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(
            f"[Decoder] {path.name} | "
            f"{self.width}x{self.height} | "
            f"{self.fps:.1f}fps | "
            f"{self.total_frames} frames | "
            f"batch_size={self.batch_size}"
        )

    def start(self):
        """Open the video capture. Call before the first __call__."""
        self._cap = cv2.VideoCapture(self.fname)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video on start(): {self.fname}")
        self._idx = 0

    def __call__(self) -> Optional[Batch]:
        """
        Read the next batch of frames.
        Returns None when the video is exhausted.

        OpenCV reads frames as BGR — we convert to RGB immediately
        so every downstream module works in RGB consistently.
        The encoder converts back to BGR when writing.
        """
        if self._cap is None:
            raise RuntimeError("BatchDecoder.start() must be called before reading.")

        frames = []
        for _ in range(self.batch_size):
            ok, frame_bgr = self._cap.read()
            if not ok:
                break
            # BGR → RGB: keeps color format consistent across the pipeline
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        if not frames:
            return None     # signal to pipeline: video is done

        batch = Batch(frame=frames, idx=self._idx)
        self._idx += 1
        return batch

    @property
    def total_batches(self) -> int:
        """How many full+partial batches this video will produce."""
        import math
        return math.ceil(self.total_frames / self.batch_size)

    def join(self):
        """Release the video capture. Call after the pipeline loop ends."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None