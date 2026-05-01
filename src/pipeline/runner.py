"""
src/pipeline/runner.py
======================
Connects all pipeline stages and runs the main loop.

This is the conveyor belt. It does not know or care how any
individual stage works internally — it just calls them in order
and passes data between them.

Stage order per batch:
  1. Decode        — read raw frames from video file
  2. Preprocess    — resize + normalize for model input
  3. Inference     — run segmentation model on GPU
  4. Postprocess   — blur background using model output
  5. Encode        — write processed frames to output file

The runner owns the timing — it wraps each stage call with
timer.start() / timer.stop() so we get per-stage measurements.

Warmup:
  The first N batches are processed but not timed. This is because:
  - First inference call triggers CUDA kernel compilation (cuDNN benchmark)
  - First CV-CUDA call initializes GPU context
  - OS page faults on first memory access
  These one-time costs would skew the benchmark numbers if included.
"""

import torch
from tqdm import tqdm
from typing import Tuple

from src.io.decoder import BatchDecoder
from src.io.encoder import BatchEncoder
from src.preprocessing.base import BasePreprocessor
from src.postprocessing.base import BasePostprocessor
from src.inference.segmentation import Segmentation
from src.utils.timer import PipelineTimer
from src.utils.config import Config


class PipelineRunner:
    """
    Runs the full video processing pipeline for one backend.

    Usage:
        runner = PipelineRunner(cfg, preprocessor, postprocessor, device)
        timer  = runner.run(backend_name="opencv")
        print(timer.summary(total_frames=474))
    """

    def __init__(
        self,
        cfg:           Config,
        preprocessor:  BasePreprocessor,
        postprocessor: BasePostprocessor,
        device:        torch.device,
    ):
        self.cfg          = cfg
        self.preprocessor = preprocessor
        self.device       = device

        # Postprocessor is built outside and passed in because
        # it needs encoder.input_layout and encoder.gpu_input
        # which are only known after the encoder is constructed
        self.postprocessor = postprocessor

        # inference_size from config — (width, height)
        self.inference_size = tuple(cfg.pipeline.inference_size)

    def run(self, backend_name: str, output_suffix: str) -> Tuple[PipelineTimer, int]:
        """
        Execute the full pipeline — decode → preprocess → infer →
        postprocess → encode — and return timing results.

        Parameters
        ----------
        backend_name  : label for timer and logging e.g. "opencv"
        output_suffix : appended to output filename e.g. "opencv"
                        produces data/output/result_opencv.mp4

        Returns
        -------
        timer        : PipelineTimer with per-stage measurements
        total_frames : how many frames were actually processed
                       (may differ from video total if batch is partial)
        """
        cfg = self.cfg

        # Build output path from config output dir + suffix
        import os
        output_path = os.path.join(
            cfg.data.output_dir, f"result_{output_suffix}.mp4"
        )

        # ------------------------------------------------------------
        # Build decoder and encoder fresh for each run
        # so both backends start from frame 0 of the same video
        # ------------------------------------------------------------
        decoder = BatchDecoder(
            fname=cfg.data.input_video,
            batch_size=cfg.pipeline.batch_size,
        )
        encoder = BatchEncoder(
            fname=output_path,
            fps=decoder.fps,
            width=decoder.width,
            height=decoder.height,
        )

        # ------------------------------------------------------------
        # Build the segmentation model
        # Loaded fresh each run so GPU memory state is consistent
        # ------------------------------------------------------------
        model = Segmentation(
            target_class=cfg.model.target_class,
            batch_size=cfg.pipeline.batch_size,
            inference_size=self.inference_size,
            device=self.device,
        )

        # cuDNN benchmark mode — on first forward pass cuDNN runs a
        # short benchmark to find the fastest conv algorithm for this
        # exact input shape, then reuses it every subsequent batch.
        # Safe here because our input shape is fixed (always 224x224).
        torch.backends.cudnn.benchmark = True

        timer        = PipelineTimer(backend=backend_name)
        total_frames = 0
        warmup       = cfg.profiling.warmup_batches

        decoder.start()
        encoder.start()

        print(f"\n[Runner] Starting pipeline — backend={backend_name}")
        print(f"[Runner] Warmup batches: {warmup} (not included in timing)")
        print(f"[Runner] Output → {output_path}\n")

        with tqdm(
            total=decoder.total_batches,
            desc=f"{backend_name:>8}",
            unit="batch",
        ) as pbar:

            idx_batch = 0

            while True:

                # --------------------------------------------------------
                # Stage 1: Decode
                # --------------------------------------------------------
                timer.decode.start()
                batch = decoder()
                timer.decode.stop()

                if batch is None:
                    break   # video exhausted

                # --------------------------------------------------------
                # Stage 2: Preprocess
                # --------------------------------------------------------
                timer.preprocess.start()
                orig_tensor, resized_tensor, normalized_tensor = (
                    self.preprocessor(batch.frame, self.inference_size)
                )
                timer.preprocess.stop()

                # --------------------------------------------------------
                # Stage 3: Inference
                # --------------------------------------------------------
                timer.inference.start()
                probabilities = model(normalized_tensor)
                # Synchronize GPU so timer captures real inference time
                # Without this, CUDA queues the work asynchronously and
                # timer.stop() fires before the GPU is actually done
                torch.cuda.synchronize()
                timer.inference.stop()

                # --------------------------------------------------------
                # Stage 4: Postprocess
                # --------------------------------------------------------
                timer.postprocess.start()
                processed_frames = self.postprocessor(
                    probabilities,
                    orig_tensor,
                    resized_tensor,
                    model.class_index,
                )
                timer.postprocess.stop()

                # --------------------------------------------------------
                # Stage 5: Encode
                # --------------------------------------------------------
                timer.encode.start()
                batch.frame = processed_frames
                encoder(batch)
                timer.encode.stop()

                # --------------------------------------------------------
                # Warmup handling — reset timers after warmup batches
                # so one-time startup costs don't skew results
                # --------------------------------------------------------
                if idx_batch == warmup - 1:
                    timer = PipelineTimer(backend=backend_name)
                    print(f"[Runner] Warmup done — timing starts now")

                total_frames += len(batch.frame)
                idx_batch    += 1
                pbar.update(1)

        decoder.join()
        encoder.join()

        # Subtract warmup frames from total since we reset the timer
        warmup_frames = warmup * cfg.pipeline.batch_size
        timed_frames  = max(total_frames - warmup_frames, 1)

        print(f"\n[Runner] Done — {total_frames} frames total "
              f"({timed_frames} timed, {warmup_frames} warmup)")

        return timer, timed_frames