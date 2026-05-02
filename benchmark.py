"""
benchmark.py
============
Entry point. Runs the full pipeline twice — once per backend —
and prints a timing comparison.

Usage:
    python benchmark.py                  # runs both backends
    python benchmark.py --backend opencv # runs only opencv
    python benchmark.py --backend cvcuda # runs only cvcuda

Nsight Systems profiling:
    nsys profile --output nsight_reports/opencv_profile \
        python benchmark.py --backend opencv

    nsys profile --output nsight_reports/cvcuda_profile \
        python benchmark.py --backend cvcuda
"""

import argparse
import torch

from src.utils.config import load_config
from src.utils.timer import compare
from src.pipeline.runner import PipelineRunner
from src.io.encoder import BatchEncoder


def build_opencv_runner(cfg, device):
    from src.preprocessing.opencv_backend import OpenCVPreprocessor
    from src.postprocessing.opencv_backend import OpenCVPostprocessor

    pre  = OpenCVPreprocessor(
        mean=cfg.normalize.mean,
        std=cfg.normalize.std,
    )
    post = OpenCVPostprocessor(
        output_layout="HWC",
        gpu_output=False,
    )
    return PipelineRunner(cfg, pre, post, device)


def build_cvcuda_runner(cfg, device):
    from src.preprocessing.cvcuda_backend import CVCUDAPreprocessor
    from src.postprocessing.cvcuda_backend import CVCUDAPostprocessor

    pre  = CVCUDAPreprocessor(
        mean=cfg.normalize.mean,
        std=cfg.normalize.std,
    )
    post = CVCUDAPostprocessor(
        output_layout="HWC",
        gpu_output=False,
    )
    return PipelineRunner(cfg, pre, post, device)


def main():
    parser = argparse.ArgumentParser(description="CV-CUDA vs OpenCV benchmark")
    parser.add_argument(
        "--backend",
        choices=["opencv", "cvcuda", "both"],
        default="both",
        help="Which backend to run (default: both)",
    )
    args = parser.parse_args()

    cfg    = load_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Backend : {args.backend}")
    print(f"Video   : {cfg.data.input_video}")
    print(f"Batch   : {cfg.pipeline.batch_size}")
    print()

    opencv_timer = None
    cvcuda_timer = None
    opencv_frames = 0
    cvcuda_frames = 0

    # ------------------------------------------------------------------
    # Run OpenCV (CPU) baseline
    # ------------------------------------------------------------------
    if args.backend in ("opencv", "both"):
        print("=" * 52)
        print("  Running OpenCV (CPU) backend")
        print("=" * 52)
        runner = build_opencv_runner(cfg, device)
        opencv_timer, opencv_frames = runner.run(
            backend_name="opencv",
            output_suffix="opencv",
        )
        print(opencv_timer.summary(total_frames=opencv_frames))

    # ------------------------------------------------------------------
    # Run CV-CUDA (GPU) backend
    # ------------------------------------------------------------------
    if args.backend in ("cvcuda", "both"):
        print("=" * 52)
        print("  Running CV-CUDA (GPU) backend")
        print("=" * 52)
        runner = build_cvcuda_runner(cfg, device)
        cvcuda_timer, cvcuda_frames = runner.run(
            backend_name="cvcuda",
            output_suffix="cvcuda",
        )
        print(cvcuda_timer.summary(total_frames=cvcuda_frames))

    # ------------------------------------------------------------------
    # Print comparison if both ran
    # ------------------------------------------------------------------
    if opencv_timer and cvcuda_timer:
        total = min(opencv_frames, cvcuda_frames)
        print(compare(opencv_timer, cvcuda_timer, total_frames=total))
        print()
        print("Output videos:")
        print(f"  data/output/result_opencv.mp4")
        print(f"  data/output/result_cvcuda.mp4")
        print()
        print("To profile with Nsight Systems:")
        print("  nsys profile --output nsight_reports/opencv_profile "
              "python benchmark.py --backend opencv")
        print("  nsys profile --output nsight_reports/cvcuda_profile "
              "python benchmark.py --backend cvcuda")


if __name__ == "__main__":
    main()