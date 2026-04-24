# CV-CUDA vs OpenCV Benchmark

Benchmarks GPU-accelerated CV-CUDA preprocessing against CPU-based OpenCV
preprocessing in a video segmentation pipeline.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python benchmark.py
```

## Profile with Nsight Systems

```bash
nsys profile --output nsight_reports/opencv_profile python benchmark.py --backend opencv
nsys profile --output nsight_reports/cvcuda_profile python benchmark.py --backend cvcuda
```

## Project Structure
data/input/       — place your input video here
data/output/      — processed videos written here
nsight_reports/   — Nsight Systems and Compute reports (not committed to git)
opencv_utils.py   — CPU-based preprocessing and postprocessing
cvcuda_utils.py   — GPU-based preprocessing and postprocessing
benchmark.py      — main benchmark script
config.yaml       — all pipeline parameters
