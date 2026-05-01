"""
src/utils/timer.py
==================
Timing utility for benchmarking pipeline stages.

Records wall-clock time for each batch, computes statistics,
and formats a comparison table between backends.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StageTimer:
    """
    Records timing for one stage (e.g. preprocessing) across all batches.

    Usage:
        timer = StageTimer("preprocessing")
        timer.start()
        ... do work ...
        timer.stop()
        print(timer.mean_ms)
    """
    name:         str
    _times:       List[float] = field(default_factory=list, repr=False)
    _start:       Optional[float] = field(default=None, repr=False)

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            raise RuntimeError(f"StageTimer '{self.name}': stop() called before start()")
        self._times.append(time.perf_counter() - self._start)
        self._start = None

    @property
    def total_s(self) -> float:
        return sum(self._times)

    @property
    def mean_ms(self) -> float:
        if not self._times:
            return 0.0
        return (sum(self._times) / len(self._times)) * 1000

    @property
    def count(self) -> int:
        return len(self._times)

    def reset(self):
        self._times.clear()
        self._start = None


@dataclass
class PipelineTimer:
    """
    Holds one StageTimer per pipeline stage.
    Call .summary() to print a formatted breakdown.

    Usage:
        pt = PipelineTimer(backend="opencv")
        pt.decode.start(); ...; pt.decode.stop()
        pt.preprocess.start(); ...; pt.preprocess.stop()
        pt.summary(total_frames=474)
    """
    backend:     str
    decode:      StageTimer = field(default_factory=lambda: StageTimer("decode"))
    preprocess:  StageTimer = field(default_factory=lambda: StageTimer("preprocess"))
    inference:   StageTimer = field(default_factory=lambda: StageTimer("inference"))
    postprocess: StageTimer = field(default_factory=lambda: StageTimer("postprocess"))
    encode:      StageTimer = field(default_factory=lambda: StageTimer("encode"))

    @property
    def total_s(self) -> float:
        return (
            self.decode.total_s
            + self.preprocess.total_s
            + self.inference.total_s
            + self.postprocess.total_s
            + self.encode.total_s
        )

    def summary(self, total_frames: int) -> str:
        fps = total_frames / self.total_s if self.total_s > 0 else 0.0
        lines = [
            "",
            f"{'='*52}",
            f"  Backend : {self.backend}",
            f"  Frames  : {total_frames}",
            f"  Total   : {self.total_s:.2f}s   ({fps:.1f} fps)",
            f"{'─'*52}",
            f"  {'Stage':<16} {'mean ms/batch':>14} {'total s':>10}",
            f"{'─'*52}",
        ]
        for stage in [
            self.decode,
            self.preprocess,
            self.inference,
            self.postprocess,
            self.encode,
        ]:
            lines.append(
                f"  {stage.name:<16} {stage.mean_ms:>13.1f}ms"
                f" {stage.total_s:>9.2f}s"
            )
        lines.append(f"{'='*52}")
        return "\n".join(lines)


def compare(opencv_timer: PipelineTimer,
            cvcuda_timer: PipelineTimer,
            total_frames: int) -> str:
    """
    Prints a side-by-side comparison of two backends.
    Called after both pipelines have run.
    """
    oc = opencv_timer
    cv = cvcuda_timer

    def speedup(a: StageTimer, b: StageTimer) -> str:
        if b.total_s == 0:
            return "n/a"
        ratio = a.total_s / b.total_s
        return f"{ratio:.2f}x"

    lines = [
        "",
        f"{'='*62}",
        f"  BENCHMARK COMPARISON  —  {total_frames} frames",
        f"{'─'*62}",
        f"  {'Stage':<16} {'OpenCV (ms)':>14} {'CV-CUDA (ms)':>14} {'Speedup':>10}",
        f"{'─'*62}",
    ]

    stage_pairs = [
        (oc.preprocess,  cv.preprocess,  "preprocess"),
        (oc.inference,   cv.inference,   "inference"),
        (oc.postprocess, cv.postprocess, "postprocess"),
    ]

    for oc_stage, cv_stage, label in stage_pairs:
        lines.append(
            f"  {label:<16}"
            f" {oc_stage.mean_ms:>13.1f}ms"
            f" {cv_stage.mean_ms:>13.1f}ms"
            f" {speedup(oc_stage, cv_stage):>10}"
        )

    lines += [
        f"{'─'*62}",
        f"  {'TOTAL':<16}"
        f" {oc.total_s:>12.2f}s"
        f" {cv.total_s:>12.2f}s"
        f" {speedup(oc, cv):>10}",
        f"  {'FPS':<16}"
        f" {total_frames/oc.total_s:>13.1f}"
        f" {total_frames/cv.total_s:>13.1f}",
        f"{'='*62}",
    ]

    return "\n".join(lines)