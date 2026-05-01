"""
src/utils/config.py
===================
Loads config.yaml and exposes settings as a typed dataclass.
Every other module imports from here — nobody reads yaml directly.

Why a dataclass instead of a raw dict:
  - cfg["pipeline"]["batch_size"]  ← dict: easy to typo, no autocomplete
  - cfg.pipeline.batch_size        ← dataclass: autocomplete, clear error if missing
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class PipelineConfig:
    batch_size:     int
    inference_size: List[int]   # [width, height]


@dataclass
class DataConfig:
    input_video: str
    output_dir:  str


@dataclass
class ProfilingConfig:
    nsight_reports_dir: str
    warmup_batches:     int


@dataclass
class ModelConfig:
    target_class: str


@dataclass
class NormalizeConfig:
    mean: List[float]
    std:  List[float]


@dataclass
class Config:
    pipeline:   PipelineConfig
    data:       DataConfig
    profiling:  ProfilingConfig
    model:      ModelConfig
    normalize:  NormalizeConfig


def load_config(path: str = "config.yaml") -> Config:
    """
    Load and parse config.yaml into a Config dataclass.

    Usage:
        from src.utils.config import load_config
        cfg = load_config()
        print(cfg.pipeline.batch_size)   # 4
        print(cfg.normalize.mean)        # [0.485, 0.456, 0.406]
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}\n"
            f"Make sure you run benchmark.py from the project root."
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        pipeline  = PipelineConfig(**raw["pipeline"]),
        data      = DataConfig(**raw["data"]),
        profiling = ProfilingConfig(**raw["profiling"]),
        model     = ModelConfig(**raw["model"]),
        normalize = NormalizeConfig(**raw["normalize"]),
    )
