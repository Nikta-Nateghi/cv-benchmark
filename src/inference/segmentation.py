"""
src/inference/segmentation.py
==============================
Neural network wrapper for semantic segmentation.

We use DeepLabV3+ with a MobileNetV3 backbone from torchvision.
This is a real pretrained segmentation model trained on COCO dataset
which includes 21 classes — one of which is "cat".

Why DeepLabV3 and not something custom:
  - Pretrained on COCO so it actually detects cats in real video
  - Small enough to run on 8GB VRAM with batch_size=4
  - Fast enough that preprocessing difference is measurable
  - No training required — works out of the box

COCO classes this model knows (index: name):
  0: background    6: bus          12: dog         18: sofa
  1: aeroplane     7: car          13: horse       19: train
  2: bicycle       8: cat          14: motorbike   20: tv
  3: bird          9: chair        15: person
  4: boat         10: cow          16: pottedplant
  5: bottle       11: diningtable  17: sheep

So class_index for "cat" is 8.

Input:  (B, 3, H, W) float32 normalized tensor on GPU
Output: (B, num_classes, H, W) float32 probability tensor on GPU
        — stays on GPU so postprocessing can read it there
"""

import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
from typing import Tuple


# COCO class names in index order
# Used to look up class_index from a string name like "cat"
COCO_CLASSES = [
    "background", "aeroplane", "bicycle",  "bird",  "boat",
    "bottle",     "bus",       "car",      "cat",   "chair",
    "cow",        "diningtable", "dog",    "horse", "motorbike",
    "person",     "pottedplant", "sheep",  "sofa",  "train", "tv",
]


class Segmentation:
    """
    Wraps a pretrained DeepLabV3 model for single-class segmentation.

    Usage:
        model = Segmentation(
            target_class="cat",
            batch_size=4,
            inference_size=(224, 224),
            device=torch.device("cuda:0"),
        )
        probabilities = model(normalized_tensor)  # (B, 21, 224, 224)
        print(model.class_index)                  # 8
    """

    def __init__(
        self,
        target_class:   str,
        batch_size:     int,
        inference_size: Tuple[int, int],
        device:         torch.device,
    ):
        self.target_class  = target_class
        self.batch_size    = batch_size
        self.inference_size = inference_size
        self.device        = device

        # Resolve class name → index
        if target_class not in COCO_CLASSES:
            raise ValueError(
                f"'{target_class}' not in COCO classes.\n"
                f"Available: {COCO_CLASSES}"
            )
        self.class_index = COCO_CLASSES.index(target_class)

        print(f"[Segmentation] target='{target_class}' "
              f"class_index={self.class_index} "
              f"device={device}")

        # ------------------------------------------------------------------
        # Load pretrained DeepLabV3 with MobileNetV3 backbone
        #
        # MobileNetV3 is chosen over ResNet101 because:
        #   - Fits in 8GB VRAM with batch_size=4 at 224x224
        #   - Fast enough that pre/postprocessing is the measurable bottleneck
        #   - Still accurate enough to detect cats in real video
        #
        # weights=DEFAULT loads COCO pretrained weights automatically
        # No manual download needed — torchvision handles it
        # ------------------------------------------------------------------
        print("[Segmentation] Loading pretrained weights (first run downloads ~30MB)...")
        self._model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        self._model.eval()
        self._model.to(device)

        # ------------------------------------------------------------------
        # torch.compile() — optional but gives ~15% speedup on inference
        # Disabled by default because first call takes ~30s to compile.
        # Enable by setting compile=True if you want maximum inference speed.
        # ------------------------------------------------------------------
        # self._model = torch.compile(self._model, mode="reduce-overhead")

        print("[Segmentation] Model ready")

    @torch.no_grad()
    def __call__(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on a batch of normalized frames.

        Parameters
        ----------
        normalized_tensor : (B, 3, H, W) float32 on GPU
            Output of preprocessor — already normalized with ImageNet stats.

        Returns
        -------
        probabilities : (B, num_classes, H, W) float32 on GPU
            Raw logits from the model head.
            Postprocessor calls argmax on these to get per-pixel class.
            We do not softmax here because argmax gives the same result
            and avoids an extra GPU operation.
        """
        # Move to GPU if not already there
        # (OpenCV backend returns CPU tensors — we move here, not in preprocessor,
        #  so the preprocessor stays backend-agnostic)
        x = normalized_tensor.to(self.device)

        # DeepLabV3 returns a dict — the main output is under "out"
        # shape: (B, num_classes, H, W)
        output = self._model(x)["out"]

        # Output stays on GPU — postprocessor reads it from there
        return output