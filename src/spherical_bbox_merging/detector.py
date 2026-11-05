"""Structural interface for any planar object detector.

Implementations do not need to inherit from this protocol. See docs/detector.md for
input image, exclusive bbox endpoint, and aligned binary mask conventions.
"""
from typing import Protocol

import numpy as np

from .data_types import Detection


class DetectorProtocol(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections in this input image's original pixel coordinates.

        Bboxes are (x1, y1, x2, y2), with exclusive x2/y2. Optional masks
        must be binary HxW arrays aligned with image, without letterbox padding.
        Return an empty list when no objects are detected.
        """
        ...
