import math
import numpy as np

from .data_types import (
    Detection,
    ProjectedView,
    SphericalBBox,
    SphericalCandidate,
    SphericalDetection,
)
from .spherical_bbox import bbox_to_spherical_bbox


def to_spherical_candidates(
    view: ProjectedView,
    detections: list[Detection],
    erp_shape: tuple[int, ...],
) -> list[SphericalCandidate]:
    candidates = []

    for detection in detections:
        spherical_bbox = bbox_to_spherical_bbox(
            view,
            detection.bbox,
            erp_shape,
            mask=detection.mask,
        )

        spherical_detection = SphericalDetection(
            class_id=detection.class_id,
            confidence=detection.confidence,
            spherical_bbox=spherical_bbox,
        )

        candidates.append(
            SphericalCandidate(
                detection=spherical_detection,
                view_ids={view.view_id},
            )
        )

    return candidates
