import numpy as np

from .data_types import SphericalDetection, ProjectionOptions, ThresholdOptions
from .detector import DetectorProtocol
from .projection import split_erp
from .spherical_candidates import to_spherical_candidates
from .merge import merge

def detect_panorama(
    erp_image: np.ndarray,
    detector: DetectorProtocol,
    projection_options: ProjectionOptions,
    threshold_options: ThresholdOptions,
) -> list[SphericalDetection]:

    views = split_erp(erp_image, projection_options)

    candidates = []

    for view in views:
        detections = detector.detect(view.image)

        new_candidates = to_spherical_candidates(
            view,
            detections,
            erp_image.shape[:2]
        )

        candidates.extend(new_candidates)

    results = merge(candidates, threshold_options)

    return results
