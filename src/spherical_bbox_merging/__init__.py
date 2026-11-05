"""Spherical bounding-box merging for user-provided planar detectors."""

from .data_types import (
    Detection,
    ProjectedView,
    ProjectionOptions,
    SphericalBBox,
    SphericalCandidate,
    SphericalDetection,
    ThresholdOptions,
)
from .detector import DetectorProtocol
from .pipeline import detect_panorama

__all__ = [
    "detect_panorama",
    "Detection",
    "DetectorProtocol",
    "ProjectedView",
    "ProjectionOptions",
    "SphericalBBox",
    "SphericalCandidate",
    "SphericalDetection",
    "ThresholdOptions",
]
