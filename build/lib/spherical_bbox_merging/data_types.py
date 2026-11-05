from dataclasses import dataclass
import numpy as np
  

@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    mask: np.ndarray | None = None  # Binary HxW mask in the projected image's coordinates.

@dataclass
class ProjectionOptions:
    cut_angle_deg: int = 60 # Must be a divisor of 180.
    cut_fov_deg: float = 105.0  # Diagonal field of view in degrees.
    width: int = 640
    height: int = 640

@dataclass
class ProjectedView:
    view_id: int
    yaw: float  # Longitude in radians; ERP x = width * (0.5 - yaw / (2*pi)).
    pitch: float  # Latitude in radians, positive north.
    image: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray


@dataclass
class SphericalBBox:
    yaw: float
    pitch: float
    horizontal_fov: float
    vertical_fov: float

@dataclass
class ThresholdOptions:
    iou_threshold: float = 0.5
    ioa_threshold: float = 0.1


@dataclass
class SphericalDetection:
    class_id: int
    confidence: float
    spherical_bbox: SphericalBBox


@dataclass
class SphericalCandidate:
    detection: SphericalDetection
    view_ids: set
