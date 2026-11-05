"""Convert planar detections into spherical bounding boxes.

The conversion follows the mathematical formulation described
in Section 3.3.2 of the accompanying research paper.

Supports bounding boxes and aligned binary instance masks.
"""

import math

import cv2
import numpy as np

from data_types import ProjectedView, SphericalBBox


def bbox_to_spherical_bbox(
    view: ProjectedView,
    bbox: tuple[float, float, float, float],
    erp_shape: tuple[int, ...],
    mask: np.ndarray | None = None,
) -> SphericalBBox:
    """Convert a bbox or aligned binary mask using the same local-angle math.

    Float endpoints are truncated toward zero; integer x2/y2 are exclusive.
    Sample four corners and four integer edge midpoints. Round the initial
    center's half coordinates away from zero. Clip corners before edge midpoints;
    derive the center from the original bbox. Empty masks fall back to the bbox.
    Equation (6)'s spans have an experiment-specific minimum of 1e-6 radians.
    """
    if len(erp_shape) < 2 or not np.all(np.isfinite(erp_shape[:2])):
        raise ValueError("erp_shape must contain finite height and width")
    erp_h, erp_w = erp_shape[:2]
    if erp_h <= 0 or erp_w <= 0:
        raise ValueError("ERP height and width must be positive")
    if view.image.ndim < 2:
        raise ValueError("view.image must have height and width dimensions")
    image_h, image_w = view.image.shape[:2]
    if view.map_x.shape != (image_h, image_w) or view.map_y.shape != (image_h, image_w):
        raise ValueError("ERP coordinate maps must match the view image dimensions")

    # Keep the bbox input contract even when a nonempty mask supplies the points.
    bbox_samples = _bbox_sample_pixels(bbox, image_w, image_h)
    mask_samples = _mask_sample_pixels(mask, image_w, image_h) if mask is not None else None
    points, center_pixel = mask_samples if mask_samples is not None else bbox_samples
    # c0 comes from the detection's center, not the view's yaw/pitch metadata.
    directions = _map_points_to_unit_vectors(
        view, np.asarray([center_pixel, *points]), erp_h, erp_w
    )
    c0, boundary = directions[0], directions[1:]

    # Equation (1): an orthonormal horizontal/vertical frame about c0.
    right, up = _local_basis(c0)

    # Equation (2): both angular offsets use the forward component p_k . c0.
    forward = boundary @ c0
    horizontal = np.arctan2(boundary @ right, forward)
    vertical = np.arctan2(boundary @ up, forward)

    # Equations (3) and (4): extrema and midpoints in the INITIAL local frame.
    u_min, u_max = float(horizontal.min()), float(horizontal.max())
    v_min, v_max = float(vertical.min()), float(vertical.max())
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0

    # Equation (5): recover a direction from the two tangent-plane slopes.
    center = c0 + math.tan(u_mid) * right + math.tan(v_mid) * up
    center /= np.linalg.norm(center)
    yaw = math.atan2(float(center[1]), float(center[0]))
    pitch = math.atan2(float(center[2]), math.hypot(center[0], center[1]))

    return SphericalBBox(
        yaw=(yaw + math.pi) % math.tau - math.pi,
        pitch=pitch,
        # Equation (6): preserve the original angular spans after recentering.
        horizontal_fov=max(1e-6, u_max - u_min),
        vertical_fov=max(1e-6, v_max - v_min),
    )


def _map_points_to_unit_vectors(
    view: ProjectedView,
    points: np.ndarray,
    erp_h: int,
    erp_w: int,
) -> np.ndarray:
    """Read ERP coordinates and independently lift yaw/pitch to the unit sphere.

    Cartesian convention: x is yaw=0, y is yaw=pi/2, z is north.
    Thus p = (cos(pitch) cos(yaw), cos(pitch) sin(yaw), sin(pitch)).
    """
    # The caller has already applied the distinct endpoint/midpoint/center
    # rules. These are integer pixels: do not round all coordinates again.
    pixels = np.asarray(points, dtype=np.intp)
    px = np.clip(pixels[:, 0], 0, view.map_x.shape[1] - 1)
    py = np.clip(pixels[:, 1], 0, view.map_x.shape[0] - 1)
    erp_x = np.asarray(view.map_x[py, px], dtype=np.float64)
    erp_y = np.asarray(view.map_y[py, px], dtype=np.float64)
    if not np.all(np.isfinite(erp_x)) or not np.all(np.isfinite(erp_y)):
        raise ValueError("Sampled ERP coordinates must be finite")

    # Periodic ERP x is reduced before trigonometry, including at the ERP seam.
    yaw = math.pi - math.tau * (np.remainder(erp_x, erp_w) / erp_w)
    pitch = math.pi / 2.0 - math.pi * (erp_y / erp_h)
    cos_pitch = np.cos(pitch)
    vectors = np.column_stack((cos_pitch * np.cos(yaw),
                               cos_pitch * np.sin(yaw), np.sin(pitch)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _local_basis(c0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equation (1), with the experiment's axis choice in Python coordinates."""
    reference = (np.array([0.0, 0.0, 1.0]) if abs(c0[2]) < 0.99
                 else np.array([0.0, -1.0, 0.0]))
    right = np.cross(reference, c0)
    right /= np.linalg.norm(right)
    up = np.cross(c0, right)
    up /= np.linalg.norm(up)
    return right, up


def _bbox_sample_pixels(
    bbox: tuple[float, float, float, float], image_w: int, image_h: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Eight representative pixels and the separately rounded initial center.

    Midpoints use the last valid pixels, with integer division toward zero.
    Corners are clipped before computing edge midpoints. The center is computed
    independently from the original integer bbox, then clipped.
    An integer bbox with no pixels (nonpositive width/height) is rejected.
    """
    if image_w <= 0 or image_h <= 0:
        raise ValueError("View image dimensions must be positive")
    coordinates = np.asarray(bbox, dtype=np.float64)
    if coordinates.shape != (4,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("bbox must contain four finite coordinates")
    if coordinates[0] > coordinates[2] or coordinates[1] > coordinates[3]:
        raise ValueError("bbox coordinates must satisfy x1 <= x2 and y1 <= y2")
    x1, y1, x2, y2 = (int(value) for value in coordinates)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Integer exclusive bbox must have positive width and height")

    def half_toward_zero(total: int) -> int:
        return total // 2 if total >= 0 else -((-total) // 2)

    def half_away_from_zero(total: int) -> int:
        return (total + 1) // 2 if total >= 0 else -((-total + 1) // 2)

    def clip_pixel(x: int, y: int) -> tuple[int, int]:
        return min(max(x, 0), image_w - 1), min(max(y, 0), image_h - 1)

    center = clip_pixel(half_away_from_zero(x1 + x2), half_away_from_zero(y1 + y2))
    left, top = clip_pixel(x1, y1)
    right, bottom = clip_pixel(x2 - 1, y2 - 1)
    mid_x = half_toward_zero(left + right)
    mid_y = half_toward_zero(top + bottom)
    points = [(left, top), (right, top), (right, bottom), (left, bottom),
              (mid_x, top), (mid_x, bottom), (left, mid_y), (right, mid_y)]
    return points, center


def _mask_sample_pixels(
    mask: np.ndarray, image_w: int, image_h: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """External contours, every fourth point per contour, in original-view pixels.

    Accept binary bool/0-1/0-255 masks only. No resizing, letterbox removal or
    probability threshold is inferred here. The caller must supply an aligned
    HxW mask. Its full nonzero extent determines the center, BEFORE subsampling.
    Use the enclosing pixel rectangle's exclusive maximum (max + 1), and the
    same half-away-from-zero center rounding as the bbox experiment.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != (image_h, image_w):
        raise ValueError("mask must be HxW in the projected image's pixel coordinates")
    if mask.dtype.kind not in "buif" or not np.all(np.isin(mask, [0, 1, 255])):
        raise ValueError("mask must be binary (bool or finite values 0, 1, 255)")
    binary = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    points = [tuple(point) for contour in contours for point in contour.reshape(-1, 2)[::4].tolist()]
    x, y, width, height = cv2.boundingRect(binary)
    center = (min((2 * x + width + 1) // 2, image_w - 1),
              min((2 * y + height + 1) // 2, image_h - 1))
    return points, center


def _clip_bbox(
    bbox: tuple[float, float, float, float], image_w: int, image_h: int
) -> tuple[float, float, float, float]:
    if image_w <= 0 or image_h <= 0:
        raise ValueError("View image dimensions must be positive")
    coordinates = np.asarray(bbox, dtype=np.float64)
    if coordinates.shape != (4,) or not np.all(np.isfinite(coordinates)):
        raise ValueError("bbox must contain four finite coordinates")
    x1, y1, x2, y2 = coordinates
    if x1 > x2 or y1 > y2:
        raise ValueError("bbox coordinates must satisfy x1 <= x2 and y1 <= y2")
    return (float(np.clip(x1, 0, image_w - 1)),
            float(np.clip(y1, 0, image_h - 1)),
            float(np.clip(x2, 0, image_w - 1)),
            float(np.clip(y2, 0, image_h - 1)))


def sample_bbox_boundary(
    bbox: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    samples_per_edge: int = 32,
) -> list[tuple[float, float]]:
    """Legacy dense sampler; NOT used by bbox_to_spherical_bbox.

    Kept for existing callers, including comparison diagnostics. Its inclusive
    endpoints, clipping and sample-count argument retain their old semantics.
    Production conversion uses _bbox_sample_pixels's eight exclusive-bbox points.
    """
    if not isinstance(samples_per_edge, (int, np.integer)) or samples_per_edge < 2:
        raise ValueError("samples_per_edge must be an integer of at least two")
    x1, y1, x2, y2 = _clip_bbox(bbox, image_w, image_h)
    xs = np.linspace(x1, x2, samples_per_edge)
    ys = np.linspace(y1, y2, samples_per_edge)
    points = []
    for x in xs:
        points.append((x, y1))
        points.append((x, y2))
    for y in ys:
        points.append((x1, y))
        points.append((x2, y))
    return points
