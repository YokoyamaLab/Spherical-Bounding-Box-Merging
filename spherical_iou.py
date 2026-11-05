
import numpy as np

from data_types import SphericalBBox

# All geometry uses unit vectors and radians.
_EPS = 1e-10


def local_axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return orthonormal (look, right, up); pitch is latitude (north positive)."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    look = np.array([cp * cy, cp * sy, sp])
    right = np.array([-sy, cy, 0.0])
    up = np.array([-sp * cy, -sp * sy, cp])
    return look, right, up


def boundary_normals(box: SphericalBBox) -> np.ndarray:
    """Four inward unit normals in the order (left, up, down, right).

    The spherical rectangle is {p in S^2 : normals @ p >= 0}.
    Geometric results are not guaranteed for FOVs outside (0, pi).
    """

    look, right, up = local_axes(box.yaw, box.pitch)
    sh, ch = np.sin(box.horizontal_fov / 2.0), np.cos(box.horizontal_fov / 2.0)
    sv, cv = np.sin(box.vertical_fov / 2.0), np.cos(box.vertical_fov / 2.0)

    left = sh * look - ch * right
    upper = sv * look - cv * up
    lower = sv * look + cv * up
    right_n = sh * look + ch * right
    return np.stack([left, upper, lower, right_n])


def plane_intersection(
    normal1: np.ndarray,
    normal2: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Unit direction of an intersection line; +/- directions lie on sphere.

    Return None for parallel or coincident origin planes.
    normal1 and normal2 are assumed to be unit vectors.
    """
    direction = np.cross(normal1, normal2)
    length = np.linalg.norm(direction)
    return None if length <= eps else direction / length


def contains_point(normals: np.ndarray, point: np.ndarray, eps: float = _EPS) -> bool:
    """Test membership in a spherical rectangle or intersection of hemispheres."""
    return bool(np.all(normals @ point >= -eps))


def spherical_rectangle_area(box: SphericalBBox) -> float:
    """Area (steradians) from Appendix A.1, algebraically simplified."""
    s = np.sin(box.horizontal_fov / 2.0) * np.sin(box.vertical_fov / 2.0)
    return float(4.0 * np.arcsin(np.clip(s, -1.0, 1.0)))


def spherical_rectangle_vertices(normals: np.ndarray) -> np.ndarray:
    """Return four corners (shape (4, 3)) of a spherical rectangle.

    Use adjacent boundary planes, choose the sign lying in all four hemispheres.
    """
    # normals order: left, up, down, right
    corners = ((0, 1), (1, 3), (3, 2), (2, 0))
    vertices = []
    for i, j in corners:
        p = plane_intersection(normals[i], normals[j])
        if p is None:
            raise ValueError("Degenerate spherical rectangle: parallel adjacent planes.")
        if contains_point(normals, p):
            vertices.append(p)
        elif contains_point(normals, -p):
            vertices.append(-p)
        else:
            raise ValueError("Failed to find a corner inside the spherical rectangle.")
    return np.stack(vertices)


def _unique_points(points: list[np.ndarray], eps: float = 1e-9) -> np.ndarray:
    """Remove duplicates using chord distance of points on the unit sphere."""
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= eps for q in unique):
            unique.append(p)
    return np.array(unique, dtype=float).reshape(-1, 3)


def _sort_spherical_vertices(points: np.ndarray) -> np.ndarray:
    """Order vertices around their mean direction, counterclockwise.

    The region must be geodesically convex and lie inside an open hemisphere.
    """
    center = np.sum(points, axis=0)
    center_norm = np.linalg.norm(center)
    if center_norm <= _EPS:
        raise ValueError("Cannot find a reference direction for sorting.")
    center /= center_norm

    aux = np.eye(3)[int(np.argmin(np.abs(center)))]
    e1 = np.cross(aux, center)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    angles = np.arctan2(points @ e2, points @ e1)
    return points[np.argsort(angles)]


def spherical_polygon_angles(points: np.ndarray) -> np.ndarray:
    """Interior angles of an ordered convex spherical polygon, in radians."""
    result = []
    n = len(points)
    for i in range(n):
        p = points[i]
        prev = points[(i - 1) % n]
        next_p = points[(i + 1) % n]

        incoming = prev - np.dot(prev, p) * p
        outgoing = next_p - np.dot(next_p, p) * p
        len1 = np.linalg.norm(incoming)
        len2 = np.linalg.norm(outgoing)
        if min(len1, len2) <= 1e-12:
            raise ValueError("Adjacent polygon vertices are identical or antipodal.")
        incoming /= len1
        outgoing /= len2

        result.append(np.arctan2(np.linalg.norm(np.cross(incoming, outgoing)),
                                 np.dot(incoming, outgoing)))
    return np.asarray(result)


def spherical_polygon_area(points: np.ndarray) -> float:
    """Area of a convex spherical polygon, using stable triangle determinants."""
    n = len(points)
    if n < 3:
        return 0.0

    p0 = points[0]
    area = 0.0

    for i in range(1, n - 1):
        a = points[i]
        b = points[i + 1]

        # det(p0, a, b) = det(p0, a-p0, b-p0). For a small triangle,
        # the differences are small directly; crossing two nearly identical
        # unit vectors first would cancel significant digits.
        numerator = abs(np.dot(p0, np.cross(a - p0, b - p0)))
        denominator = (
            1.0
            + np.dot(p0, a)
            + np.dot(a, b)
            + np.dot(b, p0)
        )

        area += 2.0 * np.arctan2(numerator, denominator)

    return float(area)


def spherical_intersection_area(box1: SphericalBBox, box2: SphericalBBox) -> float:
    """Analytic spherical area of the intersection of two spherical rectangles."""
    normals1 = boundary_normals(box1)
    normals2 = boundary_normals(box2)
    normals = np.vstack([normals1, normals2])

    corners1 = spherical_rectangle_vertices(normals1)
    corners2 = spherical_rectangle_vertices(normals2)
    points = []

    for p in corners1:
        if contains_point(normals2, p):
            points.append(p)
    for p in corners2:
        if contains_point(normals1, p):
            points.append(p)

    # Intersections of boundary planes from different rectangles.
    for n1 in normals1:
        for n2 in normals2:
            direction = plane_intersection(n1, n2)
            if direction is None:
                continue
            for p in (direction, -direction):
                if contains_point(normals, p):
                    points.append(p)

    points = _unique_points(points)
    if len(points) < 3:
        return 0.0

    ordered = _sort_spherical_vertices(points)
    area = spherical_polygon_area(ordered)
    return float(np.clip(area, 0.0, min(spherical_rectangle_area(box1),
                                         spherical_rectangle_area(box2))))


def spherical_iou(box1: SphericalBBox, box2: SphericalBBox) -> float:
    intersection = spherical_intersection_area(box1, box2)
    a1 = spherical_rectangle_area(box1)
    a2 = spherical_rectangle_area(box2)
    return float(np.clip(intersection / (a1 + a2 - intersection), 0.0, 1.0))


def spherical_ioa(box1: SphericalBBox, box2: SphericalBBox) -> float:
    """Symmetric IoA = intersection / min(area(box1), area(box2)).

    This convention measures how much of the smaller box is covered.
    """
    intersection = spherical_intersection_area(box1, box2)
    denominator = min(spherical_rectangle_area(box1), spherical_rectangle_area(box2))
    return float(np.clip(intersection / denominator, 0.0, 1.0))
