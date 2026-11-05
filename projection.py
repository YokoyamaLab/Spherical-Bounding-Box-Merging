import cv2
import numpy as np

from data_types import ProjectionOptions, ProjectedView


def split_erp(
    panorama: np.ndarray,
    options: ProjectionOptions,
) -> list[ProjectedView]:

    if options.cut_angle_deg <= 0 or 180 % options.cut_angle_deg != 0:
        raise ValueError("cut_angle_deg must be a positive divisor of 180.")

    if not (0 < options.cut_fov_deg < 180):
        raise ValueError("cut_fov_deg must be in (0, 180).")

    if options.width <= 0 or options.height <= 0:
        raise ValueError("Projected view dimensions must be positive.")

    panorama_height, panorama_width = panorama.shape[:2]

    directions = _generate_directions(options.cut_angle_deg)

    projected_views = []

    for view_id, (yaw, pitch) in enumerate(directions):

        map_x, map_y = _make_map(
            panorama_width=panorama_width,
            panorama_height=panorama_height,
            yaw=yaw,
            pitch=pitch,
            fov_deg=options.cut_fov_deg,
            width=options.width,
            height=options.height,
        )

        image = cv2.remap(
            panorama,
            map_x,
            map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        projected_views.append(
            ProjectedView(
                view_id=view_id,
                yaw=yaw,
                pitch=pitch,
                image=image,
                map_x=map_x,
                map_y=map_y,
            )
        )

    return projected_views


def _generate_directions(
    cut_angle_deg: int,
) -> list[tuple[float, float]]:
    """
    Generate (yaw, pitch) pairs in radians.

    Latitude directions run from south to north, matching the research
    C++ panorama_cut.cpp ordering (including view_id assignment).

    Longitude directions are sampled from 0 to <360 degrees.

    At the north and south poles, longitude does not matter,
    so only one view is generated.
    """

    directions = []

    for pitch_deg in range(-90, 91, cut_angle_deg):

        # North / South pole
        if abs(pitch_deg) == 90:
            directions.append(
                (
                    0.0,
                    np.deg2rad(pitch_deg),
                )
            )
            continue

        # Other latitudes
        for yaw_deg in range(0, 360, cut_angle_deg):
            directions.append(
                (
                    np.deg2rad(yaw_deg),
                    np.deg2rad(pitch_deg),
                )
            )

    return directions


def _make_map(
    panorama_width: int,
    panorama_height: int,
    yaw: float,
    pitch: float,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:

    # Match pre_eq2per's float camera geometry and diagonal FOV.
    fov = np.float32(fov_deg) * np.float32(np.pi) / np.float32(180)

    # diagonal FOV
    radius = np.tan(fov / np.float32(2))
    # C++ sqrt(int) computes in double before assigning the scale to float.
    scale = np.float32(float(radius) / np.sqrt(width**2 + height**2))

    view_width = np.float32(width) * scale
    view_height = np.float32(height) * scale

    # C++ samples index - size/2 with integer division; the positive edge is
    # excluded. For even sizes the optical axis lies on pixel (W/2, H/2).
    x = (np.arange(width, dtype=np.float32) - width // 2) * 2 * view_width / width
    y = (np.arange(height, dtype=np.float32) - height // 2) * 2 * view_height / height

    x, y = np.meshgrid(x, y)

    # Use the research implementation's sphere construction, including its
    # limit at r=0 and the subtraction order used to recover z.
    r = np.sqrt(x * x + y * y)
    factor = np.ones_like(r)
    np.divide(np.sin(np.arctan(r)), r, out=factor, where=r > np.float32(1e-8))
    x = x * factor
    y = y * factor
    z = np.sqrt(np.maximum(np.float32(0), np.float32(1) - x * x - y * y))
    xyz = np.stack((x.ravel(), y.ravel(), z.ravel()))

    # eq2per receives float angles even though the view metadata uses doubles.
    yaw = np.float32(yaw)
    pitch = np.float32(pitch)

    rx = np.array([
        [1, 0, 0],
        [0, np.sin(pitch), np.cos(pitch)],
        [0, -np.cos(pitch), np.sin(pitch)],
    ], dtype=np.float32)

    rz = np.array([
        [np.sin(yaw), np.cos(yaw), 0],
        [-np.cos(yaw), np.sin(yaw), 0],
        [0, 0, 1],
    ], dtype=np.float32)

    # Preserve the two separate OpenCV matrix products used by eq2per.
    xyz = cv2.gemm(rx, xyz, 1, None, 0)
    xyz = cv2.gemm(rz, xyz, 1, None, 0)

    # Sphere -> longitude / latitude
    lon_map = np.arctan2(
        xyz[1],
        xyz[0],
    )

    lat_map = np.arcsin(xyz[2])

    # longitude / latitude -> ERP pixel coordinate
    map_x = panorama_width * (
        0.5 - lon_map.astype(np.float64) / (2.0 * np.pi)
    )

    map_y = panorama_height * (
        0.5 - lat_map.astype(np.float64) / np.pi
    )

    # Keep the raw coordinates as C++ does. cv2.remap handles both horizontal
    # and vertical wrapping through BORDER_WRAP in split_erp.

    return (
        map_x.astype(np.float32).reshape(height, width),
        map_y.astype(np.float32).reshape(height, width),
    )
