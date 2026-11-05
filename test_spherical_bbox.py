"""Python-only mathematical checks; run: python -m unittest -v test_spherical_bbox.

Fixtures are synthetic pinhole rays or prescribed spherical directions. No
projection implementation, C++ fixture, or IoU implementation is imported.
"""

from dataclasses import astuple, replace
import math
import unittest

import numpy as np

from data_types import Detection, ProjectedView, SphericalBBox
from spherical_bbox import (
    _bbox_sample_pixels, _local_basis, bbox_to_spherical_bbox, sample_bbox_boundary,
)
from spherical_candidates import to_spherical_candidates


ERP_SHAPE = (1000, 2000, 3)


def direction(yaw, pitch):
    return np.array([math.cos(pitch) * math.cos(yaw),
                     math.cos(pitch) * math.sin(yaw), math.sin(pitch)])


def view_from_rays(rays, yaw=0.0, pitch=0.0, dtype=np.float64):
    """Encode independent Cartesian fixture rays as ERP coordinate maps."""
    longitude = np.arctan2(rays[..., 1], rays[..., 0])
    latitude = np.arctan2(rays[..., 2], np.hypot(rays[..., 0], rays[..., 1]))
    h, w = rays.shape[:2]
    return ProjectedView(
        7, yaw, pitch, np.zeros((h, w, 3), dtype=np.uint8),
        (ERP_SHAPE[1] * (0.5 - longitude / math.tau)).astype(dtype),
        (ERP_SHAPE[0] * (0.5 - latitude / math.pi)).astype(dtype),
    )


def pinhole_view(yaw=0.0, pitch=0.0, step=0.02, dtype=np.float64):
    """101x101 camera: independently form rays from a focal plane at depth 1."""
    x, y = np.meshgrid((np.arange(101) - 50) * step,
                       (50 - np.arange(101)) * step)
    east = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    north = np.array([-math.sin(pitch) * math.cos(yaw),
                      -math.sin(pitch) * math.sin(yaw), math.cos(pitch)])
    rays = direction(yaw, pitch) + x[..., None] * east + y[..., None] * north
    return view_from_rays(rays, yaw, pitch, dtype)


def convert(view, bbox):
    return bbox_to_spherical_bbox(view, bbox, ERP_SHAPE)


def analytic_frame(yaw, pitch, polar):
    """Closed-form frame for each specified axis, without cross products."""
    c = direction(yaw, pitch)
    if not polar:
        return (np.array([-math.sin(yaw), math.cos(yaw), 0.]),
                np.array([-math.sin(pitch) * math.cos(yaw),
                          -math.sin(pitch) * math.sin(yaw), math.cos(pitch)]))
    x, y, z = c
    scale = math.hypot(x, z)
    return np.array([-z, 0., x]) / scale, np.array([x*y, -(x*x + z*z), y*z]) / scale


def known_angle_fixture(yaw, pitch, polar):
    """Known [-12,34] x [-18,32] degree local bounds; half center rounds to 17."""
    c0 = direction(yaw, pitch)
    right, up = analytic_frame(yaw, pitch, polar)
    u, v = np.meshgrid(np.linspace(math.tan(math.radians(-12)), math.tan(math.radians(34)), 33),
                       np.linspace(math.tan(math.radians(-18)), math.tan(math.radians(32)), 33))
    rays = c0 + u[..., None] * right + v[..., None] * up
    rays[17, 17] = c0  # Exclusive [0,33): initial center 16.5 -> 17, edge midpoint 16.
    expected = c0 + math.tan(math.radians(11)) * right + math.tan(math.radians(7)) * up
    return view_from_rays(rays), expected


def optical_box_expectation(yaw, pitch, bbox, step=.02, polar=False):
    """Analytic focal-plane extrema for boxes whose initial pixel is (50,50).

    All linear slope extrema occur at corners. This fixture uses neither ERP
    maps nor production sampling/angle helpers to derive its expected bounds.
    """
    x1, y1, x2, y2 = bbox
    assert x1 + x2 == 100 and y1 + y2 == 100
    east, north = analytic_frame(yaw, pitch, False)
    right, up = analytic_frame(yaw, pitch, polar)
    offsets = [(x - 50) * step * east + (50 - y) * step * north
               for x in [x1, x2 - 1] for y in [y1, y2 - 1]]
    horizontal = [math.atan(float(d @ right)) for d in offsets]
    vertical = [math.atan(float(d @ up)) for d in offsets]
    lo_u, hi_u = min(horizontal), max(horizontal)
    lo_v, hi_v = min(vertical), max(vertical)
    center = (direction(yaw, pitch) + math.tan((lo_u + hi_u) / 2) * right
              + math.tan((lo_v + hi_v) / 2) * up)
    return center, hi_u - lo_u, hi_v - lo_v


class SphericalBBoxTests(unittest.TestCase):
    def assert_box_valid(self, box):
        self.assertIsInstance(box, SphericalBBox)
        self.assertTrue(np.all(np.isfinite(astuple(box))))
        self.assertGreaterEqual(box.yaw, -math.pi)
        self.assertLess(box.yaw, math.pi)
        self.assertLessEqual(abs(box.pitch), math.pi / 2)
        self.assertTrue(0 <= box.horizontal_fov <= math.tau)
        self.assertTrue(0 <= box.vertical_fov <= math.tau)

    def assert_center(self, box, expected, atol=2e-12):
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(direction(box.yaw, box.pitch), expected, atol=atol, rtol=0)

    def test_equatorial_center_and_fov_with_exclusive_endpoints(self):
        bbox = (30, 35, 70, 65)
        box = convert(pinhole_view(), bbox)
        center, hfov, vfov = optical_box_expectation(0, 0, bbox)
        self.assert_center(box, center)
        self.assertAlmostEqual(hfov, math.atan(.4) + math.atan(.38), places=12)
        self.assertAlmostEqual(vfov, math.atan(.3) + math.atan(.28), places=12)
        self.assertAlmostEqual(box.horizontal_fov, hfov, places=12)
        self.assertAlmostEqual(box.vertical_fov, vfov, places=12)
        self.assert_box_valid(box)

    def test_high_latitude_center_and_fov(self):
        bbox = (30, 35, 70, 65)
        for yaw in [0., .7, 2.1]:
            for pitch_deg in [-85, -75, 75, 85]:
                with self.subTest(yaw=yaw, pitch=pitch_deg):
                    pitch = math.radians(pitch_deg)
                    box = convert(pinhole_view(yaw, pitch), bbox)
                    center, hfov, vfov = optical_box_expectation(
                        yaw, pitch, bbox, polar=abs(math.sin(pitch)) >= .99)
                    self.assert_center(box, center)
                    self.assertAlmostEqual(box.horizontal_fov, hfov, places=12)
                    self.assertAlmostEqual(box.vertical_fov, vfov, places=12)

    def test_equation_1_strict_axis_threshold(self):
        for sign in [-1, 1]:
            for z in [np.nextafter(.99, 0.), .99, np.nextafter(.99, 1.)]:
                for yaw in [.0, .8, 2.2]:
                    with self.subTest(sign=sign, z=z, yaw=yaw):
                        radius = math.sqrt(1 - z*z)
                        c0 = np.array([radius * math.cos(yaw), radius * math.sin(yaw), sign*z])
                        right, up = _local_basis(c0)
                        expected_r, expected_u = analytic_frame(yaw, math.asin(sign*z), z >= .99)
                        np.testing.assert_allclose(right, expected_r, atol=2e-15, rtol=0)
                        np.testing.assert_allclose(up, expected_u, atol=2e-15, rtol=0)
                        frame = np.stack([c0, right, up])
                        np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=2e-15, rtol=0)

    def test_equations_2_to_6_before_and_after_axis_threshold(self):
        for z in [-.990001, -.989999, 0., .989999, .990001]:
            with self.subTest(z=z):
                view, center = known_angle_fixture(.8, math.asin(z), abs(z) >= .99)
                box = convert(view, (0, 0, 33, 33))
                self.assert_center(box, center)
                self.assertAlmostEqual(box.horizontal_fov, math.radians(46), places=12)
                self.assertAlmostEqual(box.vertical_fov, math.radians(50), places=12)

    def test_north_and_south_pole_center_directions(self):
        for sign in [-1, 1]:
            rays = np.broadcast_to([0., 0., float(sign)], (5, 5, 3)).copy()
            with self.subTest(sign=sign), np.errstate(all="raise"):
                box = convert(view_from_rays(rays), (0, 0, 5, 5))
                self.assert_center(box, np.array([0., 0., float(sign)]))
                self.assertEqual(box.horizontal_fov, 1e-6)
                self.assertEqual(box.vertical_fov, 1e-6)
                self.assert_box_valid(box)

    def test_polar_fov_matches_negative_y_reference(self):
        bbox = (26, 39, 74, 61)
        for sign in [-1, 1]:
            for yaw in [0., .2, 1.3, 2.8]:
                with self.subTest(sign=sign, yaw=yaw), np.errstate(all="raise"):
                    pitch = sign * math.pi / 2
                    box = convert(pinhole_view(yaw, pitch), bbox)
                    center, hfov, vfov = optical_box_expectation(yaw, pitch, bbox, polar=True)
                    self.assert_center(box, center)
                    self.assertAlmostEqual(box.horizontal_fov, hfov, places=12)
                    self.assertAlmostEqual(box.vertical_fov, vfov, places=12)

    def test_eight_pixels_even_bbox(self):
        points, center = _bbox_sample_pixels((288, 288, 352, 352), 640, 640)
        self.assertEqual(points, [(288, 288), (351, 288), (351, 351), (288, 351),
                                  (319, 288), (319, 351), (288, 319), (351, 319)])
        self.assertEqual(center, (320, 320))

    def test_eight_pixels_odd_bbox_and_center_half_rounding(self):
        points, center = _bbox_sample_pixels((10, 20, 15, 27), 101, 101)
        self.assertEqual(points, [(10, 20), (14, 20), (14, 26), (10, 26),
                                  (12, 20), (12, 26), (10, 23), (14, 23)])
        self.assertEqual(center, (13, 24))  # 12.5->13 and 23.5->24; never ties-to-even.
        self.assertEqual(_bbox_sample_pixels((10, 20, 16, 28), 101, 101)[1], (13, 24))

    def test_float_endpoints_truncate_toward_zero_before_sampling(self):
        self.assertEqual(_bbox_sample_pixels((10.9, 20.9, 16.9, 28.9), 101, 101),
                         _bbox_sample_pixels((10, 20, 16, 28), 101, 101))
        points, center = _bbox_sample_pixels((-2.9, -3.9, 5.9, 6.9), 101, 101)
        self.assertEqual(center, (2, 2))
        self.assertEqual(points, [(0, 0), (4, 0), (4, 5), (0, 5),
                                  (2, 0), (2, 5), (0, 2), (4, 2)])
        view = pinhole_view()
        self.assertEqual(convert(view, (10.9, 20.9, 16.9, 28.9)), convert(view, (10, 20, 16, 28)))

    def test_clip_corners_before_midpoints_but_keep_original_center(self):
        points, center = _bbox_sample_pixels((-10, -20, 80, 90), 101, 101)
        self.assertEqual(center, (35, 35))
        self.assertEqual(points, [(0, 0), (79, 0), (79, 89), (0, 89),
                                  (39, 0), (39, 89), (0, 44), (79, 44)])
        self.assertNotEqual(center, _bbox_sample_pixels((0, 0, 80, 90), 101, 101)[1])
        self.assertEqual(_bbox_sample_pixels((-10, -12, -5, -7), 101, 101), ([(0, 0)] * 8, (0, 0)))

    def test_small_bboxes_and_image_edges(self):
        view = pinhole_view()
        for x, y in [(0, 0), (50, 50), (100, 100)]:
            with self.subTest(x=x, y=y):
                points, center = _bbox_sample_pixels((x, y, x+1, y+1), 101, 101)
                self.assertEqual(points, [(x, y)] * 8)
                self.assertEqual(center, (min(x+1, 100), min(y+1, 100)))
                box = convert(view, (x, y, x+1, y+1))
                self.assert_center(box, np.array([1., (x-50)*.02, (50-y)*.02]))
                self.assertEqual(box.horizontal_fov, 1e-6)
                self.assertEqual(box.vertical_fov, 1e-6)
        points, center = _bbox_sample_pixels((0, 0, 101, 101), 101, 101)
        self.assertEqual(points, [(0, 0), (100, 0), (100, 100), (0, 100),
                                  (50, 0), (50, 100), (0, 50), (100, 50)])
        self.assertEqual(center, (51, 51))
        self.assert_box_valid(convert(view, (0, 0, 101, 101)))
        one_pixel = view_from_rays(np.array([[[1., 0., 0.]]]))
        self.assertEqual(convert(one_pixel, (0, 0, 1, 1)), SphericalBBox(0., 0., 1e-6, 1e-6))

    def test_right_and_bottom_exclusive_pixels_are_not_sampled(self):
        rays = np.broadcast_to([1., 0., 0.], (10, 10, 3)).copy()
        rays[9, :] = direction(.9, .5)
        rays[:, 9] = direction(.9, .5)
        self.assertEqual(convert(view_from_rays(rays), (0, 0, 9, 9)), SphericalBBox(0., 0., 1e-6, 1e-6))

    def test_only_corners_and_edge_midpoints_affect_fov(self):
        rays = np.broadcast_to([1., 0., 0.], (9, 9, 3)).copy()
        rays[0, 2] = direction(math.radians(60), 0)  # Not one of the eight samples.
        rays[0, 4] = direction(math.radians(30), 0)  # Top midpoint.
        box = convert(view_from_rays(rays), (0, 0, 9, 9))
        self.assertAlmostEqual(box.yaw, math.radians(15), places=12)
        self.assertAlmostEqual(box.horizontal_fov, math.radians(30), places=12)
        self.assertAlmostEqual(box.vertical_fov, 1e-6, places=12)

    def test_longitude_seam(self):
        bbox = (30, 35, 70, 65)
        for yaw in [math.pi, -math.pi, math.pi - 1e-7, -math.pi + 1e-7]:
            with self.subTest(yaw=yaw):
                box = convert(pinhole_view(yaw), bbox)
                center, hfov, vfov = optical_box_expectation(yaw, 0, bbox)
                self.assert_center(box, center)
                self.assertAlmostEqual(box.horizontal_fov, hfov, places=12)
                self.assertAlmostEqual(box.vertical_fov, vfov, places=12)

    def test_yaw_periodicity_in_maps_and_metadata(self):
        bbox = (16, 28, 62, 74)
        for pitch in [0., .7, 1.5, -1.5, math.pi/2, -math.pi/2]:
            for turns in [-5, -1, 1, 6]:
                with self.subTest(pitch=pitch, turns=turns):
                    original = pinhole_view(2.9, pitch)
                    reference = convert(original, bbox)
                    periodic = pinhole_view(2.9 + turns * math.tau, pitch)
                    np.testing.assert_allclose(astuple(convert(periodic, bbox)), astuple(reference), atol=2e-12, rtol=0)
                    translated = replace(original, map_x=original.map_x + turns * ERP_SHAPE[1])
                    np.testing.assert_allclose(astuple(convert(translated, bbox)), astuple(reference), atol=2e-12, rtol=0)

    def test_initial_center_uses_map_not_view_metadata(self):
        view = pinhole_view(.7, .5)
        bbox = (60, 10, 94, 48)
        self.assertEqual(convert(view, bbox), convert(replace(view, yaw=-2.2, pitch=-1.4), bbox))

    def test_float32_maps(self):
        for yaw, pitch in [(0., 0.), (math.pi, .6), (1.2, 1.5), (0., math.pi / 2)]:
            with self.subTest(yaw=yaw, pitch=pitch):
                bbox = (20, 30, 80, 70)
                expected = convert(pinhole_view(yaw, pitch), bbox)
                actual = convert(pinhole_view(yaw, pitch, dtype=np.float32), bbox)
                self.assert_box_valid(actual)
                self.assert_center(actual, direction(expected.yaw, expected.pitch), atol=1e-6)
                np.testing.assert_allclose(astuple(actual)[2:], astuple(expected)[2:], atol=1e-6, rtol=0)

    def test_tiny_and_large_angular_extents(self):
        for step in [1e-10, .02, 1., 1e6]:
            for bbox in [(49, 49, 51, 51), (30, 35, 70, 65), (0, 0, 100, 100)]:
                with self.subTest(step=step, bbox=bbox), np.errstate(all="raise"):
                    box = convert(pinhole_view(step=step), bbox)
                    center, hfov, vfov = optical_box_expectation(0, 0, bbox, step)
                    self.assert_center(box, center)
                    self.assertAlmostEqual(box.horizontal_fov, max(1e-6, hfov), places=11)
                    self.assertAlmostEqual(box.vertical_fov, max(1e-6, vfov), places=11)
                    self.assert_box_valid(box)

    def test_random_views_are_finite_and_periodic(self):
        rng = np.random.default_rng(20261021)
        for index in range(100):
            yaw, pitch = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi/2, math.pi/2)
            x1, x2 = sorted(rng.choice(102, 2, replace=False))
            y1, y2 = sorted(rng.choice(102, 2, replace=False))
            bbox = (x1, y1, x2, y2)
            with self.subTest(index=index), np.errstate(all="raise"):
                original = convert(pinhole_view(yaw, pitch), bbox)
                periodic = convert(pinhole_view(yaw + math.tau, pitch), bbox)
                self.assert_box_valid(original)
                self.assert_box_valid(periodic)
                np.testing.assert_allclose(astuple(original), astuple(periodic), atol=2e-12, rtol=0)

    def test_legacy_dense_sampler_interface_is_preserved(self):
        points = sample_bbox_boundary((-1, -2, 8, 9), 7, 8)
        self.assertEqual(len(points), 128)
        for corner in [(0, 0), (6, 0), (0, 7), (6, 7)]:
            self.assertEqual(points.count(corner), 2)
        self.assertEqual(len(sample_bbox_boundary((0, 0, 6, 7), 7, 8, 3)), 12)
        for count in [0, 1, -1, 2.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                sample_bbox_boundary((0, 0, 4, 4), 10, 10, count)

    def test_invalid_and_empty_integer_bboxes(self):
        view = pinhole_view()
        for bbox in [(3, 0, 2, 4), (0, 5, 4, 3), (0, 0, float("nan"), 4),
                     (0, 0, 4, float("inf")), (0, 1, 2), (50, 50, 50, 50),
                     (40, 50, 60, 50), (50.1, 40, 50.9, 60)]:
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                convert(view, bbox)
        for shape in [(), (0, 2000), (1000, -1), (float("nan"), 2000)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                bbox_to_spherical_bbox(view, (0, 0, 4, 4), shape)
        with self.assertRaises(ValueError):
            convert(replace(view, map_x=np.zeros((2, 2))), (0, 0, 4, 4))
        with self.assertRaises(ValueError):
            convert(replace(view, image=np.zeros((0, 0)), map_x=np.zeros((0, 0)),
                            map_y=np.zeros((0, 0))), (0, 0, 4, 4))
        for field in ["map_x", "map_y"]:
            for pixel in [(50, 50), (40, 40)]:
                for invalid in [float("nan"), float("inf")]:
                    bad_map = getattr(view, field).copy()
                    bad_map[pixel] = invalid
                    with self.subTest(field=field, pixel=pixel, value=invalid), self.assertRaises(ValueError):
                        convert(replace(view, **{field: bad_map}), (40, 40, 60, 60))

    def test_existing_candidate_interface(self):
        view = pinhole_view()
        detection = Detection(3, .85, (30, 35, 70, 65))
        candidates = to_spherical_candidates(view, [detection], ERP_SHAPE)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].view_ids, {7})
        self.assertEqual(candidates[0].detection.class_id, 3)
        self.assertEqual(candidates[0].detection.confidence, .85)
        self.assertEqual(candidates[0].detection.spherical_bbox, convert(view, detection.bbox))


if __name__ == "__main__":
    unittest.main()
