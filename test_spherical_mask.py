"""Independent mask-coordinate and conversion checks; no C++ implementation."""
from dataclasses import astuple
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from spherical_bbox_merging import Detection
from spherical_bbox_merging import spherical_bbox as sb
from spherical_bbox_merging.spherical_candidates import to_spherical_candidates
from test_spherical_bbox import ERP_SHAPE, pinhole_view


class MaskConversionTests(unittest.TestCase):
    def setUp(self):
        self.view = pinhole_view()
        self.bbox = (30, 30, 71, 71)
        self.mask = np.zeros((101, 101), dtype=np.uint8)

    def convert(self, mask, bbox=None):
        return sb.bbox_to_spherical_bbox(self.view, self.bbox if bbox is None else bbox,
                                         ERP_SHAPE, mask=mask)

    def test_empty_mask_falls_back_to_bbox(self):
        self.assertEqual(self.convert(self.mask), self.convert(None))
        self.assertEqual(self.convert(self.mask, (50, 50, 51, 51)).horizontal_fov, 1e-6)

    def test_invalid_bbox_still_rejected_with_mask(self):
        self.mask[30:71, 30:71] = 1
        for mask in [self.mask, np.zeros_like(self.mask), None]:
            for bbox in [(30, 30, 30, 71), (50, 50, 30, 71), (30.1, 30, 30.9, 71)]:
                with self.subTest(bbox=bbox, nonempty=mask is self.mask):
                    with self.assertRaises(ValueError):
                        self.convert(mask, bbox)

    def test_four_point_stride_and_exclusive_center(self):
        self.mask[20:25, 10:15] = 1
        points, center = sb._mask_sample_pixels(self.mask, 101, 101)
        self.assertEqual(points, [(10, 20), (10, 24), (14, 24), (14, 20)])
        self.assertEqual(center, (13, 23))  # Bounding rectangle [10,20,15,25).

    def test_rectangle_known_center_and_fov(self):
        self.mask[30:71, 30:71] = 1  # Side length 40: every corner survives stride 4.
        expected = self.convert(None)
        self.assertEqual(self.convert(self.mask), expected)
        self.assertGreater(expected.horizontal_fov, 0.7)
        self.assertLess(expected.horizontal_fov, 0.9)

    def test_triangle_and_offset_use_mask_not_bbox(self):
        cv2.fillPoly(self.mask, [np.array([[30, 30], [70, 35], [40, 70]])], 1)
        triangle = self.convert(self.mask)
        self.mask[:] = 0
        cv2.circle(self.mask, (37, 37), 7, 1, -1)
        offset = self.convert(self.mask)
        self.assertNotEqual(triangle, self.convert(None))
        self.assertNotEqual(offset, triangle)
        self.assertLess(offset.horizontal_fov, self.convert(None).horizontal_fov / 2)
        self.assertLess(offset.yaw, -0.1)
        self.assertGreater(offset.pitch, 0.1)

    def test_same_mask_independent_of_valid_bbox(self):
        self.mask[35:48, 31:46] = 1
        self.assertEqual(self.convert(self.mask), self.convert(self.mask, (0, 0, 101, 101)))

    def test_external_contour_ignores_holes(self):
        self.mask[30:71, 30:71] = 1
        filled = self.convert(self.mask)
        self.mask[40:61, 40:61] = 0
        self.assertEqual(self.convert(self.mask), filled)

    def test_stride_resets_for_each_external_component(self):
        self.mask[10:13, 10:14] = 1  # Perimeter 10, not a multiple of 4.
        self.mask[60:63, 70:74] = 1
        points, center = sb._mask_sample_pixels(self.mask, 101, 101)
        self.assertEqual(len(points), 6)
        self.assertEqual(sum(x < 20 for x, y in points), 3)
        self.assertEqual(sum(x > 60 for x, y in points), 3)
        self.assertEqual(center, (42, 37))

    def test_center_uses_full_extent_before_decimation(self):
        self.mask[20, 10:13] = 1  # Contour [10,11,12,11], only 10 sampled.
        points, center = sb._mask_sample_pixels(self.mask, 101, 101)
        self.assertEqual(points, [(10, 20)])
        self.assertEqual(center, (12, 21))

    def test_single_pixel_positive_fov_and_last_pixel_center_clip(self):
        self.mask[100, 100] = 1
        points, center = sb._mask_sample_pixels(self.mask, 101, 101)
        self.assertEqual((points, center), ([(100, 100)], (100, 100)))
        result = self.convert(self.mask)
        self.assertEqual((result.horizontal_fov, result.vertical_fov), (1e-6, 1e-6))

    def test_binary_encodings_and_noncontiguous_masks(self):
        self.mask[30:71, 30:71] = 1
        original = self.mask.copy()
        expected = self.convert(self.mask)
        for mask in [self.mask.astype(bool), self.mask * 255, self.mask.astype(float), self.mask.T]:
            self.assertEqual(self.convert(mask), expected)
        np.testing.assert_array_equal(self.mask, original)

    def test_invalid_resolution_rank_and_values_rejected(self):
        bad_masks = [np.zeros((100, 101)), np.zeros((101, 101, 1)),
                     np.zeros((1, 101, 101)), np.full((101, 101), 0.5),
                     np.full((101, 101), np.nan), np.full((101, 101), np.inf),
                     np.full((101, 101), -1), np.full((101, 101), '0')]
        for mask in bad_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    self.convert(mask)

    def test_candidates_forward_mask_and_preserve_detection_metadata(self):
        self.mask[30:41, 30:41] = 1
        detection = Detection(3, .75, self.bbox, self.mask)
        result = to_spherical_candidates(self.view, [detection], ERP_SHAPE)[0]
        self.assertEqual(result.detection.spherical_bbox, self.convert(self.mask))
        self.assertEqual((result.detection.class_id, result.detection.confidence, result.view_ids),
                         (3, .75, {7}))

    def test_mask_and_bbox_share_local_math(self):
        self.mask[30:71, 30:71] = 1
        samples = sb._mask_sample_pixels(self.mask, 101, 101)
        for pitch in [0, 1.3, math.pi / 2, -math.pi / 2]:
            self.view = pinhole_view(yaw=.4, pitch=pitch)
            mask_result = self.convert(self.mask)
            with patch.object(sb, '_bbox_sample_pixels', return_value=samples):
                self.assertEqual(self.convert(None), mask_result)
            self.assertTrue(np.isfinite(astuple(mask_result)).all())


if __name__ == '__main__':
    unittest.main()
