import unittest
import numpy as np

from data_types import ProjectionOptions
from projection import split_erp


class TestProjectionOptions(unittest.TestCase):

    def test_invalid_options(self):
        image = np.zeros((64, 128, 3), dtype=np.uint8)

        invalid_options = [
            ProjectionOptions(cut_angle_deg=0),
            ProjectionOptions(cut_angle_deg=-60),
            ProjectionOptions(cut_angle_deg=70),
            ProjectionOptions(cut_fov_deg=0),
            ProjectionOptions(cut_fov_deg=180),
            ProjectionOptions(cut_fov_deg=210),
            ProjectionOptions(cut_fov_deg=float("nan")),
            ProjectionOptions(width=0),
            ProjectionOptions(height=-1),
        ]

        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    split_erp(image, options)