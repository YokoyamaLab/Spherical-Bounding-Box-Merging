"""Model-free detector contract and pre-refactor pipeline regressions."""
from dataclasses import astuple
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data_types import Detection, ProjectionOptions, ThresholdOptions
from detector import DetectorProtocol
import pipeline
from projection import split_erp


class SyntheticDetector:
    """No inheritance, model metadata, inference library, or weights required."""

    def __init__(self, use_masks=False):
        self.use_masks = use_masks
        self.images = []
        self.outputs = []

    def detect(self, image: np.ndarray) -> list[Detection]:
        index = len(self.images)
        self.images.append(image)
        rows = {
            1: [(7, .7, (4, 3, 44, 29)), (3, .9, (12, 8, 28, 24))],
            2: [(7, .8, (2, 4, 40, 30))],
        }.get(index, [])
        detections = []
        for class_id, confidence, bbox in rows:
            mask = None
            if self.use_masks:
                mask = np.zeros(image.shape[:2], np.uint8)
                x1, y1, x2, y2 = bbox
                # Deliberately different from bbox: catches silent bbox fallback.
                mask[y1 + 1:y2 - 1, x1 + 2:x2 - 2] = 1
            detections.append(Detection(class_id, confidence, bbox, mask))
        self.outputs.append(detections)
        return detections


class DetectorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.erp = np.zeros((64, 128, 3), np.uint8)
        self.projection = ProjectionOptions(width=48, height=32)
        self.thresholds = ThresholdOptions(.2, .2)

    def assert_baseline(self, use_masks, expected_boxes):
        detector: DetectorProtocol = SyntheticDetector(use_masks)
        results = pipeline.detect_panorama(self.erp, detector, self.projection, self.thresholds)
        self.assertEqual(len(detector.images), 14)
        self.assertTrue(all(image.shape == (32, 48, 3) for image in detector.images))
        self.assertEqual([(d.class_id, d.confidence) for d in results], [(7, .8), (3, .9)])
        # Values captured from the original pipeline before the interface change.
        # Three planar detections in two overlapping views merge to two results.
        np.testing.assert_allclose([astuple(d.spherical_bbox) for d in results],
                                   expected_boxes, atol=1e-12, rtol=0)

    def test_bbox_only_matches_pre_refactor_conversion_and_merge(self):
        self.assert_baseline(False, [
            (0.5735222154498238, -0.5681893082454585, 2.5621480128321243, 1.1514116907980365),
            (0.20389984343418988, -0.5119691391005516, 0.6896429895169316, 0.6975228962398909),
        ])

    def test_bbox_and_mask_match_pre_refactor_conversion_and_merge(self):
        self.assert_baseline(True, [
            (0.5769193884009596, -0.5627577782850677, 2.4598103837847365, 1.072055706950286),
            (0.21430242396603738, -0.5045452957478012, 0.5142608260093313, 0.6018199561535498),
        ])

    def test_empty_detector_still_receives_every_view(self):
        class EmptyDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, image: np.ndarray) -> list[Detection]:
                self.calls += 1
                return []

        detector = EmptyDetector()
        self.assertEqual(pipeline.detect_panorama(
            self.erp, detector, self.projection, self.thresholds), [])
        self.assertEqual(detector.calls, 14)

    def test_view_detection_and_candidate_order_and_masks_are_preserved(self):
        views = split_erp(self.erp, self.projection)
        detector = SyntheticDetector(use_masks=True)
        with patch.object(pipeline, "split_erp", return_value=views), \
             patch.object(pipeline, "to_spherical_candidates", wraps=pipeline.to_spherical_candidates) as convert, \
             patch.object(pipeline, "merge", wraps=pipeline.merge) as merge:
            pipeline.detect_panorama(self.erp, detector, self.projection, self.thresholds)
        for view, image, detections, call in zip(
                views, detector.images, detector.outputs, convert.call_args_list, strict=True):
            self.assertIs(image, view.image)
            self.assertIs(call.args[0], view)
            self.assertIs(call.args[1], detections)
            self.assertEqual(call.args[2], self.erp.shape[:2])
        candidates, thresholds = merge.call_args.args
        self.assertIs(thresholds, self.thresholds)
        self.assertEqual([(c.detection.class_id, c.detection.confidence, c.view_ids) for c in candidates],
                         [(7, .7, {1}), (3, .9, {1}), (7, .8, {2})])
        merge.assert_called_once()

    def test_extracted_core_imports_and_runs_with_model_imports_blocked(self):
        # Copy exactly the public modules: imports cannot accidentally rely on
        # research/, the experiment runner, test fixtures, or the repository.
        modules = ("data_types", "detector", "pipeline", "projection", "spherical_candidates",
                   "spherical_bbox", "spherical_iou", "merge", "main")
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            for name in modules:
                shutil.copyfile(root / f"{name}.py", Path(temporary) / f"{name}.py")
            code = '''
import importlib
import importlib.abc
import sys
import typing
sys.path.insert(0, sys.argv[1])
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'ultralytics', 'research'}:
            raise AssertionError('Unexpected core dependency: ' + fullname)
sys.meta_path.insert(0, BlockModels())
for name in sys.argv[2:]:
    importlib.import_module(name)
import numpy as np
from data_types import Detection, ProjectionOptions, ThresholdOptions
from detector import DetectorProtocol
from pipeline import detect_panorama
class EmptyDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return []
assert typing.get_type_hints(detect_panorama)['detector'] is DetectorProtocol
assert detect_panorama(np.zeros((16,32,3), np.uint8), EmptyDetector(),
                       ProjectionOptions(width=16,height=16), ThresholdOptions()) == []
assert not {'torch', 'torchvision', 'ultralytics', 'research'} & sys.modules.keys()
'''
            result = subprocess.run([sys.executable, "-I", "-c", code, temporary, *modules],
                                    cwd=temporary, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
