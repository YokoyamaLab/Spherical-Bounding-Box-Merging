# Configuration and limitations

After [installing the package](../README.md#quick-start), pass `ProjectionOptions` and `ThresholdOptions` to `detect_panorama()`. Both option objects are required arguments; constructing them without arguments selects the defaults below:

```python
from spherical_bbox_merging import ProjectionOptions, ThresholdOptions

projection = ProjectionOptions()
thresholds = ThresholdOptions()
```

## Defaults

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `ProjectionOptions.cut_angle_deg` | `60` | Angular spacing between view centers (must divide 180) |
| `ProjectionOptions.cut_fov_deg` | `105.0` | **Diagonal** field of view of each perspective view, in degrees |
| `ProjectionOptions.width` | `640` | Projected view width in pixels |
| `ProjectionOptions.height` | `640` | Projected view height in pixels |
| `ThresholdOptions.iou_threshold` | **`0.5`** | First-stage spherical IoU clustering threshold |
| `ThresholdOptions.ioa_threshold` | **`0.1`** | Second-stage spherical IoA clustering threshold |

The default projection creates **14 views**; using `cut_angle_deg=90` creates **6 views**. The **0.5 / 0.1 spherical merging thresholds** match the settings reported in the SNPD 2026 paper. They are different from a detector's own *planar NMS* threshold (e.g., `0.45` in the [YOLO example](ultralytics.md)).

Merging runs independently per class, first with spherical IoU and then with spherical IoA. IoA is the intersection area divided by the **smaller** box area. Detections from the same source view are not merged together. The final box encloses its merged cluster.

## Important limitations

- Spherical IoU and IoA calculations are intended for boxes with horizontal and vertical FOVs strictly between 0 and 180 degrees. Merging may produce boxes outside this range; subsequent overlap calculations may return inaccurate results or raise exceptions. Behavior for such boxes is not guaranteed.
- The detector must return bboxes and masks in the original perspective-view pixel coordinates. The library does not undo network letterboxing or resize predicted masks.
- This repository contains the core implementation and synthetic tests, **not** the original experimental evaluation environment, pretrained weights, or datasets.

## Tests

Tests are included in the repository checkout, not in the installed package. Follow the [local development setup](../README.md#local-development-and-tests), then run from the repository root:

```bash
python -m unittest -v test_projection_options test_detector_protocol test_spherical_bbox test_spherical_mask
```

The tests do not require YOLO, a GPU, or model weights.
