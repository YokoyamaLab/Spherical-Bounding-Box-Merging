# Using your own detector

The pipeline works with **any planar object detector** that accepts a projected view and returns the common `Detection` format. No inheritance is required.

```python
import numpy as np
from data_types import Detection

class MyDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        # Run your model here, converting its output to Detection objects.
        height, width = image.shape[:2]
        return [Detection(
            class_id=0,
            confidence=1.0,
            bbox=(width / 4, height / 4, 3 * width / 4, 3 * height / 4),
        )]
```

The example above returns a fixed box for demonstration; it does **not** perform object recognition. For a complete runnable example, see [`main.py`](../main.py).

## Call the pipeline

```python
import cv2
from data_types import ProjectionOptions, ThresholdOptions
from pipeline import detect_panorama

panorama = cv2.imread("panorama.jpg")
if panorama is None:
    raise FileNotFoundError("panorama.jpg")

results = detect_panorama(
    panorama,
    MyDetector(),
    ProjectionOptions(),
    ThresholdOptions(),
)
for detection in results:
    print(detection)
```

Replace `MyDetector` with your model's adapter. The core handles projection, spherical bounding-box conversion, and cross-view merging; your adapter handles model-specific preprocessing, inference, confidence filtering, and planar NMS.

## Detection contract

`detect(image: np.ndarray) -> list[Detection]` receives a perspective view, normally an `H × W × 3` NumPy array. OpenCV input is typically BGR; the core does not change color channels.

| Field | Requirement |
| --- | --- |
| `class_id` | Integer class ID, consistent across views |
| `confidence` | Finite score |
| `bbox` | `(x1, y1, x2, y2)` in **input-view pixel coordinates**, not normalized or ERP coordinates |
| `mask` | Optional binary `H × W` instance mask aligned with the input view; default `None` |

Coordinates start at the top-left corner. `x2` and `y2` are **exclusive**: `(0, 0, W, H)` covers the whole image. A bbox must have positive width and height after integer conversion. The core clips sampled positions to the image.

An optional mask must have the same height and width as the input view and contain binary values (`bool`, `0/1`, or `0/255`). Restore cropped or letterboxed masks to the original view coordinates **inside your adapter**. The core does not resize masks. Empty masks fall back to bbox sampling, and a valid bbox remains required. Return `[]` for no detections.

## Output

`detect_panorama()` returns `list[SphericalDetection]`. Each element contains `class_id`, `confidence`, and `spherical_bbox`, whose fields are `yaw`, `pitch`, `horizontal_fov`, and `vertical_fov` (all **radians**). Pitch is latitude, positive toward the north pole.

For projection options and the spherical merging thresholds, see [Configuration and limitations](configuration.md).
