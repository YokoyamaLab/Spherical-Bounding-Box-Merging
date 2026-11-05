# Optional Ultralytics YOLO example

The library itself is **detector-agnostic** and does not import or install Ultralytics. This example shows how a separately installed YOLO model can provide the same `Detection` format for **bbox detection or instance segmentation**.

First [install this package](../README.md#quick-start), then install Ultralytics separately:

```bash
python -m pip install ultralytics
```

Save the code below as `user_yolo.py` in any directory, then run it with your own ERP image and local detection or segmentation weights:

```bash
python user_yolo.py /path/to/panorama.jpg /path/to/model.pt
```

```python
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from spherical_bbox_merging import Detection, ProjectionOptions, ThresholdOptions
from spherical_bbox_merging import detect_panorama


class YOLODetector:
    def __init__(self, model):
        self.model = model

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=image,
            conf=0.25,
            iou=0.45,  # Planar NMS, not spherical merging
            imgsz=640,
            device="cpu",
            verbose=False,
            retina_masks=True,
        )[0]

        shape = tuple(image.shape[:2])
        if tuple(result.orig_shape) != shape:
            raise ValueError("Detection coordinates do not match the input view")

        masks = None
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            if (tuple(result.masks.orig_shape) != shape
                    or masks.shape != (len(result.boxes), *shape)):
                raise ValueError("Masks are not aligned with the input view")
            if not np.all(np.isin(masks, [0, 1])):
                raise ValueError("Masks must be binary")
        elif self.model.task == "segment" and len(result.boxes):
            raise ValueError("Segmentation detections have no masks")

        detections = []
        for i, box in enumerate(result.boxes):
            detections.append(Detection(
                class_id=int(box.cls[0]),
                confidence=float(box.conf[0]),
                bbox=tuple(float(v) for v in box.xyxy[0].cpu().numpy()),
                mask=None if masks is None else np.array(masks[i], dtype=np.uint8, copy=True),
            ))
        return detections


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python user_yolo.py PANORAMA_IMAGE MODEL_WEIGHTS")
    image_path, weights_path = sys.argv[1:]
    if not Path(weights_path).is_file():
        raise FileNotFoundError(weights_path)

    panorama = cv2.imread(image_path)
    if panorama is None:
        raise ValueError(f"Cannot load ERP image: {image_path}")

    model = YOLO(weights_path)
    if model.task not in {"detect", "segment"}:
        raise ValueError("A detection or segmentation model is required")

    results = detect_panorama(
        panorama,
        YOLODetector(model),
        ProjectionOptions(),
        ThresholdOptions(),  # Spherical IoU=0.5, IoA=0.1
    )
    for detection in results:
        print(detection)
```

`retina_masks=True` requests masks in the original perspective-view resolution. Model weights are supplied by the user and are not distributed by this repository. The example uses CPU inference; change `device` to suit your environment.

**Third-party licensing:** Ultralytics software and weights have their own terms, including AGPL-3.0 and Enterprise licensing. Combining or distributing the resulting application may trigger additional obligations. See [Ultralytics licensing](https://www.ultralytics.com/license). The MIT license of this repository applies only to its own code, not to Ultralytics or third-party weights.
