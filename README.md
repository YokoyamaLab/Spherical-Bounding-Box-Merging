# Spherical Bounding Box Merging

**360° object detection with your choice of planar object detector.**

This Python implementation applies a planar detector to an equirectangular panorama (ERP) by:

1. Projecting the panorama into perspective views.
2. Running a **user-provided detector** on each view.
3. Converting detections into spherical bounding boxes.
4. Merging overlapping boxes across views using spherical IoU and IoA.

It supports bounding-box detections and instance masks. **The core does not depend on YOLO, Ultralytics, or PyTorch.**

## Quick start

Requires Python 3.10+.

### Installation

Use a virtual environment for installation. For example, on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1` instead.

Install directly from GitHub (Git is required):

```bash
python -m pip install "git+https://github.com/YokoyamaLab/Spherical-Bounding-Box-Merging.git"
```

The distribution name is `spherical-bounding-box-merging`; the import name is `spherical_bbox_merging`. Runtime dependencies are NumPy and OpenCV (`opencv-python-headless`), installed automatically. This package is not published on PyPI.

### Minimal example

After installation, the following example runs from any directory, without model weights or an input image:

```python
import numpy as np

from spherical_bbox_merging import (
    detect_panorama,
    Detection,
    ProjectionOptions,
    ThresholdOptions,
)


class CenterBoxDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        return [Detection(0, 1.0, (width / 4, height / 4, 3 * width / 4, 3 * height / 4))]


panorama = np.zeros((64, 128, 3), dtype=np.uint8)
results = detect_panorama(
    panorama,
    CenterBoxDetector(),
    ProjectionOptions(width=64, height=64),
    ThresholdOptions(),
)
for detection in results:
    print(detection)
```

This synthetic detector returns a fixed box; it does not perform object recognition. The small projection size keeps the example fast. To use a real model, provide an object with a `detect(image) -> list[Detection]` method. See [Using your own detector](docs/detector.md).

`detect_panorama()` returns `list[SphericalDetection]`. `DetectorProtocol`, `SphericalBBox`, `SphericalDetection`, `ProjectedView`, and `SphericalCandidate` can also be imported directly from `spherical_bbox_merging`.

If you used the earlier repository layout, update imports such as `from data_types import Detection` and `from pipeline import detect_panorama` to use `from spherical_bbox_merging import ...`. Lower-level functions are available through package modules, for example `from spherical_bbox_merging.spherical_iou import spherical_iou`.

## Local development and tests

Clone the repository and install in editable mode using your virtual environment:

```bash
git clone https://github.com/YokoyamaLab/Spherical-Bounding-Box-Merging.git
cd Spherical-Bounding-Box-Merging
python -m pip install -e .
python main.py
```

[`main.py`](main.py) runs the same synthetic example. The example script and tests are available in the repository checkout and are not installed by pip. Library source files live in `src/spherical_bbox_merging/`; editable installation makes changes there available to your Python environment.

Run all 42 unit tests from the repository root after installation:

```bash
python -m unittest discover -v
```

Package metadata and dependencies are defined in [`pyproject.toml`](pyproject.toml). From the repository root, `python -m pip install -r requirements.txt` also installs the local package in editable mode. Running the scripts and tests requires installing the package first.

## Build and install a wheel

From the repository root, build a wheel locally without publishing it:

```bash
python -m pip wheel --no-deps . --wheel-dir dist
```

Install the resulting wheel in the target Python environment:

```bash
python -m pip install dist/spherical_bounding_box_merging-0.1.0-py3-none-any.whl
```

The filename reflects the version in `pyproject.toml`. NumPy and OpenCV are installed as dependencies; they are not bundled in this wheel.

## Documentation

- [Detector interface and output format](docs/detector.md) — connect your own model; bbox and mask conventions
- [Configuration and limitations](docs/configuration.md) — projection, merging thresholds, and supported ranges
- [Ultralytics YOLO example](docs/ultralytics.md) — optional third-party integration for bbox and segmentation models

## Research

This repository accompanies the paper by **Hiroto Igeta and Shohei Yokoyama**, *“Spherical Bounding Box Merging for 360-Degree Object Detection with Planar Detectors,”* presented at **SNPD 2026** and awarded **Best Paper** ([Tokyo Metropolitan University announcement](https://www.sd.tmu.ac.jp/news/prize/12671.html)).

The experiments reported in the paper were conducted using a C++ implementation. This repository provides a Python reimplementation of the proposed method, designed for easier integration with various planar object detectors. The original experimental code and evaluation environment are not included. Consequently, the processing times reported in the paper do not represent the performance of this Python implementation.

The Springer publication is **not yet available**. Its DOI and final citation will be added after publication.

The spherical IoU calculation is independently implemented based on the mathematical method proposed by [Dai et al., *Unbiased IoU for Spherical Image Object Detection* (AAAI 2022)](https://doi.org/10.1609/aaai.v36i1.19929).

## License

The original code is licensed under the [MIT License](LICENSE). Third-party detectors, model weights, datasets, and the research paper are not covered by this license.
