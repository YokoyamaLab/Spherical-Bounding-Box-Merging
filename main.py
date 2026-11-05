"""Run a synthetic example without a model or weights: python main.py."""
import numpy as np

from data_types import Detection, ProjectionOptions, ThresholdOptions
from pipeline import detect_panorama


class CenterBoxDetector:
    """Example only: returns a fixed central box, without image recognition."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        return [Detection(0, 1.0, (width / 4, height / 4, 3 * width / 4, 3 * height / 4))]


def main() -> None:
    panorama = np.zeros((64, 128, 3), dtype=np.uint8)
    results = detect_panorama(
        panorama, CenterBoxDetector(),
        ProjectionOptions(width=64, height=64), ThresholdOptions(),
    )
    for detection in results:
        print(detection)


if __name__ == "__main__":
    main()
