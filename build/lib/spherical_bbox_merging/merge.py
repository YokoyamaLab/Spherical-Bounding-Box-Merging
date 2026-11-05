from collections.abc import Callable
from .data_types import SphericalBBox, SphericalCandidate, ThresholdOptions, SphericalDetection
from collections import defaultdict
from .spherical_iou import spherical_iou, spherical_ioa
import math

def greedy_cluster(
    candidates: list[SphericalCandidate],
    overlap_fn: Callable[[SphericalBBox, SphericalBBox], float],
    threshold: float,
) -> list[list[SphericalCandidate]]:

    n = len(candidates)

    # Calculate overlap for every pair.
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            box_i = candidates[i].detection.spherical_bbox
            box_j = candidates[j].detection.spherical_bbox

            try:
                overlap = overlap_fn(box_i, box_j)
            except ValueError:
                overlap = 0.0

            if overlap > threshold:
                pairs.append((overlap, i, j))

    # Process pairs in descending order of overlap.
    pairs.sort(reverse=True)

    # Initialize disjoint sets.
    parent = list(range(n))
    source_view_ids = [
        c.view_ids.copy() for c in candidates
    ]

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Greedy clustering.
    for overlap, i, j in pairs:
        root_i = find(i)
        root_j = find(j)

        if root_i == root_j:
            continue

        # Do not merge clusters containing the same source view.
        if not source_view_ids[root_i].isdisjoint(
            source_view_ids[root_j]
        ):
            continue

        parent[root_j] = root_i
        source_view_ids[root_i].update(source_view_ids[root_j])

    # Collect clusters.
    clusters = {}

    for i, candidate in enumerate(candidates):
        root = find(i)
        clusters.setdefault(root, []).append(candidate)

    return list(clusters.values())

def _enclosing_longitude_interval(boxes: list[SphericalBBox]) -> tuple[float, float]:
    """Return (yaw, width) of the shortest circular interval covering the boxes.

    Each input interval is yaw +/- horizontal_fov/2, not the longitude
    projection of its spherical polygon. Remove the largest uncovered gap
    from the circle. For gaps equal within 32 ulps of 2*pi, choose the enclosing
    interval with the smallest start in [0, 2*pi). This tie rule is independent
    of input order; use the selected gap's actual endpoints to retain coverage.

    Full-circle coverage has the canonical result (0, 2*pi). Widths >= pi
    are deliberately preserved: clipping them would discard input intervals.
    Such outputs are outside spherical_iou's FOV domain.
    """
    period = math.tau
    intervals = []
    for box in boxes:
        width = box.horizontal_fov
        if width >= period:
            return 0.0, period
        start = (box.yaw - width / 2) % period
        end = start + width
        if end <= period:
            intervals.append((start, end))
        else:
            intervals.extend(((start, period), (0.0, end - period)))

    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    if len(merged) == 1 and merged[0] == (0.0, period):
        return 0.0, period

    gaps = []
    for i, (_, end) in enumerate(merged):
        start = merged[(i + 1) % len(merged)][0]
        wraps = i == len(merged) - 1
        gap = start + (period if wraps else 0.0) - end
        width = end + (0.0 if wraps else period) - start
        gaps.append((gap, start, width))

    largest_gap = max(gap for gap, _, _ in gaps)
    tie_tolerance = 32 * math.ulp(period)
    _, start, width = min(
        (item for item in gaps if largest_gap - item[0] <= tie_tolerance),
        key=lambda item: item[1],
    )
    yaw = (start + width / 2 + math.pi) % period - math.pi
    return yaw, width


def enclose_cluster(
    cluster: list[SphericalCandidate],
) -> SphericalCandidate:
    """Enclose angular intervals; the resulting FOV is not clipped to < pi."""
    if not cluster:
        raise ValueError("Cluster must not be empty.")

    yaw_center, horizontal_fov = _enclosing_longitude_interval(
        [candidate.detection.spherical_bbox for candidate in cluster]
    )
    pitch_min = math.inf
    pitch_max = -math.inf

    confidence = 0.0
    view_ids = set()

    for candidate in cluster:
        detection = candidate.detection
        box = detection.spherical_bbox

        pitch_min = min(pitch_min, box.pitch - box.vertical_fov / 2)
        pitch_max = max(pitch_max, box.pitch + box.vertical_fov / 2)

        confidence = max(confidence, detection.confidence)
        view_ids.update(candidate.view_ids)

    # Construct the enclosing spherical bounding box.
    pitch_center = (pitch_min + pitch_max) / 2

    spherical_bbox = SphericalBBox(
        yaw=yaw_center,
        pitch=pitch_center,
        horizontal_fov=horizontal_fov,
        vertical_fov=pitch_max - pitch_min,
    )

    spherical_detection = SphericalDetection(
        class_id=cluster[0].detection.class_id,
        confidence=confidence,
        spherical_bbox=spherical_bbox,
    )

    return SphericalCandidate(
        detection=spherical_detection,
        view_ids=view_ids,
    )


def merge(
    candidates: list[SphericalCandidate],
    options: ThresholdOptions,
) -> list[SphericalDetection]:

    # Group candidates by class.
    candidates_by_class = defaultdict(list)

    for candidate in candidates:
        class_id = candidate.detection.class_id
        candidates_by_class[class_id].append(candidate)

    results = []

    for class_candidates in candidates_by_class.values():

        # First stage: IoU-based clustering.
        iou_clusters = greedy_cluster(
            class_candidates,
            spherical_iou,
            options.iou_threshold,
        )

        iou_candidates = [
            enclose_cluster(cluster)
            for cluster in iou_clusters
        ]

        # Second stage: IoA-based clustering.
        ioa_clusters = greedy_cluster(
            iou_candidates,
            spherical_ioa,
            options.ioa_threshold,
        )

        results.extend(
            enclose_cluster(cluster).detection
            for cluster in ioa_clusters
        )

    return results
