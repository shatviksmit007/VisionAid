import numpy as np


def box_iou(box_a, box_b):
    """
    IoU for xyxy boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def classwise_nms(detections, iou_threshold=0.70):
    """
    Simple class-wise NMS.

    Each detection must contain:
        bbox
        class_id
        confidence

    Returns the surviving detections.
    """

    if not detections:
        return []

    kept = []

    class_ids = sorted(
        set(int(d.class_id) for d in detections)
    )

    for class_id in class_ids:

        candidates = [
            d for d in detections
            if int(d.class_id) == class_id
        ]

        candidates.sort(
            key=lambda d: float(d.confidence),
            reverse=True,
        )

        while candidates:

            best = candidates.pop(0)
            kept.append(best)

            survivors = []

            for candidate in candidates:

                iou = box_iou(
                    best.bbox,
                    candidate.bbox,
                )

                if iou < iou_threshold:
                    survivors.append(candidate)

            candidates = survivors

    return kept
