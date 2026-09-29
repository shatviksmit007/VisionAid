import os
import glob
import cv2
import time
import torch
import numpy as np

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from hazard_engine import Detection, HazardEngine


IMGDIR = "/home/shatviksmit/VisionAid/ultralytics/coco100/val2017"
YOLO_PATH = "/home/shatviksmit/VisionAid/ultralytics/yolo11n.pt"
DEPTH_MODEL = "depth-anything/depth-anything-v2-small-hf"

YOLO_SIZE = 640
CONF = 0.25

# Number of synthetic temporal frames.
N_FRAMES = 12

# Simulated approach.
APPROACH_SCALES = np.linspace(1.00, 1.35, N_FRAMES)

# Simulated retreat.
RETREAT_SCALES = np.linspace(1.35, 1.00, N_FRAMES)

# Simulated frame interval.
DT = 1.0 / 12.0

HAZARD_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "dog",
    "cat",
    "chair",
    "bench",
    "backpack",
    "suitcase",
}


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

print("Loading Depth Anything...")
processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL
).cuda().half()

depth_model.eval()


# ============================================================
# HELPERS
# ============================================================

def resize_zoom(frame, scale):

    h, w = frame.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # Center crop back to original resolution.
    x1 = (new_w - w) // 2
    y1 = (new_h - h) // 2

    x2 = x1 + w
    y2 = y1 + h

    if x2 > new_w:
        x2 = new_w
        x1 = x2 - w

    if y2 > new_h:
        y2 = new_h
        y1 = y2 - h

    result = resized[y1:y2, x1:x2]

    return result


def get_depth(frame):

    inputs = processor(
        images=frame,
        return_tensors="pt"
    )

    inputs = {
        k: v.cuda()
        for k, v in inputs.items()
        if torch.is_tensor(v)
    }

    with torch.inference_mode():

        output = depth_model(**inputs)

        depth = output.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    torch.cuda.synchronize()

    return depth.float().cpu().numpy()


def sample_depth(depth, bbox):

    x1, y1, x2, y2 = bbox

    h, w = depth.shape

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))

    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    # Central 50%.
    cx1 = int(x1 + 0.25 * bw)
    cx2 = int(x1 + 0.75 * bw)

    cy1 = int(y1 + 0.25 * bh)
    cy2 = int(y1 + 0.75 * bh)

    roi = depth[cy1:cy2, cx1:cx2]

    values = roi[
        np.isfinite(roi) &
        (roi > 1e-4)
    ]

    if values.size == 0:
        return None

    return float(np.median(values))


def detect(frame):

    results = yolo.predict(
        frame,
        imgsz=YOLO_SIZE,
        conf=CONF,
        verbose=False,
        device=0
    )

    result = results[0]

    detections = []

    if result.boxes is None:
        return detections

    for i in range(len(result.boxes)):

        class_id = int(
            result.boxes.cls[i].item()
        )

        class_name = yolo.names[class_id]

        if class_name not in HAZARD_CLASSES:
            continue

        confidence = float(
            result.boxes.conf[i].item()
        )

        bbox = tuple(
            float(x)
            for x in result.boxes.xyxy[i].tolist()
        )

        detections.append(
            (
                class_id,
                class_name,
                confidence,
                bbox
            )
        )

    return detections


def choose_target(detections, frame_width):

    if not detections:
        return None

    # Prefer an object whose center is near the walking corridor.
    candidates = []

    for det in detections:

        class_id, class_name, conf, bbox = det

        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2.0
        normalized_x = center_x / frame_width

        distance_from_center = abs(
            normalized_x - 0.5
        )

        candidates.append(
            (
                distance_from_center,
                -conf,
                det
            )
        )

    candidates.sort()

    return candidates[0][2]


def run_sequence(name, base_frame, scales):

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    engine = HazardEngine()

    frame_h, frame_w = base_frame.shape[:2]

    rows = []

    target_class = None

    for i, scale in enumerate(scales):

        frame = resize_zoom(
            base_frame,
            float(scale)
        )

        detections = detect(frame)

        if target_class is None:

            target = choose_target(
                detections,
                frame_w
            )

            if target is None:
                print("No suitable target found.")
                return

            target_class = target[1]

            print(
                f"Target: {target_class}"
            )

        # Prefer the same class as target.
        candidates = [
            d for d in detections
            if d[1] == target_class
        ]

        if not candidates:

            print(
                f"{i:02d} scale={scale:.3f} "
                "TARGET LOST"
            )

            continue

        # Choose highest confidence same-class detection.
        target = max(
            candidates,
            key=lambda x: x[2]
        )

        class_id, class_name, conf, bbox = target

        depth_map = get_depth(frame)

        depth = sample_depth(
            depth_map,
            bbox
        )

        if depth is None:

            print(
                f"{i:02d} scale={scale:.3f} "
                "INVALID DEPTH"
            )

            continue

        detection = Detection(
            class_id=class_id,
            class_name=class_name,
            bbox=bbox,
            confidence=conf,
            relative_depth=depth
        )

        timestamp = i * DT

        hazards = engine.update(
            [detection],
            frame_width=frame_w,
            timestamp=timestamp
        )

        if not hazards:
            continue

        obj = hazards[0]

        v = obj.velocity
        ttc = obj.ttc

        rows.append(
            (
                i,
                scale,
                depth,
                v,
                ttc,
                obj.path_overlap,
                obj.risk
            )
        )

        vtxt = "---" if v is None else f"{v:+.4f}"
        ttxt = "---" if ttc is None else f"{ttc:.3f}"

        print(
            f"{i:02d} "
            f"scale={scale:.3f} "
            f"depth={depth:.4f} "
            f"vrel={vtxt:>9} "
            f"ttc={ttxt:>7} "
            f"path={obj.path_overlap:.3f} "
            f"{obj.risk}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    if len(rows) < 3:

        print()
        print("Not enough valid temporal samples.")

        return

    depths = np.array([
        r[2]
        for r in rows
    ])

    print()
    print("SUMMARY")

    print(
        f"Depth first : {depths[0]:.5f}"
    )

    print(
        f"Depth last  : {depths[-1]:.5f}"
    )

    print(
        f"Depth delta : {depths[-1] - depths[0]:+.5f}"
    )

    velocities = [
        r[3]
        for r in rows
        if r[3] is not None
    ]

    if velocities:

        print(
            f"Velocity median : "
            f"{np.median(velocities):+.5f}"
        )

    ttcs = [
        r[4]
        for r in rows
        if r[4] is not None
    ]

    if ttcs:

        print(
            f"TTC first valid : "
            f"{ttcs[0]:.3f}s"
        )

        print(
            f"TTC last valid  : "
            f"{ttcs[-1]:.3f}s"
        )

    print()


# ============================================================
# FIND A GOOD COCO IMAGE
# ============================================================

print("Searching for suitable COCO image...")

images = sorted(
    glob.glob(
        os.path.join(
            IMGDIR,
            "*.jpg"
        )
    )
)

selected = None

for path in images:

    frame = cv2.imread(path)

    if frame is None:
        continue

    detections = detect(frame)

    target = choose_target(
        detections,
        frame.shape[1]
    )

    if target is not None:

        selected = frame

        print(
            "Selected:",
            os.path.basename(path),
            "| target:",
            target[1]
        )

        break


if selected is None:

    raise RuntimeError(
        "Could not find a suitable COCO image"
    )


# ============================================================
# WARMUP
# ============================================================

print("Warmup...")

dummy = np.zeros_like(selected)

for _ in range(3):

    yolo.predict(
        dummy,
        imgsz=YOLO_SIZE,
        conf=CONF,
        verbose=False,
        device=0
    )

_ = get_depth(dummy)

print("Warmup complete")


# ============================================================
# RUN EXPERIMENTS
# ============================================================

run_sequence(
    "APPROACHING CAMERA",
    selected,
    APPROACH_SCALES
)

run_sequence(
    "MOVING AWAY",
    selected,
    RETREAT_SCALES
)

print()
print("=" * 90)
print("TEMPORAL TTC TEST COMPLETE")
print("=" * 90)
