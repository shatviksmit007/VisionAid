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


print("Loading images...")
images = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))

if not images:
    raise RuntimeError("No COCO images found")

print(f"Images: {len(images)}")


print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

print("Loading Depth Anything...")
processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL
).cuda().half()

depth_model.eval()


def get_depth_map(frame):

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


def sample_depth(depth_map, bbox):

    x1, y1, x2, y2 = bbox

    h, w = depth_map.shape

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))

    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    # Central 50% of bbox.
    cx1 = int(x1 + 0.25 * bw)
    cx2 = int(x1 + 0.75 * bw)

    cy1 = int(y1 + 0.25 * bh)
    cy2 = int(y1 + 0.75 * bh)

    roi = depth_map[cy1:cy2, cx1:cx2]

    values = roi[np.isfinite(roi)]

    if values.size == 0:
        return None

    return float(np.median(values))


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

dummy = np.zeros(
    (480, 640, 3),
    dtype=np.uint8
)

for _ in range(5):

    yolo.predict(
        dummy,
        imgsz=YOLO_SIZE,
        conf=CONF,
        verbose=False,
        device=0
    )

inputs = processor(
    images=dummy,
    return_tensors="pt"
)

inputs = {
    k: v.cuda()
    for k, v in inputs.items()
    if torch.is_tensor(v)
}

with torch.inference_mode():

    for _ in range(5):
        depth_model(**inputs)

torch.cuda.synchronize()

print("Warmup complete")


# ------------------------------------------------------------
# Process COCO
# ------------------------------------------------------------

engine = HazardEngine()

all_objects = []

processed = 0

start_total = time.perf_counter()

for image_path in images:

    frame = cv2.imread(image_path)

    if frame is None:
        continue

    timestamp = time.perf_counter()

    # YOLO
    results = yolo.predict(
        frame,
        imgsz=YOLO_SIZE,
        conf=CONF,
        verbose=False,
        device=0
    )

    result = results[0]

    # Depth
    depth_map = get_depth_map(frame)

    detections = []

    if result.boxes is not None:

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

            depth = sample_depth(
                depth_map,
                bbox
            )

            if depth is None:
                continue

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=confidence,
                    relative_depth=depth
                )
            )

    # --------------------------------------------------------
    # Static hazard metrics
    # --------------------------------------------------------

    hazards = engine.update(
        detections,
        frame_width=frame.shape[1],
        timestamp=timestamp
    )

    for obj in hazards:

        depth = obj.depth_history[-1][1]

        all_objects.append({
            "image": os.path.basename(image_path),
            "id": obj.track_id,
            "class": obj.class_name,
            "confidence": obj.confidence,
            "depth": depth,
            "path_overlap": obj.path_overlap,
            "velocity": obj.velocity,
            "ttc": obj.ttc,
            "risk": obj.risk,
        })

    processed += 1

    if processed % 100 == 0:

        elapsed = time.perf_counter() - start_total

        print(
            f"Processed {processed}/{len(images)} "
            f"({processed / elapsed:.2f} img/s)"
        )


print()
print("=" * 80)
print("COCO HAZARD TEST COMPLETE")
print("=" * 80)

print(f"Images processed : {processed}")
print(f"Objects found    : {len(all_objects)}")


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

if all_objects:

    depths = np.array([
        x["depth"]
        for x in all_objects
        if np.isfinite(x["depth"])
    ])

    overlaps = np.array([
        x["path_overlap"]
        for x in all_objects
    ])

    print()
    print("DEPTH")
    print(f"  Mean   : {np.mean(depths):.4f}")
    print(f"  Median : {np.median(depths):.4f}")
    print(f"  Min    : {np.min(depths):.4f}")
    print(f"  Max    : {np.max(depths):.4f}")

    print()
    print("PATH OVERLAP")
    print(f"  Mean   : {np.mean(overlaps):.4f}")
    print(f"  Median : {np.median(overlaps):.4f}")
    print(f"  > 0    : {np.mean(overlaps > 0) * 100:.2f}%")
    print(f"  >= 0.5 : {np.mean(overlaps >= 0.5) * 100:.2f}%")
    print(f"  == 1   : {np.mean(overlaps >= 0.999) * 100:.2f}%")

    print()
    print("RISK")
    
    risks = {}

    for x in all_objects:
        risks[x["risk"]] = risks.get(x["risk"], 0) + 1

    for risk, count in sorted(risks.items()):
        print(
            f"  {risk:<8}: {count:6d} "
            f"({count / len(all_objects) * 100:.2f}%)"
        )

print()
print("Done.")
