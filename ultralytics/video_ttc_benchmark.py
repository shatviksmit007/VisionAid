import os
import csv
import time
import cv2
import torch
import numpy as np

from ultralytics import YOLO
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from detection_nms import classwise_nms
from detection_nms import classwise_nms
from hazard_engine import Detection, HazardEngine


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = os.path.expanduser(
    "~/VisionAid/test_videos/test_vid1.mp4"
)

OUTPUT_CSV = os.path.expanduser(
    "~/VisionAid/test_videos/test_vid1_ttc.csv"
)

OUTPUT_VIDEO = os.path.expanduser(
    "~/VisionAid/test_videos/test_vid1_annotated.mp4"
)

YOLO_PATH = (
    "/home/shatviksmit/VisionAid/ultralytics/yolo11n.pt"
)

DEPTH_MODEL = (
    "depth-anything/depth-anything-v2-small-hf"
)

YOLO_SIZE = 640
CONF = 0.25

# Only test pedestrians for this experiment.
TARGET_CLASS = "person"

# Save annotated video?
SAVE_VIDEO = True


# ============================================================
# LOAD VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

FPS = cap.get(cv2.CAP_PROP_FPS)
FRAME_COUNT = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

WIDTH = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

HEIGHT = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

DURATION = FRAME_COUNT / FPS

print()
print("=" * 80)
print("VIDEO")
print("=" * 80)
print(f"Path       : {VIDEO_PATH}")
print(f"Resolution : {WIDTH} x {HEIGHT}")
print(f"FPS        : {FPS:.6f}")
print(f"Frames     : {FRAME_COUNT}")
print(f"Duration   : {DURATION:.3f}s")
print()


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO...")

yolo = YOLO(YOLO_PATH)

print("Loading Depth Anything...")

processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = (
    AutoModelForDepthEstimation
    .from_pretrained(DEPTH_MODEL)
    .cuda()
    .half()
)

depth_model.eval()

print("Models loaded.")


# ============================================================
# DEPTH FUNCTION
# ============================================================

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

        # Convert [1,H,W] → [1,1,H,W]
        depth = depth.unsqueeze(1)

        # Return to ORIGINAL VIDEO resolution.
        depth = torch.nn.functional.interpolate(
            depth,
            size=(HEIGHT, WIDTH),
            mode="bicubic",
            align_corners=False
        )

        depth = depth.squeeze()

    torch.cuda.synchronize()

    return depth.float().cpu().numpy()


# ============================================================
# DEPTH SAMPLING
# ============================================================

def sample_bbox_depth(depth_map, bbox):

    x1, y1, x2, y2 = bbox

    h, w = depth_map.shape

    x1 = max(
        0,
        min(w - 1, int(x1))
    )

    x2 = max(
        0,
        min(w, int(x2))
    )

    y1 = max(
        0,
        min(h - 1, int(y1))
    )

    y2 = max(
        0,
        min(h, int(y2))
    )

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    # Central 50% of bounding box.
    cx1 = int(
        x1 + 0.25 * bw
    )

    cx2 = int(
        x1 + 0.75 * bw
    )

    cy1 = int(
        y1 + 0.25 * bh
    )

    cy2 = int(
        y1 + 0.75 * bh
    )

    roi = depth_map[
        cy1:cy2,
        cx1:cx2
    ]

    values = roi[
        np.isfinite(roi) &
        (roi > 1e-4)
    ]

    if values.size == 0:
        return None

    return float(
        np.median(values)
    )


# ============================================================
# YOLO DETECTIONS
# ============================================================

def get_person_detections(frame):

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

        if class_name != TARGET_CLASS:
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


# ============================================================
# TRACKING
# ============================================================

def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# WARMUP
# ============================================================

print("Warming up...")

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

print("Warmup complete.")


# ============================================================
# VIDEO WRITER
# ============================================================

writer = None

if SAVE_VIDEO:

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        FPS,
        (WIDTH, HEIGHT)
    )


# ============================================================
# HAZARD ENGINE
# ============================================================

engine = HazardEngine()


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "timestamp",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "depth",
    "velocity",
    "ttc",
    "x_velocity",
    "predicted_x",
    "path_overlap",
    "predicted_path_overlap",
    "risk",
])


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("=" * 80)
print("PROCESSING")
print("=" * 80)

start_time = time.perf_counter()

frame_index = 0
total_person_detections = 0
total_valid_depth = 0

risk_counts = {
    "NONE": 0,
    "LOW": 0,
    "MEDIUM": 0,
    "URGENT": 0,
}

while True:

    ok, frame = cap.read()

    if not ok:
        break

    # Actual video timestamp.
    timestamp = frame_index / FPS

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    raw_detections = get_person_detections(
        frame
    )

    total_person_detections += len(
        raw_detections
    )

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    depth_map = get_depth_map(
        frame
    )

    # --------------------------------------------------------
    # BUILD DETECTIONS
    # --------------------------------------------------------

    detections = []

    for (
        class_id,
        class_name,
        confidence,
        bbox
    ) in raw_detections:

        depth = sample_bbox_depth(
            depth_map,
            bbox
        )

        if depth is None:
            continue

        total_valid_depth += 1

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
    # DUPLICATE SUPPRESSION
    # --------------------------------------------------------
    detections = classwise_nms(detections, iou_threshold=0.60)

    # --------------------------------------------------------
    # HAZARD ENGINE
    # --------------------------------------------------------

    hazards = engine.update(
        detections,
        frame_width=WIDTH,
        frame_height=HEIGHT,
        timestamp=timestamp
    )

    # --------------------------------------------------------
    # RECORD RESULTS
    # --------------------------------------------------------

    for obj in hazards:

        depth = (
            obj.depth_history[-1][1]
        )

        velocity = (
            ""
            if obj.velocity is None
            else obj.velocity
        )

        ttc = (
            ""
            if obj.ttc is None
            else obj.ttc
        )

        csv_writer.writerow([
            frame_index,
            f"{timestamp:.6f}",
            obj.track_id,
            obj.class_name,
            f"{obj.confidence:.4f}",
            f"{obj.bbox[0]:.2f}",
            f"{obj.bbox[1]:.2f}",
            f"{obj.bbox[2]:.2f}",
            f"{obj.bbox[3]:.2f}",
            f"{depth:.6f}",
            velocity,
            ttc,
            "" if obj.x_velocity is None else f"{obj.x_velocity:.6f}",
            "" if obj.predicted_x is None else f"{obj.predicted_x:.6f}",
            f"{obj.path_overlap:.6f}",
            f"{obj.predicted_path_overlap:.6f}",
            obj.risk,
        ])

        risk_counts[obj.risk] += 1

        # ----------------------------------------------------
        # DRAW
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            obj.bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        if obj.velocity is None:
            vtxt = "VREL ---"
        else:
            vtxt = (
                f"VREL "
                f"{obj.velocity:+.3f}"
            )

        if obj.ttc is None:
            ttxt = "TTC ---"
        else:
            ttxt = (
                f"TTC "
                f"{obj.ttc:.2f}s"
            )

        label = (
            f"ID {obj.track_id} "
            f"D {depth:.2f} "
            f"{vtxt} "
            f"{ttxt} "
            f"{obj.risk}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # DRAW WALKING CORRIDOR
    # --------------------------------------------------------

    corridor_x1 = int(
        WIDTH * 0.30
    )

    corridor_x2 = int(
        WIDTH * 0.70
    )

    cv2.line(
        frame,
        (corridor_x1, 0),
        (corridor_x1, HEIGHT),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (corridor_x2, 0),
        (corridor_x2, HEIGHT),
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # FRAME INFO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Frame {frame_index}/{FRAME_COUNT} "
            f"Time {timestamp:.2f}s"
        ),
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    if writer is not None:
        writer.write(frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_index % 30 == 0:

        elapsed = (
            time.perf_counter() -
            start_time
        )

        fps = (
            (frame_index + 1) /
            elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            (FRAME_COUNT - frame_index) /
            fps
            if fps > 0
            else 0
        )

        print(
            f"\rFrame "
            f"{frame_index:4d}/{FRAME_COUNT} | "
            f"{fps:5.2f} proc FPS | "
            f"ETA {remaining:6.1f}s",
            end="",
            flush=True
        )

    frame_index += 1


# ============================================================
# CLEANUP
# ============================================================

cap.release()

if writer is not None:
    writer.release()

csv_file.close()

elapsed = (
    time.perf_counter() -
    start_time
)

print()
print()
print("=" * 80)
print("VIDEO TTC BENCHMARK COMPLETE")
print("=" * 80)

print(
    f"Frames processed       : {frame_index}"
)

print(
    f"Processing time        : {elapsed:.2f}s"
)

print(
    f"Processing FPS         : "
    f"{frame_index / elapsed:.2f}"
)

print(
    f"Person detections      : "
    f"{total_person_detections}"
)

print(
    f"Valid depth samples    : "
    f"{total_valid_depth}"
)

print()
print("RISK COUNTS")

total_risk = sum(
    risk_counts.values()
)

for risk, count in risk_counts.items():

    pct = (
        count / total_risk * 100
        if total_risk
        else 0
    )

    print(
        f"  {risk:<8}: "
        f"{count:6d} "
        f"({pct:6.2f}%)"
    )

print()
print(f"CSV      : {OUTPUT_CSV}")

if SAVE_VIDEO:
    print(
        f"Video    : {OUTPUT_VIDEO}"
    )

print()
