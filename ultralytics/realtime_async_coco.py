import cv2
import time
import threading
import numpy as np
import torch

from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)


# ============================================================
# CONFIG
# ============================================================

COCO_DIR = Path(
    "/home/shatviksmit/VisionAid/ultralytics/coco100/val2017"
)

INPUT_FPS = 30.0

YOLO_MODEL = "yolo11n.pt"
YOLO_SIZE = 640
YOLO_CONF = 0.25

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_HZ = 12.0

DEVICE = "cuda"


# ============================================================
# SHARED STATE
# ============================================================

@dataclass
class SharedState:

    # Latest dataset frame
    frame: np.ndarray | None = None
    frame_id: int = 0
    frame_timestamp: float = 0.0

    # Latest YOLO result
    detections: list = field(default_factory=list)
    yolo_frame_id: int = -1
    yolo_timestamp: float = 0.0
    yolo_latency_ms: float = 0.0

    # Latest depth result
    depth_map: np.ndarray | None = None
    depth_frame_id: int = -1
    depth_timestamp: float = 0.0
    depth_latency_ms: float = 0.0

    # Latest fusion result
    hazards: list = field(default_factory=list)
    fusion_frame_id: int = -1

    lock: threading.Lock = field(
        default_factory=threading.Lock
    )


state = SharedState()

running = True


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    paths = sorted(
        p for p in COCO_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise RuntimeError(
            f"No images found in {COCO_DIR}"
        )

    print(
        f"Found {len(paths)} COCO images"
    )

    return paths


# ============================================================
# VIRTUAL CAMERA
# ============================================================

def dataset_worker():

    global running

    image_paths = load_dataset()

    frame_interval = 1.0 / INPUT_FPS

    frame_id = 0

    print(
        f"Dataset camera started "
        f"({INPUT_FPS:.1f} FPS)"
    )

    while running:

        for image_path in image_paths:

            if not running:
                break

            # Read image
            frame = cv2.imread(
                str(image_path)
            )

            if frame is None:
                print(
                    f"\nWARNING: failed to read "
                    f"{image_path}"
                )
                continue

            frame_id += 1

            timestamp = time.perf_counter()

            # Publish latest frame
            with state.lock:

                state.frame = frame
                state.frame_id = frame_id
                state.frame_timestamp = timestamp

            # Simulate camera frame interval
            time.sleep(frame_interval)

    print("\nDataset camera stopped")


# ============================================================
# YOLO WORKER
# ============================================================

def yolo_worker():

    global running

    print("Loading YOLO...")

    model = YOLO(YOLO_MODEL)

    print("YOLO loaded")

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    dummy = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    for _ in range(20):

        model.predict(
            dummy,
            imgsz=YOLO_SIZE,
            conf=YOLO_CONF,
            device=DEVICE,
            verbose=False
        )

    torch.cuda.synchronize()

    print("YOLO warmup complete")

    last_frame_id = -1

    while running:

        # Get latest frame
        with state.lock:

            frame = state.frame
            frame_id = state.frame_id
            frame_timestamp = state.frame_timestamp

        if frame is None:

            time.sleep(0.001)
            continue

        # Don't process same frame twice
        if frame_id == last_frame_id:

            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=YOLO_SIZE,
            conf=YOLO_CONF,
            device=DEVICE,
            verbose=False
        )

        torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        detections = []

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                cls_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                x1, y1, x2, y2 = (
                    boxes.xyxy[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                detections.append({

                    "class_id": cls_id,

                    "class_name":
                        result.names[cls_id],

                    "confidence":
                        confidence,

                    "bbox": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    )
                })

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        with state.lock:

            state.detections = detections

            state.yolo_frame_id = frame_id

            state.yolo_timestamp = (
                time.perf_counter()
            )

            state.yolo_latency_ms = latency_ms


# ============================================================
# DEPTH WORKER
# ============================================================

def depth_worker():

    global running

    print(
        "Loading Depth Anything V2..."
    )

    processor = AutoImageProcessor.from_pretrained(
        DEPTH_MODEL
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL
    )

    model = model.to(DEVICE)

    model = model.half()

    model.eval()

    print("Depth model loaded")

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    dummy = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    with torch.inference_mode():

        inputs = processor(
            images=dummy,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        if "pixel_values" in inputs:

            inputs["pixel_values"] = (
                inputs["pixel_values"].half()
            )

        for _ in range(10):

            model(**inputs)

    torch.cuda.synchronize()

    print("Depth warmup complete")

    last_frame_id = -1

    next_run = time.perf_counter()

    interval = 1.0 / DEPTH_HZ

    while running:

        now = time.perf_counter()

        if now < next_run:

            time.sleep(0.001)
            continue

        next_run += interval

        # ----------------------------------------------------
        # Get latest frame
        # ----------------------------------------------------

        with state.lock:

            frame = state.frame
            frame_id = state.frame_id

        if frame is None:
            continue

        if frame_id == last_frame_id:
            continue

        last_frame_id = frame_id

        # ----------------------------------------------------
        # Depth inference
        # ----------------------------------------------------

        t0 = time.perf_counter()

        with torch.inference_mode():

            inputs = processor(
                images=frame,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            if "pixel_values" in inputs:

                inputs["pixel_values"] = (
                    inputs["pixel_values"].half()
                )

            outputs = model(**inputs)

            predicted_depth = (
                outputs.predicted_depth
            )

            predicted_depth = (
                torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(
                        frame.shape[0],
                        frame.shape[1]
                    ),
                    mode="bicubic",
                    align_corners=False
                )
                .squeeze(1)
            )

            depth = (
                predicted_depth[0]
                .float()
                .cpu()
                .numpy()
            )

        torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        # ----------------------------------------------------
        # Publish latest depth
        # ----------------------------------------------------

        with state.lock:

            state.depth_map = depth

            state.depth_frame_id = frame_id

            state.depth_timestamp = (
                time.perf_counter()
            )

            state.depth_latency_ms = latency_ms


# ============================================================
# BBOX DEPTH
# ============================================================

def get_bbox_depth(
    depth_map,
    bbox
):

    if depth_map is None:
        return None

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

    # Central 50% of bounding box
    bw = x2 - x1
    bh = y2 - y1

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

    region = depth_map[
        cy1:cy2,
        cx1:cx2
    ]

    if region.size == 0:
        return None

    return float(
        np.median(region)
    )


# ============================================================
# DIRECTION
# ============================================================

def get_direction(
    bbox,
    frame_width
):

    x1, _, x2, _ = bbox

    center_x = (
        x1 + x2
    ) / 2.0

    normalized = (
        center_x / frame_width
    )

    if normalized < 0.33:
        return "left"

    if normalized < 0.66:
        return "center"

    return "right"


# ============================================================
# COLLISION CORRIDOR
# ============================================================

def in_collision_corridor(
    bbox,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = bbox

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    horizontal = (
        frame_width * 0.25
        <= cx
        <= frame_width * 0.75
    )

    vertical = (
        cy >= frame_height * 0.35
    )

    return (
        horizontal
        and vertical
    )


# ============================================================
# FUSION
# ============================================================

def fusion_loop():

    global running

    last_frame_id = -1

    print("Fusion loop started")

    while running:

        with state.lock:

            frame_id = state.frame_id

            frame = state.frame

            detections = list(
                state.detections
            )

            depth_map = state.depth_map

        if frame is None:

            time.sleep(0.001)
            continue

        if frame_id == last_frame_id:

            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        frame_height, frame_width = (
            frame.shape[:2]
        )

        hazards = []

        for detection in detections:

            detection = dict(
                detection
            )

            bbox = detection["bbox"]

            depth = get_bbox_depth(
                depth_map,
                bbox
            )

            direction = get_direction(
                bbox,
                frame_width
            )

            collision = (
                in_collision_corridor(
                    bbox,
                    frame_width,
                    frame_height
                )
            )

            # Simple initial hazard score
            score = 0.0

            if collision:
                score += 0.5

            if depth is not None:

                if depth < 2.0:
                    score += 0.4

                elif depth < 4.0:
                    score += 0.2

            score = min(
                score,
                1.0
            )

            if score >= 0.7:
                severity = "HIGH"

            elif score >= 0.4:
                severity = "MEDIUM"

            else:
                severity = "LOW"

            detection["depth"] = depth

            detection["direction"] = (
                direction
            )

            detection[
                "collision_corridor"
            ] = collision

            detection[
                "hazard_score"
            ] = score

            detection[
                "severity"
            ] = severity

            hazards.append(
                detection
            )

        with state.lock:

            state.hazards = hazards

            state.fusion_frame_id = (
                frame_id
            )


# ============================================================
# TELEMETRY
# ============================================================

def telemetry_loop():

    global running

    frame_count = 0
    last_frame_id = -1

    fps_start = time.perf_counter()

    while running:

        time.sleep(0.05)

        with state.lock:

            frame_id = state.frame_id

            yolo_frame_id = (
                state.yolo_frame_id
            )

            depth_frame_id = (
                state.depth_frame_id
            )

            yolo_ms = (
                state.yolo_latency_ms
            )

            depth_ms = (
                state.depth_latency_ms
            )

            hazards = list(
                state.hazards
            )

            now = time.perf_counter()

            yolo_age = (
                now
                - state.yolo_timestamp
            ) * 1000.0

            depth_age = (
                now
                - state.depth_timestamp
            ) * 1000.0

            frame_age = (
                now
                - state.frame_timestamp
            ) * 1000.0

        if frame_id != last_frame_id:

            frame_count += 1
            last_frame_id = frame_id

        now = time.perf_counter()

        if (
            now - fps_start
            >= 1.0
        ):

            fps = (
                frame_count
                /
                (now - fps_start)
            )

            frame_count = 0
            fps_start = now

            print(
                f"\r"
                f"FPS {fps:5.1f} | "
                f"YOLO {yolo_ms:6.1f} ms | "
                f"Depth {depth_ms:6.1f} ms | "
                f"YOLO age {yolo_age:6.1f} ms | "
                f"Depth age {depth_age:6.1f} ms | "
                f"Objects {len(hazards):2d}",
                end="",
                flush=True
            )


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    print()
    print("=" * 75)
    print("VISIONAID — ASYNCHRONOUS COCO PIPELINE")
    print("=" * 75)

    print(
        f"\nDataset : {COCO_DIR}"
    )

    print(
        f"Input   : {INPUT_FPS:.1f} FPS"
    )

    print(
        f"Depth   : {DEPTH_HZ:.1f} Hz"
    )

    print()

    # Start workers
    threads = [

        threading.Thread(
            target=dataset_worker,
            daemon=True
        ),

        threading.Thread(
            target=yolo_worker,
            daemon=True
        ),

        threading.Thread(
            target=depth_worker,
            daemon=True
        ),

        threading.Thread(
            target=fusion_loop,
            daemon=True
        ),

        threading.Thread(
            target=telemetry_loop,
            daemon=True
        )

    ]

    for thread in threads:
        thread.start()

    print(
        "\nPipeline running."
    )

    print(
        "Press ENTER to stop.\n"
    )

    try:

        input()

    except KeyboardInterrupt:

        pass

    running = False

    print(
        "\n\nStopping..."
    )

    time.sleep(2)

    print(
        "Done."
    )


if __name__ == "__main__":
    main()
