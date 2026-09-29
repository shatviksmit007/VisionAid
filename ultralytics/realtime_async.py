import cv2
import time
import threading
import queue
import numpy as np
import torch

from dataclasses import dataclass, field
from ultralytics import YOLO

from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

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

    # Latest camera frame
    frame: np.ndarray | None = None
    frame_id: int = 0
    frame_timestamp: float = 0.0

    # Latest YOLO result
    detections: list = field(default_factory=list)
    yolo_timestamp: float = 0.0
    yolo_latency_ms: float = 0.0

    # Latest depth result
    depth_map: np.ndarray | None = None
    depth_timestamp: float = 0.0
    depth_latency_ms: float = 0.0

    # Latest fusion result
    hazards: list = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


state = SharedState()

running = True


# ============================================================
# CAMERA
# ============================================================

def camera_worker():

    global running

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    # MJPEG is important for your WSL webcam
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        running = False
        return

    print("Camera started")

    frame_id = 0

    while running:

        ret, frame = cap.read()

        if not ret:
            print("Camera read failed")
            continue

        frame_id += 1

        now = time.perf_counter()

        # IMPORTANT:
        # Replace the old frame.
        # Never build a queue of stale frames.
        with state.lock:

            state.frame = frame
            state.frame_id = frame_id
            state.frame_timestamp = now

    cap.release()

    print("Camera stopped")


# ============================================================
# YOLO
# ============================================================

def yolo_worker():

    global running

    print("Loading YOLO...")

    model = YOLO(YOLO_MODEL)

    print("YOLO loaded")

    # Warmup
    dummy = np.zeros(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
        dtype=np.uint8
    )

    for _ in range(10):

        model.predict(
            dummy,
            imgsz=YOLO_SIZE,
            conf=YOLO_CONF,
            device=DEVICE,
            verbose=False
        )

    print("YOLO warmup complete")

    last_frame_id = -1

    while running:

        # Get latest frame
        with state.lock:

            frame = state.frame
            frame_id = state.frame_id

        if frame is None:
            time.sleep(0.001)
            continue

        # Don't process the same frame twice
        if frame_id == last_frame_id:
            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        t0 = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=YOLO_SIZE,
            conf=YOLO_CONF,
            device=DEVICE,
            verbose=False
        )

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
                    "class_name": result.names[cls_id],
                    "confidence": confidence,
                    "bbox": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    )
                })

        # Publish result
        with state.lock:

            state.detections = detections
            state.yolo_timestamp = time.perf_counter()
            state.yolo_latency_ms = latency_ms


    print("YOLO stopped")


# ============================================================
# DEPTH
# ============================================================

def depth_worker():

    global running

    print("Loading Depth Anything V2...")

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

    # Warmup
    dummy = np.zeros(
        (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
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

        for _ in range(5):

            outputs = model(**inputs)

    print("Depth warmup complete")

    last_run = 0.0
    last_frame_id = -1

    interval = 1.0 / DEPTH_HZ

    while running:

        now = time.perf_counter()

        if now - last_run < interval:

            time.sleep(0.001)
            continue

        last_run = now

        with state.lock:

            frame = state.frame
            frame_id = state.frame_id

        if frame is None:
            continue

        # We don't need to process the same frame repeatedly
        if frame_id == last_frame_id:
            continue

        last_frame_id = frame_id

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

            predicted_depth = outputs.predicted_depth

            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(
                    frame.shape[0],
                    frame.shape[1]
                ),
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            depth = predicted_depth[0].float().cpu().numpy()

        latency_ms = (
            time.perf_counter() - t0
        ) * 1000.0

        with state.lock:

            state.depth_map = depth
            state.depth_timestamp = time.perf_counter()
            state.depth_latency_ms = latency_ms


    print("Depth stopped")


# ============================================================
# DEPTH FROM BBOX
# ============================================================

def get_bbox_depth(
    depth_map,
    bbox
):

    if depth_map is None:
        return None

    x1, y1, x2, y2 = bbox

    h, w = depth_map.shape

    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))

    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    # Use central region to avoid bbox edges
    bw = x2 - x1
    bh = y2 - y1

    cx1 = int(x1 + 0.25 * bw)
    cx2 = int(x1 + 0.75 * bw)

    cy1 = int(y1 + 0.25 * bh)
    cy2 = int(y1 + 0.75 * bh)

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
# FUSION
# ============================================================

def fusion_loop():

    global running

    last_frame_id = -1

    print("Fusion loop started")

    while running:

        with state.lock:

            frame_id = state.frame_id
            detections = list(state.detections)
            depth_map = state.depth_map

        if frame_id == last_frame_id:

            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        hazards = []

        for detection in detections:

            depth = get_bbox_depth(
                depth_map,
                detection["bbox"]
            )

            detection = dict(detection)

            detection["depth"] = depth

            hazards.append(
                detection
            )

        with state.lock:

            state.hazards = hazards

    print("Fusion stopped")


# ============================================================
# TELEMETRY
# ============================================================

def telemetry_loop():

    global running

    last_print = time.perf_counter()

    frame_count = 0
    last_frame_id = -1

    fps_start = time.perf_counter()

    while running:

        time.sleep(0.05)

        with state.lock:

            frame_id = state.frame_id

            yolo_ms = state.yolo_latency_ms
            depth_ms = state.depth_latency_ms

            hazards = list(state.hazards)

            yolo_age = (
                time.perf_counter()
                - state.yolo_timestamp
            ) * 1000.0

            depth_age = (
                time.perf_counter()
                - state.depth_timestamp
            ) * 1000.0

        if frame_id != last_frame_id:

            frame_count += 1
            last_frame_id = frame_id

        now = time.perf_counter()

        if now - fps_start >= 1.0:

            fps = frame_count / (
                now - fps_start
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
    print("=" * 70)
    print("VISIONAID ASYNCHRONOUS PIPELINE")
    print("=" * 70)
    print()

    threads = [

        threading.Thread(
            target=camera_worker,
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

    print()
    print("Pipeline running.")
    print("Press ENTER to stop.")
    print()

    try:

        input()

    except KeyboardInterrupt:

        pass

    running = False

    print()
    print("Stopping pipeline...")

    time.sleep(2)

    print("Done.")


if __name__ == "__main__":
    main()
