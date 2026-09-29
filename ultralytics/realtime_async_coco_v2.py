import cv2
import time
import threading
import numpy as np
import torch

from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


COCO_DIR = Path("/home/shatviksmit/VisionAid/ultralytics/coco100/val2017")

INPUT_FPS = 30.0

YOLO_MODEL = "yolo11n.pt"
YOLO_SIZE = 640
YOLO_CONF = 0.25

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_HZ = 12.0

DEVICE = "cuda"


@dataclass
class SharedState:

    frame: np.ndarray | None = None
    frame_id: int = 0
    frame_timestamp: float = 0.0

    detections: list = field(default_factory=list)
    yolo_frame_id: int = -1
    yolo_frame_timestamp: float = 0.0
    yolo_start_timestamp: float = 0.0
    yolo_timestamp: float = 0.0
    yolo_latency_ms: float = 0.0

    depth_map: np.ndarray | None = None
    depth_frame_id: int = -1
    depth_frame_timestamp: float = 0.0
    depth_timestamp: float = 0.0
    depth_latency_ms: float = 0.0

    hazards: list = field(default_factory=list)
    fusion_frame_id: int = -1
    fusion_timestamp: float = 0.0

    yolo_e2e_samples: list = field(default_factory=list)
    fusion_e2e_samples: list = field(default_factory=list)
    fusion_camera_samples: list = field(default_factory=list)
    depth_age_samples: list = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


state = SharedState()
running = True


def load_dataset():

    extensions = {".jpg", ".jpeg", ".png", ".webp"}

    paths = sorted(
        p for p in COCO_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not paths:
        raise RuntimeError(
            f"No images found in {COCO_DIR}"
        )

    print(f"Found {len(paths)} COCO images")

    return paths


def dataset_worker():

    global running

    image_paths = load_dataset()

    frame_interval = 1.0 / INPUT_FPS

    frame_id = 0

    print(
        f"Dataset camera started ({INPUT_FPS:.1f} FPS)"
    )

    while running:

        for image_path in image_paths:

            if not running:
                break

            frame = cv2.imread(str(image_path))

            if frame is None:
                print(
                    f"WARNING: failed to read {image_path}"
                )
                continue

            frame_id += 1

            timestamp = time.perf_counter()

            with state.lock:

                state.frame = frame
                state.frame_id = frame_id
                state.frame_timestamp = timestamp

            time.sleep(frame_interval)

    print("\nDataset camera stopped")


def yolo_worker():

    global running

    print("Loading YOLO...")

    model = YOLO(YOLO_MODEL)

    print("YOLO loaded")

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

        with state.lock:

            frame = state.frame
            frame_id = state.frame_id
            frame_timestamp = state.frame_timestamp

        if frame is None:
            time.sleep(0.001)
            continue

        if frame_id == last_frame_id:
            time.sleep(0.001)
            continue

        last_frame_id = frame_id

        t0 = time.perf_counter()

        with state.lock:
            state.yolo_start_timestamp = t0

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
                    "class_name": result.names[cls_id],
                    "confidence": confidence,
                    "bbox": (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2)
                    )
                })

        result_timestamp = time.perf_counter()

        yolo_e2e_ms = (
            result_timestamp
            - frame_timestamp
        ) * 1000.0

        with state.lock:

            state.detections = detections

            state.yolo_frame_id = frame_id

            state.yolo_frame_timestamp = (
                frame_timestamp
            )

            state.yolo_timestamp = (
                result_timestamp
            )

            state.yolo_latency_ms = latency_ms

            state.yolo_e2e_samples.append(
                yolo_e2e_ms
            )


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

        with state.lock:

            frame = state.frame
            frame_id = state.frame_id
            frame_timestamp = state.frame_timestamp

        if frame is None:
            continue

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

        depth_result_timestamp = (
            time.perf_counter()
        )

        with state.lock:

            state.depth_map = depth

            state.depth_frame_id = frame_id

            state.depth_frame_timestamp = (
                frame_timestamp
            )

            state.depth_timestamp = (
                depth_result_timestamp
            )

            state.depth_latency_ms = latency_ms


def get_bbox_depth(depth_map, bbox):

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

    return float(np.median(region))


def get_direction(bbox, frame_width):

    x1, _, x2, _ = bbox

    center_x = (x1 + x2) / 2.0

    normalized = center_x / frame_width

    if normalized < 0.33:
        return "left"

    if normalized < 0.66:
        return "center"

    return "right"


def in_collision_corridor(
    bbox,
    frame_width,
    frame_height
):

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    horizontal = (
        frame_width * 0.25
        <= cx
        <= frame_width * 0.75
    )

    vertical = (
        cy >= frame_height * 0.35
    )

    return horizontal and vertical


def fusion_loop():

    global running

    last_yolo_frame_id = -1

    print("Fusion loop started")

    while running:

        with state.lock:

            yolo_frame_id = (
                state.yolo_frame_id
            )

            yolo_frame_timestamp = (
                state.yolo_frame_timestamp
            )

            detections = list(
                state.detections
            )

            depth_map = state.depth_map

            depth_timestamp = (
                state.depth_timestamp
            )

        if yolo_frame_id < 0:
            time.sleep(0.001)
            continue

        if yolo_frame_id == last_yolo_frame_id:
            time.sleep(0.001)
            continue

        last_yolo_frame_id = yolo_frame_id

        hazards = []

        frame_width = 640
        frame_height = 480

        for detection in detections:

            detection = dict(detection)

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

            score = 0.0

            if collision:
                score += 0.5

            if depth is not None:

                if depth < 2.0:
                    score += 0.4

                elif depth < 4.0:
                    score += 0.2

            score = min(score, 1.0)

            if score >= 0.7:
                severity = "HIGH"

            elif score >= 0.4:
                severity = "MEDIUM"

            else:
                severity = "LOW"

            detection["depth"] = depth
            detection["direction"] = direction
            detection["collision_corridor"] = collision
            detection["hazard_score"] = score
            detection["severity"] = severity

            hazards.append(detection)

        fusion_timestamp = time.perf_counter()

        fusion_from_yolo_ms = (
            fusion_timestamp
            - state.yolo_timestamp
        ) * 1000.0

        fusion_camera_ms = (
            fusion_timestamp
            - yolo_frame_timestamp
        ) * 1000.0

        if depth_timestamp > 0:

            depth_age_ms = (
                fusion_timestamp
                - depth_timestamp
            ) * 1000.0

        else:

            depth_age_ms = None

        with state.lock:

            state.hazards = hazards

            state.fusion_frame_id = (
                yolo_frame_id
            )

            state.fusion_timestamp = (
                fusion_timestamp
            )

            state.fusion_e2e_samples.append(
                fusion_from_yolo_ms
            )

            state.fusion_camera_samples.append(
                fusion_camera_ms
            )

            if depth_age_ms is not None:

                state.depth_age_samples.append(
                    depth_age_ms
                )


def percentile(values, p):

    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(values),
            p
        )
    )


def telemetry_loop():

    global running

    while running:

        time.sleep(1.0)

        with state.lock:

            yolo_samples = list(
                state.yolo_e2e_samples
            )

            fusion_samples = list(
                state.fusion_e2e_samples
            )

            fusion_camera_samples = list(
                state.fusion_camera_samples
            )

            depth_age_samples = list(
                state.depth_age_samples
            )

            yolo_ms = (
                state.yolo_latency_ms
            )

            depth_ms = (
                state.depth_latency_ms
            )

            objects = len(
                state.hazards
            )

            frame_id = state.frame_id

            yolo_frame_id = (
                state.yolo_frame_id
            )

            depth_frame_id = (
                state.depth_frame_id
            )

        if not yolo_samples:
            continue

        print()
        print("=" * 75)

        print(
            f"Frames generated : {frame_id}"
        )

        print(
            f"YOLO frames      : {yolo_frame_id}"
        )

        print(
            f"Depth frame      : {depth_frame_id}"
        )

        print(
            f"Objects          : {objects}"
        )

        print()

        print("YOLO inference")

        print(
            f"  latest         : "
            f"{yolo_ms:7.2f} ms"
        )

        print("Camera -> YOLO result")

        print(
            f"  P50            : "
            f"{percentile(yolo_samples, 50):7.2f} ms"
        )

        print(
            f"  P95            : "
            f"{percentile(yolo_samples, 95):7.2f} ms"
        )

        print(
            f"  P99            : "
            f"{percentile(yolo_samples, 99):7.2f} ms"
        )

        if fusion_samples:

            print("YOLO result -> Fusion")

            print(
                f"  P50            : "
                f"{percentile(fusion_samples, 50):7.2f} ms"
            )

            print(
                f"  P95            : "
                f"{percentile(fusion_samples, 95):7.2f} ms"
            )

            print(
                f"  P99            : "
                f"{percentile(fusion_samples, 99):7.2f} ms"
            )

        if fusion_camera_samples:

            print("Camera -> Fusion")

            print(
                f"  P50            : "
                f"{percentile(fusion_camera_samples, 50):7.2f} ms"
            )

            print(
                f"  P95            : "
                f"{percentile(fusion_camera_samples, 95):7.2f} ms"
            )

            print(
                f"  P99            : "
                f"{percentile(fusion_camera_samples, 99):7.2f} ms"
            )

        if depth_age_samples:

            print("Depth staleness")

            print(
                f"  P50            : "
                f"{percentile(depth_age_samples, 50):7.2f} ms"
            )

            print(
                f"  P95            : "
                f"{percentile(depth_age_samples, 95):7.2f} ms"
            )

            print(
                f"  P99            : "
                f"{percentile(depth_age_samples, 99):7.2f} ms"
            )

        print("=" * 75)


def main():

    global running

    print()
    print("=" * 75)
    print("VISIONAID - ASYNCHRONOUS COCO PIPELINE")
    print("=" * 75)

    print(
        f"Dataset : {COCO_DIR}"
    )

    print(
        f"Input   : {INPUT_FPS:.1f} FPS"
    )

    print(
        f"Depth   : {DEPTH_HZ:.1f} Hz"
    )

    print()

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

    print("Done.")


if __name__ == "__main__":
    main()
