import time
import sys
import select
import cv2
import torch
import numpy as np

from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda"

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

CAMERA = "/dev/video0"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_SIZE = 640
CONF = 0.25

WINDOW_NAME = "VisionAid - Live Pipeline"


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("=" * 70)
print("VISIONAID LIVE PIPELINE")
print("=" * 70)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(
    f"VRAM: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)
print()


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO11n...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# LOAD DEPTH ANYTHING V2
# ============================================================

print("Loading Depth Anything V2 Small FP16...")

processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL,
    dtype=torch.float16,
).to(DEVICE).eval()

print("Depth model loaded.")
print()


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(
    CAMERA,
    cv2.CAP_V4L2,
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open camera: {CAMERA}"
    )

# IMPORTANT:
# Keep the camera in MJPEG mode.
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG"),
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH,
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT,
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS,
)

print(
    "Camera:",
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

print()


# ============================================================
# DEPTH FUNCTION
# ============================================================

@torch.inference_mode()
def run_depth(frame_rgb):

    image = Image.fromarray(frame_rgb)

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    # Convert image tensor to FP16.
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].half()

    outputs = depth_model(**inputs)

    predicted_depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(
            frame_rgb.shape[0],
            frame_rgb.shape[1],
        ),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    return depth


# ============================================================
# DEPTH + DETECTION FUSION
# ============================================================

def fuse_depth(
    result,
    depth,
    frame_width,
    frame_height,
):

    if result.boxes is None:
        return []

    if len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy
    classes = result.boxes.cls
    confidences = result.boxes.conf

    depth = depth.squeeze(0)

    output = []

    for box, cls, conf in zip(
        boxes,
        classes,
        confidences,
    ):

        x1, y1, x2, y2 = (
            box.detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        # Clamp coordinates.
        x1 = max(
            0,
            min(frame_width - 1, x1),
        )

        x2 = max(
            0,
            min(frame_width, x2),
        )

        y1 = max(
            0,
            min(frame_height - 1, y1),
        )

        y2 = max(
            0,
            min(frame_height, y2),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Central 50% of bounding box.
        # This avoids sampling too much background.
        bx1 = x1 + int(
            (x2 - x1) * 0.25
        )

        bx2 = x2 - int(
            (x2 - x1) * 0.25
        )

        by1 = y1 + int(
            (y2 - y1) * 0.25
        )

        by2 = y2 - int(
            (y2 - y1) * 0.25
        )

        roi = depth[
            by1:by2,
            bx1:bx2,
        ]

        if roi.numel() == 0:
            continue

        # Median depth.
        depth_value = float(
            torch.median(roi).item()
        )

        output.append(
            {
                "class": int(cls.item()),
                "confidence": float(conf.item()),
                "box": (x1, y1, x2, y2),
                "depth": depth_value,
            }
        )

    return output


# ============================================================
# WARMUP
# ============================================================

print("Warming up models...")

for _ in range(10):

    ret, frame = cap.read()

    if not ret:
        continue

    yolo.predict(
        source=frame,
        imgsz=YOLO_SIZE,
        conf=CONF,
        device=0,
        verbose=False,
    )

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    run_depth(frame_rgb)

torch.cuda.synchronize()

print("Warmup complete.")
print()


# ============================================================
# LIVE LOOP
# ============================================================

print("=" * 70)
print("LIVE PIPELINE")
print("=" * 70)
print("Press Q to quit.")
print()


frame_times = []

last_time = time.perf_counter()

while True:

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("Camera frame read failed.")
        break

    frame_height, frame_width = frame.shape[:2]

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    torch.cuda.synchronize()

    start = time.perf_counter()

    results = yolo.predict(
        source=frame,
        imgsz=YOLO_SIZE,
        conf=CONF,
        device=0,
        verbose=False,
    )

    torch.cuda.synchronize()

    yolo_ms = (
        time.perf_counter() - start
    ) * 1000

    result = results[0]

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    torch.cuda.synchronize()

    start = time.perf_counter()

    depth = run_depth(frame_rgb)

    torch.cuda.synchronize()

    depth_ms = (
        time.perf_counter() - start
    ) * 1000

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    start = time.perf_counter()

    fused = fuse_depth(
        result,
        depth,
        frame_width,
        frame_height,
    )

    fusion_ms = (
        time.perf_counter() - start
    ) * 1000

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    torch.cuda.synchronize()

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    frame_times.append(total_ms)

    if len(frame_times) > 30:
        frame_times.pop(0)

    avg_ms = np.mean(frame_times)

    fps = (
        1000.0 / avg_ms
        if avg_ms > 0
        else 0
    )

    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    for item in fused:

        x1, y1, x2, y2 = item["box"]

        cls = item["class"]
        conf = item["confidence"]
        depth_value = item["depth"]

        name = yolo.names[cls]

        label = (
            f"{name} "
            f"{conf:.2f} "
            f"D:{depth_value:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    # --------------------------------------------------------
    # PERFORMANCE INFO
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (280, 90),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Total: {total_ms:.1f} ms",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"YOLO: {yolo_ms:.1f}  Depth: {depth_ms:.1f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print("\033[2J\033[H", end="")
    print("=" * 70)
    print("VISIONAID LIVE PIPELINE")
    print("=" * 70)
    print(f"FPS: {fps:.1f}   |   Total: {total_ms:.1f} ms   |   YOLO: {yolo_ms:.1f} ms   |   Depth: {depth_ms:.1f} ms")
    print()
    print(f"Objects detected: {len(fused)}")
    print("-" * 70)

    for item in fused:
        cls = item["class"]
        print(f"{yolo.names[cls]:20s} conf={item['confidence']:.2f} depth={item['depth']:.4f} box={item['box']}")

    print()
    print("Press Q + Enter to quit.", flush=True)

    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.readline().strip().lower()
        if key == "q":
            break




# ============================================================
# CLEANUP
# ============================================================

cap.release()


print()
print("VisionAid stopped.")
