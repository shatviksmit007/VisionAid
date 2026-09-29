import time
import gc
import cv2
import torch
import numpy as np

from PIL import Image
from ultralytics import YOLO
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda"

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

IMAGE_PATH = "bus.jpg"

YOLO_SIZE = 640
DEPTH_SIZE = 518

CONF = 0.25

WARMUP = 30
ITERATIONS = 200


# ============================================================
# CLEAN GPU
# ============================================================

gc.collect()
torch.cuda.empty_cache()

assert torch.cuda.is_available()

print("=" * 70)
print("VISIONAID — YOLO11n + DEPTH ANYTHING V2")
print("FP16 DEPTH")
print("=" * 70)

print(
    f"GPU: "
    f"{torch.cuda.get_device_name(0)}"
)

print()


# ============================================================
# IMAGE
# ============================================================

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

height, width = image_rgb.shape[:2]

print(
    f"Input: {width}x{height}"
)

print()


# ============================================================
# YOLO
# ============================================================

print("Loading YOLO11n...")

yolo = YOLO(YOLO_MODEL)

print("YOLO loaded.")


# ============================================================
# DEPTH
# ============================================================

print("Loading Depth Anything V2 Small...")

processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL,
    torch_dtype=torch.float16,
).to(DEVICE).eval()

print("Depth loaded.")

print()


# ============================================================
# DEPTH INPUT
# ============================================================

depth_image = Image.fromarray(
    image_rgb
).resize(
    (DEPTH_SIZE, DEPTH_SIZE)
)

depth_inputs = processor(
    images=depth_image,
    return_tensors="pt",
)

depth_inputs = {
    k: v.to(DEVICE)
    for k, v in depth_inputs.items()
}


# ============================================================
# DEPTH
# ============================================================

def run_depth():

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            outputs = depth_model(
                **depth_inputs
            )

            predicted_depth = (
                outputs.predicted_depth
            )

            depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )[0, 0]

    return depth


# ============================================================
# YOLO
# ============================================================

def run_yolo():

    results = yolo.predict(
        source=image_bgr,
        imgsz=YOLO_SIZE,
        conf=CONF,
        device=0,
        verbose=False,
    )

    return results[0]


# ============================================================
# FUSION
# ============================================================

def fuse_depth(result, depth):

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return []

    boxes = result.boxes.xyxy
    classes = result.boxes.cls
    confidences = result.boxes.conf

    output = []

    for box, cls, conf in zip(
        boxes,
        classes,
        confidences,
    ):

        x1, y1, x2, y2 = (
            box.int()
        )

        x1 = max(
            0,
            min(width - 1, int(x1))
        )

        x2 = max(
            0,
            min(width, int(x2))
        )

        y1 = max(
            0,
            min(height - 1, int(y1))
        )

        y2 = max(
            0,
            min(height, int(y2))
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Central 50% of bounding box
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
            bx1:bx2
        ]

        if roi.numel() == 0:
            continue

        # Entirely GPU-side
        depth_value = torch.median(
            roi
        ).item()

        output.append({
            "class": int(cls.item()),
            "confidence": float(conf.item()),
            "box": (
                x1,
                y1,
                x2,
                y2,
            ),
            "depth": depth_value,
        })

    return output


# ============================================================
# WARMUP
# ============================================================

print("=" * 70)
print("WARMUP")
print("=" * 70)

for _ in range(WARMUP):

    result = run_yolo()

    depth = run_depth()

    fused = fuse_depth(
        result,
        depth,
    )

torch.cuda.synchronize()

print("Warmup complete.")
print()


# ============================================================
# BENCHMARK ARRAYS
# ============================================================

yolo_times = []
depth_times = []
fusion_times = []
total_times = []


last_result = None
last_depth = None
last_fused = None


# ============================================================
# BENCHMARK
# ============================================================

print(
    f"Running {ITERATIONS} iterations..."
)

for i in range(ITERATIONS):

    torch.cuda.synchronize()

    total_start = time.perf_counter()


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    torch.cuda.synchronize()

    start = time.perf_counter()

    result = run_yolo()

    torch.cuda.synchronize()

    yolo_ms = (
        time.perf_counter()
        - start
    ) * 1000


    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    torch.cuda.synchronize()

    start = time.perf_counter()

    depth = run_depth()

    torch.cuda.synchronize()

    depth_ms = (
        time.perf_counter()
        - start
    ) * 1000


    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    torch.cuda.synchronize()

    start = time.perf_counter()

    fused = fuse_depth(
        result,
        depth,
    )

    torch.cuda.synchronize()

    fusion_ms = (
        time.perf_counter()
        - start
    ) * 1000


    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    torch.cuda.synchronize()

    total_ms = (
        time.perf_counter()
        - total_start
    ) * 1000


    yolo_times.append(
        yolo_ms
    )

    depth_times.append(
        depth_ms
    )

    fusion_times.append(
        fusion_ms
    )

    total_times.append(
        total_ms
    )


    last_result = result
    last_depth = depth
    last_fused = fused


# ============================================================
# STATISTICS
# ============================================================

def stats(values):

    values = np.asarray(
        values
    )

    return {
        "min": np.min(values),
        "p50": np.percentile(
            values,
            50
        ),
        "p95": np.percentile(
            values,
            95
        ),
        "p99": np.percentile(
            values,
            99
        ),
        "max": np.max(values),
        "mean": np.mean(values),
    }


ys = stats(
    yolo_times
)

ds = stats(
    depth_times
)

fs = stats(
    fusion_times
)

ts = stats(
    total_times
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)


print()
print("YOLO11n")
print("-" * 40)

print(
    f"Min : {ys['min']:.2f} ms"
)

print(
    f"P50 : {ys['p50']:.2f} ms"
)

print(
    f"P95 : {ys['p95']:.2f} ms"
)

print(
    f"P99 : {ys['p99']:.2f} ms"
)

print(
    f"Max : {ys['max']:.2f} ms"
)

print(
    f"FPS : {1000 / ys['p50']:.2f}"
)


print()
print("Depth Anything V2 Small — FP16")
print("-" * 40)

print(
    f"Min : {ds['min']:.2f} ms"
)

print(
    f"P50 : {ds['p50']:.2f} ms"
)

print(
    f"P95 : {ds['p95']:.2f} ms"
)

print(
    f"P99 : {ds['p99']:.2f} ms"
)

print(
    f"Max : {ds['max']:.2f} ms"
)

print(
    f"FPS : {1000 / ds['p50']:.2f}"
)


print()
print("GPU DEPTH FUSION")
print("-" * 40)

print(
    f"P50 : {fs['p50']:.2f} ms"
)

print(
    f"P95 : {fs['p95']:.2f} ms"
)

print(
    f"P99 : {fs['p99']:.2f} ms"
)


print()
print("END-TO-END")
print("-" * 40)

print(
    f"Min : {ts['min']:.2f} ms"
)

print(
    f"P50 : {ts['p50']:.2f} ms"
)

print(
    f"P95 : {ts['p95']:.2f} ms"
)

print(
    f"P99 : {ts['p99']:.2f} ms"
)

print(
    f"Max : {ts['max']:.2f} ms"
)

print(
    f"Mean: {ts['mean']:.2f} ms"
)

print(
    f"FPS : {1000 / ts['p50']:.2f}"
)


# ============================================================
# DETECTIONS
# ============================================================

print()
print("=" * 70)
print("DETECTIONS")
print("=" * 70)

names = yolo.names

for item in last_fused:

    cls = item["class"]

    print(
        f"{names[cls]:20s} "
        f"conf={item['confidence']:.2f} "
        f"depth={item['depth']:.4f} "
        f"box={item['box']}"
    )


print()
print("Done.")
