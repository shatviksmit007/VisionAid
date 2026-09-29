
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ORT_NUM_THREADS"] = "2"

import gc
import time
import statistics
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from rapidocr import RapidOCR

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

IMAGE_DIR = Path(
    "/home/shatviksmit/VisionAid/ultralytics/coco100/val2017"
)

MAX_IMAGES = 100
WARMUP = 5

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

print("Loading YOLO...")
yolo = YOLO("yolo11n.pt")

print("Loading Depth Anything V2...")
# Use the same depth setup you benchmarked previously.
# Adjust import/path ONLY if your existing depth benchmark used
# a different local loader.
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).half().cuda().eval()

print("Loading RapidOCR...")
ocr = RapidOCR(params={
    "EngineConfig.onnxruntime.use_cuda": True,
    "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0
})

print("Models loaded.")
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# ------------------------------------------------------------
# IMAGES
# ------------------------------------------------------------

images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)[:MAX_IMAGES]

print(f"Images: {len(images)}")

# ------------------------------------------------------------
# TIMING STORAGE
# ONLY NUMBERS — NO IMAGES / RESULTS
# ------------------------------------------------------------

yolo_times = []
depth_times = []
ocr_times = []
fusion_times = []
total_times = []

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def pct(values, p):
    values = sorted(values)
    idx = min(len(values) - 1, int(p * len(values)))
    return values[idx]

# ------------------------------------------------------------
# WARMUP
# ------------------------------------------------------------

print(f"\nWarming up ({WARMUP} images)...")

for path in images[:WARMUP]:
    img = cv2.imread(str(path))

    if img is None:
        continue

    # YOLO
    _ = yolo.predict(
        img,
        verbose=False,
        device=0
    )

    # Depth
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(
        images=rgb,
        return_tensors="pt"
    )

    inputs = {
        k: v.half().cuda()
        if v.is_floating_point()
        else v.cuda()
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        _ = depth_model(**inputs)

    # OCR
    _ = ocr(img)

    del img, rgb, inputs
    gc.collect()

torch.cuda.synchronize()
torch.cuda.empty_cache()

print("Warmup complete.")

# ------------------------------------------------------------
# BENCHMARK
# ------------------------------------------------------------

print("\nRunning memory-safe benchmark...\n")

for i, path in enumerate(images):

    img = cv2.imread(str(path))

    if img is None:
        continue

    total_start = time.perf_counter()

    # ---------------- YOLO ----------------

    t = time.perf_counter()

    yolo_result = yolo.predict(
        img,
        verbose=False,
        device=0
    )

    torch.cuda.synchronize()

    yolo_ms = (time.perf_counter() - t) * 1000

    # ---------------- DEPTH ----------------

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = time.perf_counter()

    inputs = processor(
        images=rgb,
        return_tensors="pt"
    )

    inputs = {
        k: v.half().cuda()
        if v.is_floating_point()
        else v.cuda()
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        depth_output = depth_model(**inputs)

    torch.cuda.synchronize()

    depth_ms = (time.perf_counter() - t) * 1000

    # ---------------- OCR ----------------
    #

    # t = time.perf_counter()
    # ocr_result = None
    # ocr_ms = 0.0

    # ---------------- FUSION ----------------

    t = time.perf_counter()

    # Lightweight placeholder for the actual fusion stage.
    # Accessing the outputs prevents this stage from becoming
    # completely artificial while keeping memory bounded.

    if yolo_result:
        boxes = yolo_result[0].boxes
        _ = len(boxes) if boxes is not None else 0

    if depth_output is not None:
        _ = depth_output.predicted_depth.shape

    if ocr_result is not None:
        _ = ocr_result

    fusion_ms = (time.perf_counter() - t) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    yolo_times.append(yolo_ms)
    depth_times.append(depth_ms)
    ocr_times.append(ocr_ms)
    fusion_times.append(fusion_ms)
    total_times.append(total_ms)

    # CRITICAL: don't retain model outputs/images
    del (
        img,
        rgb,
        inputs,
        yolo_result,
        depth_output,
        ocr_result
    )

    # Periodic cleanup instead of hammering CUDA every frame
    if (i + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

    if (i + 1) % 10 == 0:
        print(
            f"[{i+1:3d}/{len(images)}] "
            f"Total P50 so far: {statistics.median(total_times):.2f} ms"
        )

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL BENCHMARK")
print("=" * 60)

def report(name, values):
    print(f"\n{name}")
    print(f"  P50 : {pct(values, 0.50):8.2f} ms")
    print(f"  P95 : {pct(values, 0.95):8.2f} ms")
    print(f"  P99 : {pct(values, 0.99):8.2f} ms")
    print(f"  Mean: {statistics.mean(values):8.2f} ms")
    print(f"  Min : {min(values):8.2f} ms")
    print(f"  Max : {max(values):8.2f} ms")

report("YOLO", yolo_times)
report("DEPTH", depth_times)
report("OCR", ocr_times)
report("FUSION", fusion_times)
report("TOTAL", total_times)

print("\nEstimated throughput from mean total:")
print(f"  {1000 / statistics.mean(total_times):.2f} FPS")

print("\nCUDA memory:")
print(
    f"  allocated: "
    f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
)
print(
    f"  reserved:  "
    f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB"
)

print("=" * 60)

# Final cleanup
gc.collect()
torch.cuda.empty_cache()
