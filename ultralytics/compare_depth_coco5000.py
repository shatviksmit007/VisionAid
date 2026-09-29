import os
import csv
import time
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda"

MODEL_NAME = (
    "depth-anything/Depth-Anything-V2-Small-hf"
)

IMAGE_DIR = "coco100/val2017"

OUTPUT_CSV = "coco_fp32_vs_fp16_results.csv"

WARMUP = 10


# ============================================================
# SETUP
# ============================================================

assert torch.cuda.is_available(), "CUDA unavailable"

print("=" * 70)
print("DEPTH ANYTHING V2 — COCO VAL2017")
print("FP32 vs FP16 — 5000 IMAGE BENCHMARK")
print("=" * 70)

print(
    f"GPU: {torch.cuda.get_device_name(0)}"
)

print(
    f"VRAM: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

print()


# ============================================================
# FIND IMAGES
# ============================================================

files = sorted(
    [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]
)

print(f"Images found: {len(files)}")

if len(files) != 5000:

    print(
        "WARNING: Expected 5000 COCO validation images."
    )

    print(
        f"Found {len(files)} instead."
    )

print()


# ============================================================
# PROCESSOR
# ============================================================

print("Loading processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")
print()


# ============================================================
# LOAD FP32
# ============================================================

print("Loading FP32 model...")

model32 = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,
).to(DEVICE).eval()

print("FP32 loaded.")


# ============================================================
# LOAD FP16
# ============================================================

print("Loading FP16 model...")

model16 = AutoModelForDepthEstimation.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
).to(DEVICE).eval()

print("FP16 loaded.")

print()


# ============================================================
# WARMUP
# ============================================================

print("=" * 70)
print("WARMUP")
print("=" * 70)

sample_image = cv2.imread(files[0])

sample_image = cv2.cvtColor(
    sample_image,
    cv2.COLOR_BGR2RGB,
)

sample_pil = Image.fromarray(
    sample_image
)

sample_inputs = processor(
    images=sample_pil,
    return_tensors="pt",
)

sample_inputs = {
    k: v.to(DEVICE)
    for k, v in sample_inputs.items()
}


with torch.inference_mode():

    for _ in range(WARMUP):

        _ = model32(
            **sample_inputs
        ).predicted_depth

    torch.cuda.synchronize()


with torch.inference_mode():

    for _ in range(WARMUP):

        _ = model16(
            **sample_inputs
        ).predicted_depth

    torch.cuda.synchronize()


print("Warmup complete.")
print()


# ============================================================
# CSV
# ============================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
)

writer = csv.writer(csv_file)

writer.writerow(
    [
        "image",
        "mae",
        "median_error",
        "rmse",
        "max_error",
        "correlation",
        "fp32_ms",
        "fp16_ms",
    ]
)


# ============================================================
# METRIC STORAGE
# ============================================================

maes = []
median_errors = []
rmses = []
max_errors = []
correlations = []

fp32_times = []
fp16_times = []


# ============================================================
# MAIN LOOP
# ============================================================

total_start = time.perf_counter()

processed = 0


for index, path in enumerate(files, 1):

    image = cv2.imread(path)

    if image is None:

        print(
            f"WARNING: Could not read {path}"
        )

        continue


    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


    height, width = image.shape[:2]


    # ========================================================
    # PREPROCESS
    # ========================================================

    inputs = processor(
        images=Image.fromarray(image),
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }


    # ========================================================
    # FP32
    # ========================================================

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        predicted32 = model32(
            **inputs
        ).predicted_depth

    torch.cuda.synchronize()

    fp32_ms = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # FP32 → ORIGINAL RESOLUTION
    # ========================================================

    depth32 = torch.nn.functional.interpolate(
        predicted32.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]


    # ========================================================
    # FP16
    # ========================================================

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        predicted16 = model16(
            **inputs
        ).predicted_depth

    torch.cuda.synchronize()

    fp16_ms = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # FP16 → ORIGINAL RESOLUTION
    # ========================================================

    depth16 = torch.nn.functional.interpolate(
        predicted16.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )[0, 0]


    # ========================================================
    # CONVERT FP16 → FP32 FOR COMPARISON
    # ========================================================

    d32 = depth32.detach().float()

    d16 = depth16.detach().float()


    # ========================================================
    # DIFFERENCE
    # ========================================================

    difference = torch.abs(
        d32 - d16
    )


    # ========================================================
    # METRICS
    # ========================================================

    mae = difference.mean().item()

    median_error = (
        difference.median().item()
    )

    max_error = (
        difference.max().item()
    )

    rmse = torch.sqrt(
        torch.mean(
            (d32 - d16) ** 2
        )
    ).item()


    correlation = torch.corrcoef(
        torch.stack(
            [
                d32.flatten(),
                d16.flatten(),
            ]
        )
    )[0, 1].item()


    # ========================================================
    # STORE
    # ========================================================

    maes.append(mae)

    median_errors.append(
        median_error
    )

    rmses.append(rmse)

    max_errors.append(
        max_error
    )

    correlations.append(
        correlation
    )

    fp32_times.append(
        fp32_ms
    )

    fp16_times.append(
        fp16_ms
    )


    # ========================================================
    # CSV
    # ========================================================

    writer.writerow(
        [
            os.path.basename(path),
            f"{mae:.10f}",
            f"{median_error:.10f}",
            f"{rmse:.10f}",
            f"{max_error:.10f}",
            f"{correlation:.10f}",
            f"{fp32_ms:.4f}",
            f"{fp16_ms:.4f}",
        ]
    )


    processed += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index % 50 == 0
        or index == len(files)
    ):

        elapsed = (
            time.perf_counter()
            - total_start
        )

        rate = (
            processed / elapsed
        )

        remaining = (
            len(files) - processed
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"[{index:4d}/{len(files)}] "
            f"MAE={mae:.6f} "
            f"RMSE={rmse:.6f} "
            f"Corr={correlation:.8f} "
            f"| "
            f"{rate:.2f} img/s "
            f"| ETA {eta/60:.1f} min"
        )


# ============================================================
# CLOSE CSV
# ============================================================

csv_file.close()


# ============================================================
# NUMPY
# ============================================================

maes = np.asarray(maes)

median_errors = np.asarray(
    median_errors
)

rmses = np.asarray(
    rmses
)

max_errors = np.asarray(
    max_errors
)

correlations = np.asarray(
    correlations
)

fp32_times = np.asarray(
    fp32_times
)

fp16_times = np.asarray(
    fp16_times
)


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def print_stats(
    name,
    values,
    precision=8,
):

    print()
    print(name)
    print("-" * 40)

    print(
        f"Mean   : {np.mean(values):.{precision}f}"
    )

    print(
        f"Median : {np.median(values):.{precision}f}"
    )

    print(
        f"P95    : {np.percentile(values,95):.{precision}f}"
    )

    print(
        f"P99    : {np.percentile(values,99):.{precision}f}"
    )

    print(
        f"Max    : {np.max(values):.{precision}f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print("=" * 70)
print("COCO 5000 RESULTS")
print("=" * 70)

print(
    f"Processed: {processed}"
)

print_stats(
    "MEAN ABSOLUTE ERROR",
    maes,
)

print_stats(
    "MEDIAN ABSOLUTE ERROR",
    median_errors,
)

print_stats(
    "RMSE",
    rmses,
)

print_stats(
    "MAXIMUM ABSOLUTE ERROR",
    max_errors,
)

print_stats(
    "CORRELATION",
    correlations,
    precision=10,
)


# ============================================================
# LATENCY
# ============================================================

print()

print("=" * 70)
print("LATENCY")
print("=" * 70)


print()
print("FP32")
print("-" * 40)

print(
    f"P50 : {np.percentile(fp32_times,50):.2f} ms"
)

print(
    f"P95 : {np.percentile(fp32_times,95):.2f} ms"
)

print(
    f"P99 : {np.percentile(fp32_times,99):.2f} ms"
)

print(
    f"Mean: {np.mean(fp32_times):.2f} ms"
)

print(
    f"FPS : {1000 / np.median(fp32_times):.2f}"
)


print()
print("FP16")
print("-" * 40)

print(
    f"P50 : {np.percentile(fp16_times,50):.2f} ms"
)

print(
    f"P95 : {np.percentile(fp16_times,95):.2f} ms"
)

print(
    f"P99 : {np.percentile(fp16_times,99):.2f} ms"
)

print(
    f"Mean: {np.mean(fp16_times):.2f} ms"
)

print(
    f"FPS : {1000 / np.median(fp16_times):.2f}"
)


# ============================================================
# SPEEDUP
# ============================================================

speedup = (
    np.median(fp32_times)
    /
    np.median(fp16_times)
)

print()

print("=" * 70)
print("FP16 SPEEDUP")
print("=" * 70)

print(
    f"Median speedup: {speedup:.2f}x"
)


# ============================================================
# EXTREMES
# ============================================================

worst_mae_index = np.argmax(maes)

worst_corr_index = np.argmin(
    correlations
)

print()

print("=" * 70)
print("EXTREME CASES")
print("=" * 70)

print()

print(
    "Highest MAE:"
)

print(
    os.path.basename(
        files[worst_mae_index]
    )
)

print(
    f"MAE: {maes[worst_mae_index]:.8f}"
)

print()

print(
    "Lowest correlation:"
)

print(
    os.path.basename(
        files[worst_corr_index]
    )
)

print(
    f"Correlation: "
    f"{correlations[worst_corr_index]:.10f}"
)


# ============================================================
# DONE
# ============================================================

print()

print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_CSV}"
)
