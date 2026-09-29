import os

# Limit CPU thread explosion before importing ONNX Runtime / Torch.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["ORT_NUM_THREADS"] = "2"

import gc
import glob
import json
import time

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rapidocr import RapidOCR


# ============================================================
# CONFIG
# ============================================================

COCO_DIR = "/home/shatviksmit/VisionAid/ultralytics/coco100/val2017"

RESULTS_FILE = "benchmark_results.jsonl"
SUMMARY_FILE = "benchmark_summary.json"

YOLO_MODEL = "yolo11n.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

YOLO_IMGSZ = 640
YOLO_CONF = 0.25

DEVICE = "cuda"
DEPTH_DTYPE = torch.float16

PRINT_EVERY = 25


# ============================================================
# HELPERS
# ============================================================

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), p))


def make_stats(values):
    if not values:
        return {}

    arr = np.asarray(values, dtype=np.float64)

    p50 = float(np.percentile(arr, 50))
    mean = float(np.mean(arr))

    return {
        "count": len(values),
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps_from_mean": 1000.0 / mean if mean > 0 else 0,
        "fps_from_p50": 1000.0 / p50 if p50 > 0 else 0,
    }


def print_stats(name, values):

    s = make_stats(values)

    print(
        f"{name:<25} "
        f"P50={s['p50_ms']:7.2f} ms | "
        f"P95={s['p95_ms']:7.2f} ms | "
        f"P99={s['p99_ms']:7.2f} ms | "
        f"Mean={s['mean_ms']:7.2f} ms"
    )


# ============================================================
# FIND DATASET
# ============================================================

if not os.path.isdir(COCO_DIR):
    raise RuntimeError(
        f"COCO directory does not exist:\n{COCO_DIR}"
    )

image_paths = sorted(
    glob.glob(os.path.join(COCO_DIR, "*.jpg"))
)

if not image_paths:
    raise RuntimeError(
        f"No .jpg images found in:\n{COCO_DIR}"
    )

print(f"COCO directory : {COCO_DIR}")
print(f"Images found   : {len(image_paths)}")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLO11n...")
yolo = YOLO(YOLO_MODEL)

print("Loading Depth Anything V2 Small...")
depth_processor = AutoImageProcessor.from_pretrained(
    DEPTH_MODEL
)

depth_model = AutoModelForDepthEstimation.from_pretrained(
    DEPTH_MODEL,
    torch_dtype=DEPTH_DTYPE,
).to(DEVICE)

depth_model.eval()

print("Loading RapidOCR...")
ocr = RapidOCR()

print("\nModels loaded.")


# ============================================================
# WARMUP
# ============================================================

print("\nWarming up...")

warmup_bgr = cv2.imread(image_paths[0])
warmup_rgb = cv2.cvtColor(
    warmup_bgr,
    cv2.COLOR_BGR2RGB
)

for _ in range(5):

    _ = yolo.predict(
        warmup_bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=0,
        verbose=False,
    )

    inputs = depth_processor(
        images=warmup_rgb,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=DEPTH_DTYPE,
    )

    with torch.inference_mode():
        _ = depth_model(
            pixel_values=pixel_values
        )

    del inputs
    del pixel_values

    sync_cuda()

for _ in range(3):
    _ = ocr(warmup_bgr)

del warmup_bgr
del warmup_rgb

gc.collect()
torch.cuda.empty_cache()

print("Warmup complete.")


# ============================================================
# TIMING ARRAYS
#
# These are tiny: 5000 floats ~= negligible RAM.
# We do NOT store image results in memory.
# ============================================================

yolo_times = []
depth_times = []
ocr_times = []
fusion_times = []
full_times = []


# ============================================================
# RESULT FILE
# ============================================================

# Stream results directly to disk.
results_file = open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
)


# ============================================================
# BENCHMARK
# ============================================================

print("\n" + "=" * 100)
print("STARTING BENCHMARK")
print("=" * 100)


for index, image_path in enumerate(
    image_paths,
    start=1
):

    image_name = os.path.basename(image_path)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    bgr = cv2.imread(image_path)

    if bgr is None:
        print(f"WARNING: failed to read {image_path}")
        continue

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = bgr.shape[:2]


    # ========================================================
    # FULL PIPELINE TIMER
    # ========================================================

    sync_cuda()
    full_start = time.perf_counter()


    # ========================================================
    # YOLO
    # ========================================================

    sync_cuda()
    start = time.perf_counter()

    yolo_result = yolo.predict(
        bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=0,
        verbose=False,
    )[0]

    sync_cuda()

    yolo_ms = (
        time.perf_counter() - start
    ) * 1000.0

    yolo_times.append(yolo_ms)


    detections = []

    if yolo_result.boxes is not None:

        for box in yolo_result.boxes:

            cls_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            detections.append({
                "class_id": cls_id,
                "class_name": yolo_result.names[cls_id],
                "confidence": confidence,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
            })


    # ========================================================
    # DEPTH
    # ========================================================

    inputs = depth_processor(
        images=rgb,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=DEPTH_DTYPE,
    )

    sync_cuda()
    start = time.perf_counter()

    with torch.inference_mode():

        depth_output = depth_model(
            pixel_values=pixel_values
        )

        predicted_depth = (
            depth_output.predicted_depth
        )

        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)

    sync_cuda()

    depth_ms = (
        time.perf_counter() - start
    ) * 1000.0

    depth_times.append(depth_ms)


    # Convert only once.
    depth_cpu = (
        depth_map
        .float()
        .cpu()
        .numpy()
    )


    # ========================================================
    # FUSION
    # ========================================================

    start = time.perf_counter()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            detection["relative_depth"] = None
            continue

        # Central 50% of bbox.
        cx1 = x1 + box_width // 4
        cx2 = x2 - box_width // 4

        cy1 = y1 + box_height // 4
        cy2 = y2 - box_height // 4

        roi = depth_cpu[
            cy1:cy2,
            cx1:cx2
        ]

        if roi.size:
            detection["relative_depth"] = float(
                np.median(roi)
            )
        else:
            detection["relative_depth"] = None

    fusion_ms = (
        time.perf_counter() - start
    ) * 1000.0

    fusion_times.append(fusion_ms)


    # ========================================================
    # OCR
    # ========================================================

    start = time.perf_counter()

    ocr_result = ocr(bgr)

    ocr_ms = (
        time.perf_counter() - start
    ) * 1000.0

    ocr_times.append(ocr_ms)


    ocr_detections = []

    if (
        ocr_result is not None
        and ocr_result.txts is not None
    ):

        for box, text, score in zip(
            ocr_result.boxes,
            ocr_result.txts,
            ocr_result.scores,
        ):

            ocr_detections.append({
                "text": str(text),
                "confidence": float(score),
                "box": np.asarray(
                    box
                ).tolist(),
            })


    # ========================================================
    # FULL PIPELINE TIME
    # ========================================================

    sync_cuda()

    full_ms = (
        time.perf_counter() - full_start
    ) * 1000.0

    full_times.append(full_ms)


    # ========================================================
    # SAVE THIS IMAGE IMMEDIATELY
    # ========================================================

    record = {
        "image": image_name,

        "resolution": [
            width,
            height,
        ],

        "timing_ms": {
            "yolo": yolo_ms,
            "depth": depth_ms,
            "fusion": fusion_ms,
            "ocr": ocr_ms,
            "full_pipeline": full_ms,
        },

        "yolo": {
            "num_detections": len(
                detections
            ),
            "detections": detections,
        },

        "depth": {
            "min": float(
                np.min(depth_cpu)
            ),
            "max": float(
                np.max(depth_cpu)
            ),
            "mean": float(
                np.mean(depth_cpu)
            ),
            "median": float(
                np.median(depth_cpu)
            ),
        },

        "ocr": {
            "num_text_regions": len(
                ocr_detections
            ),
            "detections": ocr_detections,
        },
    }

    results_file.write(
        json.dumps(
            record,
            ensure_ascii=False
        ) + "\n"
    )

    # Flush periodically, not every image.
    if index % 25 == 0:
        results_file.flush()


    # ========================================================
    # CLEANUP
    # ========================================================

    del (
        bgr,
        rgb,
        inputs,
        pixel_values,
        depth_output,
        predicted_depth,
        depth_map,
        depth_cpu,
        yolo_result,
        ocr_result,
        detections,
        ocr_detections,
        record,
    )

    # Don't call this every image; periodic GC is enough.
    if index % 50 == 0:
        gc.collect()
        torch.cuda.empty_cache()


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index <= 5
        or index % PRINT_EVERY == 0
    ):

        print(
            f"\n[{index}/{len(image_paths)}] "
            f"{image_name}"
        )

        print(
            f"  YOLO   : {yolo_ms:8.2f} ms"
        )

        print(
            f"  Depth  : {depth_ms:8.2f} ms"
        )

        print(
            f"  Fusion : {fusion_ms:8.2f} ms"
        )

        print(
            f"  OCR    : {ocr_ms:8.2f} ms"
        )

        print(
            f"  TOTAL  : {full_ms:8.2f} ms "
            f"({1000/full_ms:6.2f} FPS)"
        )

        print(
            f"  Objects: {len(detections)} | "
            f"Text regions: "
            f"{len(ocr_detections)}"
        )

        print("\nRunning statistics:")

        print_stats(
            "YOLO",
            yolo_times
        )

        print_stats(
            "Depth",
            depth_times
        )

        print_stats(
            "Fusion",
            fusion_times
        )

        print_stats(
            "OCR",
            ocr_times
        )

        print_stats(
            "FULL PIPELINE",
            full_times
        )


# ============================================================
# CLOSE RESULTS
# ============================================================

results_file.flush()
results_file.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {
    "dataset": {
        "directory": COCO_DIR,
        "images_requested": len(
            image_paths
        ),
        "images_processed": len(
            full_times
        ),
    },

    "models": {
        "yolo": YOLO_MODEL,
        "depth": DEPTH_MODEL,
        "depth_dtype": "float16",
        "ocr": "RapidOCR",
    },

    "device": DEVICE,

    "timing": {
        "yolo": make_stats(
            yolo_times
        ),

        "depth": make_stats(
            depth_times
        ),

        "fusion": make_stats(
            fusion_times
        ),

        "ocr": make_stats(
            ocr_times
        ),

        "full_pipeline": make_stats(
            full_times
        ),
    },
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n\n")
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

print(
    f"\nImages processed: "
    f"{len(full_times)}"
)

print()

print_stats(
    "YOLO",
    yolo_times
)

print_stats(
    "Depth",
    depth_times
)

print_stats(
    "Fusion",
    fusion_times
)

print_stats(
    "OCR",
    ocr_times
)

print_stats(
    "FULL PIPELINE",
    full_times
)

print("\nFiles:")
print(f"  Per-image results : {RESULTS_FILE}")
print(f"  Summary           : {SUMMARY_FILE}")

print("=" * 100)
