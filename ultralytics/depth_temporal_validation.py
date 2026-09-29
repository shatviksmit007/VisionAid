import argparse
import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# Temporal windows we want to compare.
HISTORY_LENGTHS = [5, 10, 15]

# Central ROI.
# We deliberately avoid the extreme image boundaries.
ROI_X_MIN = 0.25
ROI_X_MAX = 0.75
ROI_Y_MIN = 0.25
ROI_Y_MAX = 0.75

# Warmup frames.
WARMUP = 20

# Minimum absolute velocity to count as a sign.
# For analysis only — NOT our final threshold.
SIGN_EPSILON = 1e-6


# ============================================================
# DEPTH
# ============================================================

def load_model():

    print("Loading Depth Anything V2 Small...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model = model.cuda()
    model.eval()

    print("Model loaded.")

    return processor, model


@torch.inference_mode()
def estimate_depth(
    frame,
    processor,
    model,
):

    # BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    inputs = processor(
        images=rgb,
        return_tensors="pt",
    )

    inputs = {
        k: v.cuda()
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    predicted_depth = (
        outputs.predicted_depth
    )

    # [1,H,W]
    predicted_depth = (
        predicted_depth
        .squeeze(0)
        .float()
    )

    # Resize to original frame size.
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0).unsqueeze(0),
        size=frame.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    depth = predicted_depth.cpu().numpy()

    return depth


# ============================================================
# ROI
# ============================================================

def central_roi_depth(depth):

    h, w = depth.shape

    x1 = int(
        ROI_X_MIN * w
    )

    x2 = int(
        ROI_X_MAX * w
    )

    y1 = int(
        ROI_Y_MIN * h
    )

    y2 = int(
        ROI_Y_MAX * h
    )

    roi = depth[
        y1:y2,
        x1:x2
    ]

    # Median is deliberately used because a few extreme
    # depth pixels shouldn't dominate the temporal signal.
    return float(
        np.median(roi)
    )


# ============================================================
# TEMPORAL VELOCITY
# ============================================================

def estimate_velocity(
    values,
    timestamps,
):

    if len(values) < 2:
        return None

    values = np.asarray(
        values,
        dtype=np.float64
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    # Relative depth per second.
    velocity, _ = np.polyfit(
        timestamps,
        values,
        1,
    )

    return float(velocity)


# ============================================================
# STATISTICS
# ============================================================

def percentile(values, p):

    if values is None or len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(values),
            p
        )
    )


def print_stats(
    history_length,
    depths,
    velocities,
):

    depths = np.asarray(
        depths,
        dtype=np.float64
    )

    velocities = np.asarray(
        [
            v
            for v in velocities
            if v is not None
            and np.isfinite(v)
        ],
        dtype=np.float64
    )

    if len(depths) < 2:
        return

    depth_deltas = np.diff(
        depths
    )

    abs_velocities = np.abs(
        velocities
    )

    # Sign flips.
    signs = np.where(
        velocities > SIGN_EPSILON,
        1,
        np.where(
            velocities < -SIGN_EPSILON,
            -1,
            0,
        ),
    )

    nonzero_signs = (
        signs[signs != 0]
    )

    if len(nonzero_signs) >= 2:

        sign_flips = int(
            np.sum(
                nonzero_signs[1:]
                != nonzero_signs[:-1]
            )
        )

    else:

        sign_flips = 0

    print()
    print("-" * 70)
    print(
        f"HISTORY = {history_length} frames "
        f"({history_length / 30:.3f}s @ 30 FPS)"
    )
    print("-" * 70)

    print(
        f"Depth mean              : "
        f"{np.mean(depths):.6f}"
    )

    print(
        f"Depth std               : "
        f"{np.std(depths):.6f}"
    )

    print(
        f"|frame Δdepth| median   : "
        f"{np.median(np.abs(depth_deltas)):.6f}"
    )

    print(
        f"|frame Δdepth| P95      : "
        f"{percentile(np.abs(depth_deltas), 95):.6f}"
    )

    print(
        f"VREL count              : "
        f"{len(velocities)}"
    )

    if len(velocities) > 0:

        print(
            f"VREL mean               : "
            f"{np.mean(velocities):+.6f}"
        )

        print(
            f"VREL median             : "
            f"{np.median(velocities):+.6f}"
        )

        print(
            f"VREL std                : "
            f"{np.std(velocities):.6f}"
        )

        print(
            f"|VREL| P95             : "
            f"{percentile(abs_velocities, 95):.6f}"
        )

        print(
            f"|VREL| P99             : "
            f"{percentile(abs_velocities, 99):.6f}"
        )

        print(
            f"VREL min               : "
            f"{np.min(velocities):+.6f}"
        )

        print(
            f"VREL max               : "
            f"{np.max(velocities):+.6f}"
        )

    print(
        f"Sign flips              : "
        f"{sign_flips}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="depth_temporal_validation.csv",
    )

    args = parser.parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        raise FileNotFoundError(
            video_path
        )

    processor, model = load_model()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print()
    print(
        f"Video: {video_path}"
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print()
    print(
        f"Warming up ({WARMUP} frames)..."
    )

    warmup_count = 0

    while warmup_count < WARMUP:

        ok, frame = cap.read()

        if not ok:
            break

        estimate_depth(
            frame,
            processor,
            model,
        )

        warmup_count += 1

    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    depth_histories = {
        h: deque(maxlen=h)
        for h in HISTORY_LENGTHS
    }

    time_histories = {
        h: deque(maxlen=h)
        for h in HISTORY_LENGTHS
    }

    all_depths = []
    velocity_history = {
        h: []
        for h in HISTORY_LENGTHS
    }

    rows = []

    frame_idx = WARMUP

    start = time.perf_counter()

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_idx / fps
            if fps > 0
            else frame_idx / 30.0
        )

        depth = estimate_depth(
            frame,
            processor,
            model,
        )

        roi_depth = (
            central_roi_depth(depth)
        )

        all_depths.append(
            roi_depth
        )

        row = {
            "frame": frame_idx,
            "timestamp": timestamp,
            "depth": roi_depth,
        }

        for history_length in HISTORY_LENGTHS:

            depth_histories[
                history_length
            ].append(
                roi_depth
            )

            time_histories[
                history_length
            ].append(
                timestamp
            )

            velocity = estimate_velocity(
                depth_histories[
                    history_length
                ],
                time_histories[
                    history_length
                ],
            )

            velocity_history[
                history_length
            ].append(
                velocity
            )

            row[
                f"vrel_{history_length}"
            ] = (
                ""
                if velocity is None
                else velocity
            )

        rows.append(row)

        frame_idx += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    cap.release()

    print()
    print(
        f"Processed {len(rows)} frames "
        f"in {elapsed:.2f}s"
    )

    if elapsed > 0:

        print(
            f"Processing FPS: "
            f"{len(rows) / elapsed:.2f}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    for history_length in HISTORY_LENGTHS:

        print_stats(
            history_length,
            all_depths,
            velocity_history[
                history_length
            ],
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    fieldnames = [
        "frame",
        "timestamp",
        "depth",
    ]

    fieldnames += [
        f"vrel_{h}"
        for h in HISTORY_LENGTHS
    ]

    with open(
        args.output,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
