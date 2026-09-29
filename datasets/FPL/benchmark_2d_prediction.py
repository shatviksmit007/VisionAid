import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("data/trajectories")

FPS = 20.0

# Collision corridor in normalized image coordinates.
# Same x corridor as the previous benchmark.
CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65

# We use the lower portion of the image as the actual
# collision-relevant region.
#
# This is deliberately configurable because FPL itself
# does not define our VisionAid collision corridor.
CORRIDOR_Y_MIN = 0.00
CORRIDOR_Y_MAX = 1.00

HISTORY_LENGTHS = [5, 8, 10, 15, 20]

HORIZONS_SEC = [0.5, 1.0, 1.5]

MIN_TRAJECTORY_LENGTH = 20


# ============================================================
# GEOMETRY
# ============================================================

def inside_corridor(x, y):
    return (
        CORRIDOR_X_MIN <= x <= CORRIDOR_X_MAX
        and
        CORRIDOR_Y_MIN <= y <= CORRIDOR_Y_MAX
    )


def first_corridor_entry(xs, ys):

    for i, (x, y) in enumerate(zip(xs, ys)):

        if inside_corridor(x, y):
            return i

    return None


# ============================================================
# VELOCITY
# ============================================================

def estimate_velocity(values, fps):

    if len(values) < 2:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float64
    )

    t = np.arange(
        len(values),
        dtype=np.float64
    ) / fps

    velocity, _ = np.polyfit(
        t,
        values,
        1
    )

    return float(velocity)


def estimate_2d_velocity(xs, ys):

    vx = estimate_velocity(
        xs,
        FPS
    )

    vy = estimate_velocity(
        ys,
        FPS
    )

    return vx, vy


# ============================================================
# 2D PREDICTION
# ============================================================

def predict_corridor_entry(
    current_x,
    current_y,
    vx,
    vy,
    horizon_sec,
):
    """
    Predict constant-velocity 2D motion.

    Returns:
        predicted_entry
        predicted_x
        predicted_y
        predicted_time
    """

    samples = max(
        2,
        int(horizon_sec * FPS)
    )

    for i in range(samples + 1):

        t = (
            i / samples
        ) * horizon_sec

        predicted_x = (
            current_x + vx * t
        )

        predicted_y = (
            current_y + vy * t
        )

        # Don't allow extrapolation outside image.
        if not (
            0.0 <= predicted_x <= 1.0
            and
            0.0 <= predicted_y <= 1.0
        ):
            continue

        if inside_corridor(
            predicted_x,
            predicted_y
        ):
            return (
                True,
                predicted_x,
                predicted_y,
                t
            )

    return (
        False,
        current_x + vx * horizon_sec,
        current_y + vy * horizon_sec,
        horizon_sec
    )


# ============================================================
# SINGLE TRAJECTORY
# ============================================================

def evaluate_trajectory(
    xs,
    ys,
    history_length,
    horizon_sec,
):

    n = len(xs)

    entry_idx = first_corridor_entry(
        xs,
        ys
    )

    # --------------------------------------------------------
    # NEVER ENTERS
    # --------------------------------------------------------

    if entry_idx is None:

        false_prediction = False

        for current_idx in range(
            history_length - 1,
            n
        ):

            history_start = (
                current_idx
                - history_length
                + 1
            )

            hx = xs[
                history_start:
                current_idx + 1
            ]

            hy = ys[
                history_start:
                current_idx + 1
            ]

            vx, vy = estimate_2d_velocity(
                hx,
                hy
            )

            predicted, _, _, _ = (
                predict_corridor_entry(
                    xs[current_idx],
                    ys[current_idx],
                    vx,
                    vy,
                    horizon_sec
                )
            )

            if predicted:
                false_prediction = True
                break

        if false_prediction:

            return {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
                "warning_frames": None,
            }

        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 1,
            "warning_frames": None,
        }

    # --------------------------------------------------------
    # ENTERS CORRIDOR
    # --------------------------------------------------------

    prediction_idx = None

    # Only predict BEFORE actual entry.
    for current_idx in range(
        history_length - 1,
        entry_idx
    ):

        history_start = (
            current_idx
            - history_length
            + 1
        )

        hx = xs[
            history_start:
            current_idx + 1
        ]

        hy = ys[
            history_start:
            current_idx + 1
        ]

        vx, vy = estimate_2d_velocity(
            hx,
            hy
        )

        predicted, _, _, _ = (
            predict_corridor_entry(
                xs[current_idx],
                ys[current_idx],
                vx,
                vy,
                horizon_sec
            )
        )

        if predicted:

            prediction_idx = (
                current_idx
            )

            break

    # --------------------------------------------------------
    # TRUE POSITIVE
    # --------------------------------------------------------

    if prediction_idx is not None:

        warning_frames = (
            entry_idx
            - prediction_idx
        )

        return {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "warning_frames": warning_frames,
        }

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    return {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 0,
        "warning_frames": None,
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_trajectories():

    files = sorted(
        DATASET_ROOT.glob(
            "*_trajectories_dynamic.json"
        )
    )

    print(
        f"Trajectory files: {len(files)}"
    )

    trajectories = []

    for path in files:

        try:

            with path.open("r") as f:
                data = json.load(f)

        except Exception as e:

            print(
                f"[WARN] {path}: {e}"
            )

            continue

        for track_id, track in data.items():

            traj = track.get(
                "traj_sm"
            )

            if not traj:
                continue

            if len(traj) < MIN_TRAJECTORY_LENGTH:
                continue

            xs = []
            ys = []

            for point in traj:

                if not isinstance(
                    point,
                    (list, tuple)
                ):
                    continue

                if len(point) < 2:
                    continue

                x = float(point[0])
                y = float(point[1])

                # FPL trajectories use the original
                # 1280 x 720 image coordinates.
                x_norm = np.clip(
                    x / 1280.0,
                    0.0,
                    1.0
                )

                y_norm = np.clip(
                    y / 720.0,
                    0.0,
                    1.0
                )

                xs.append(
                    float(x_norm)
                )

                ys.append(
                    float(y_norm)
                )

            if len(xs) < MIN_TRAJECTORY_LENGTH:
                continue

            trajectories.append({
                "file": path.name,
                "track_id": str(track_id),
                "xs": xs,
                "ys": ys,
            })

    return trajectories


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(
    trajectories
):

    results = []

    for history_length in HISTORY_LENGTHS:

        for horizon_sec in HORIZONS_SEC:

            totals = defaultdict(int)

            warnings = []

            for traj in trajectories:

                result = evaluate_trajectory(
                    traj["xs"],
                    traj["ys"],
                    history_length,
                    horizon_sec
                )

                for key in (
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ):

                    totals[key] += (
                        result[key]
                    )

                if (
                    result["warning_frames"]
                    is not None
                ):

                    warnings.append(
                        result[
                            "warning_frames"
                        ] / FPS
                    )

            tp = totals["tp"]
            fp = totals["fp"]
            fn = totals["fn"]
            tn = totals["tn"]

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0.0
            )

            fpr = (
                fp / (fp + tn)
                if fp + tn
                else 0.0
            )

            if warnings:

                median_warning = float(
                    np.median(warnings)
                )

                p95_warning = float(
                    np.percentile(
                        warnings,
                        95
                    )
                )

            else:

                median_warning = float("nan")
                p95_warning = float("nan")

            results.append({
                "history": history_length,
                "horizon": horizon_sec,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "median_warning": median_warning,
                "p95_warning": p95_warning,
            })

    return results


# ============================================================
# PRINT
# ============================================================

def print_results(results):

    print()
    print("=" * 110)
    print(
        "2D ONLINE TRAJECTORY-ENTRY "
        "PREDICTION BENCHMARK"
    )
    print("=" * 110)

    print(
        f"Corridor X: "
        f"[{CORRIDOR_X_MIN:.2f}, "
        f"{CORRIDOR_X_MAX:.2f}]"
    )

    print(
        f"Corridor Y: "
        f"[{CORRIDOR_Y_MIN:.2f}, "
        f"{CORRIDOR_Y_MAX:.2f}]"
    )

    print(
        f"FPS: {FPS}"
    )

    print()

    header = (
        f"{'HIST':>5} "
        f"{'HORIZON':>8} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6} "
        f"{'TN':>6} "
        f"{'PREC':>8} "
        f"{'RECALL':>8} "
        f"{'FPR':>8} "
        f"{'MED WARN':>10} "
        f"{'P95 WARN':>10}"
    )

    print(header)
    print("-" * len(header))

    for r in results:

        print(
            f"{r['history']:5d} "
            f"{r['horizon']:8.2f} "
            f"{r['tp']:6d} "
            f"{r['fp']:6d} "
            f"{r['fn']:6d} "
            f"{r['tn']:6d} "
            f"{r['precision'] * 100:7.2f}% "
            f"{r['recall'] * 100:7.2f}% "
            f"{r['fpr'] * 100:7.2f}% "
            f"{r['median_warning']:9.3f}s "
            f"{r['p95_warning']:9.3f}s"
        )

    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading FPL trajectories..."
    )

    trajectories = (
        load_trajectories()
    )

    print(
        f"Usable trajectories: "
        f"{len(trajectories)}"
    )

    entering = sum(
        first_corridor_entry(
            t["xs"],
            t["ys"]
        ) is not None
        for t in trajectories
    )

    print(
        f"Trajectories entering "
        f"corridor: {entering}"
    )

    print()
    print(
        "Running 2D benchmark..."
    )

    results = run_benchmark(
        trajectories
    )

    print_results(
        results
    )


if __name__ == "__main__":
    main()
