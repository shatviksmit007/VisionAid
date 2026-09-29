import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("data/trajectories")

FPS = 20.0

# Same initial corridor used in our previous FPL analysis.
CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65

# History lengths to test.
HISTORY_LENGTHS = [5, 8, 10, 15, 20]

# How far into the future we predict.
HORIZONS_SEC = [0.5, 1.0, 1.5]

# Minimum trajectory length.
MIN_TRAJECTORY_LENGTH = 20

# A future point must actually enter this corridor to count
# as a positive event.
ENTRY_MARGIN = 0.0


# ============================================================
# BASIC GEOMETRY
# ============================================================

def inside_corridor(x):
    return (
        CORRIDOR_X_MIN - ENTRY_MARGIN
        <= x
        <= CORRIDOR_X_MAX + ENTRY_MARGIN
    )


def first_corridor_entry(xs):
    """
    Return the first trajectory index where the pedestrian
    enters the corridor.

    Returns None if the trajectory never enters.
    """
    for i, x in enumerate(xs):
        if inside_corridor(x):
            return i
    return None


# ============================================================
# VELOCITY ESTIMATION
# ============================================================

def estimate_velocity(xs, fps):
    """
    Linear regression over the supplied history.

    Returns normalized x velocity / second.
    """
    if len(xs) < 2:
        return 0.0

    xs = np.asarray(xs, dtype=np.float64)

    t = np.arange(len(xs), dtype=np.float64) / fps

    # x = vt + b
    velocity, _ = np.polyfit(t, xs, 1)

    return float(velocity)


# ============================================================
# ONLINE PREDICTION
# ============================================================

def predict_entry(
    history_x,
    current_x,
    velocity,
    horizon_sec,
):
    """
    Predict whether the pedestrian will enter the corridor
    within horizon_sec.

    Uses ONLY information available at the current frame.
    """

    # Sample the predicted trajectory instead of checking
    # only the endpoint. This catches a trajectory that enters
    # and then leaves the corridor within the horizon.
    samples = max(2, int(horizon_sec * FPS))

    for i in range(samples + 1):
        t = (i / samples) * horizon_sec

        predicted_x = current_x + velocity * t

        if inside_corridor(predicted_x):
            return True, predicted_x, t

    return False, current_x + velocity * horizon_sec, horizon_sec


# ============================================================
# PER-TRAJECTORY EVALUATION
# ============================================================

def evaluate_trajectory(
    xs,
    history_length,
    horizon_sec,
):
    """
    Evaluate online prediction for one trajectory.

    Returns a dictionary containing:
      TP
      FP
      FN
      TN
      prediction_frame
      actual_entry_frame
      warning_frames
    """

    n = len(xs)

    entry_idx = first_corridor_entry(xs)

    # --------------------------------------------------------
    # Case 1: trajectory never enters corridor.
    # We can evaluate the whole trajectory for false positives.
    # --------------------------------------------------------

    if entry_idx is None:

        fp = 0

        for current_idx in range(history_length - 1, n):

            history = xs[
                current_idx - history_length + 1:
                current_idx + 1
            ]

            current_x = xs[current_idx]

            velocity = estimate_velocity(
                history,
                FPS,
            )

            predicted, _, _ = predict_entry(
                history,
                current_x,
                velocity,
                horizon_sec,
            )

            if predicted:
                fp += 1

        if fp > 0:
            # We count this trajectory as one false-positive event,
            # rather than allowing one trajectory to generate hundreds
            # of false positives.
            return {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
                "prediction_frame": None,
                "actual_entry_frame": None,
                "warning_frames": None,
            }

        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 1,
            "prediction_frame": None,
            "actual_entry_frame": None,
            "warning_frames": None,
        }

    # --------------------------------------------------------
    # Case 2: trajectory enters corridor.
    # --------------------------------------------------------

    prediction_idx = None

    # Only evaluate frames BEFORE actual entry.
    #
    # Once the pedestrian is already inside the corridor,
    # predicting "they will enter" is no longer a prediction.
    max_idx = min(entry_idx, n - 1)

    for current_idx in range(
        history_length - 1,
        max_idx
    ):

        history = xs[
            current_idx - history_length + 1:
            current_idx + 1
        ]

        current_x = xs[current_idx]

        velocity = estimate_velocity(
            history,
            FPS,
        )

        predicted, _, _ = predict_entry(
            history,
            current_x,
            velocity,
            horizon_sec,
        )

        if predicted:
            prediction_idx = current_idx
            break

    # --------------------------------------------------------
    # True positive
    # --------------------------------------------------------

    if prediction_idx is not None:

        warning_frames = entry_idx - prediction_idx

        return {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "prediction_frame": prediction_idx,
            "actual_entry_frame": entry_idx,
            "warning_frames": warning_frames,
        }

    # --------------------------------------------------------
    # False negative
    # --------------------------------------------------------

    return {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 0,
        "prediction_frame": None,
        "actual_entry_frame": entry_idx,
        "warning_frames": None,
    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_trajectories():

    files = sorted(
        DATASET_ROOT.glob("*_trajectories_dynamic.json")
    )

    print(f"Trajectory files: {len(files)}")

    trajectories = []

    for path in files:

        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        for track_id, track in data.items():

            traj = track.get("traj_sm")

            if not traj:
                continue

            if len(traj) < MIN_TRAJECTORY_LENGTH:
                continue

            xs = []

            for point in traj:

                if not isinstance(point, (list, tuple)):
                    continue

                if len(point) < 2:
                    continue

                x = float(point[0])

                # FPL trajectory x coordinates are pixel coordinates.
                # The original video width is 1280.
                x_norm = x / 1280.0

                x_norm = float(
                    np.clip(x_norm, 0.0, 1.0)
                )

                xs.append(x_norm)

            if len(xs) < MIN_TRAJECTORY_LENGTH:
                continue

            trajectories.append({
                "file": path.name,
                "track_id": str(track_id),
                "xs": xs,
            })

    return trajectories


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(trajectories):

    results = []

    for history_length in HISTORY_LENGTHS:

        for horizon_sec in HORIZONS_SEC:

            totals = defaultdict(int)

            warning_times = []

            total_entering = 0

            for traj in trajectories:

                xs = traj["xs"]

                result = evaluate_trajectory(
                    xs,
                    history_length,
                    horizon_sec,
                )

                for key in (
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                ):
                    totals[key] += result[key]

                if result["actual_entry_frame"] is not None:
                    total_entering += 1

                if result["warning_frames"] is not None:

                    warning_sec = (
                        result["warning_frames"] / FPS
                    )

                    warning_times.append(
                        warning_sec
                    )

            tp = totals["tp"]
            fp = totals["fp"]
            fn = totals["fn"]
            tn = totals["tn"]

            precision = (
                tp / (tp + fp)
                if tp + fp > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn > 0
                else 0.0
            )

            fpr = (
                fp / (fp + tn)
                if fp + tn > 0
                else 0.0
            )

            if warning_times:

                median_warning = float(
                    np.median(warning_times)
                )

                p95_warning = float(
                    np.percentile(
                        warning_times,
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
                "entering": total_entering,
            })

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print()
    print("=" * 105)
    print("ONLINE TRAJECTORY-ENTRY PREDICTION BENCHMARK")
    print("=" * 105)

    print(
        f"Corridor: [{CORRIDOR_X_MIN:.2f}, "
        f"{CORRIDOR_X_MAX:.2f}]"
    )

    print(f"FPS: {FPS}")
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

    print("=" * 105)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading FPL trajectories...")

    trajectories = load_trajectories()

    print(
        f"Usable trajectories: {len(trajectories)}"
    )

    entering = 0

    for traj in trajectories:

        if first_corridor_entry(traj["xs"]) is not None:
            entering += 1

    print(
        f"Trajectories entering corridor: {entering}"
    )

    print()
    print("Running benchmark...")

    results = run_benchmark(trajectories)

    print_results(results)


if __name__ == "__main__":
    main()
