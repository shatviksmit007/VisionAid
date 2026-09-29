import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(__file__).resolve().parent / "data" / "trajectories"

FPS = 20.0

CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65

HISTORY_LENGTHS = [5, 8, 10, 15, 20]
HORIZONS_SEC = [0.5, 1.0, 1.5]

# Thresholds to test.
# Velocity is normalized image-units / second.
VELOCITY_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0]

MIN_TRAJECTORY_LENGTH = 20


# ============================================================
# GEOMETRY
# ============================================================

def inside_corridor(x):
    return CORRIDOR_X_MIN <= x <= CORRIDOR_X_MAX


def first_corridor_entry(xs):
    for i, x in enumerate(xs):
        if inside_corridor(x):
            return i
    return None


# ============================================================
# VELOCITY
# ============================================================

def estimate_velocity(xs):
    if len(xs) < 2:
        return 0.0

    xs = np.asarray(xs, dtype=np.float64)
    t = np.arange(len(xs), dtype=np.float64) / FPS

    velocity, _ = np.polyfit(t, xs, 1)

    return float(velocity)


# ============================================================
# PREDICTION
# ============================================================

def predict_entry(
    current_x,
    velocity,
    horizon_sec,
    velocity_threshold,
):
    """
    Predict whether the object will enter the corridor.

    If the estimated velocity magnitude is below the threshold,
    treat the motion as camera/noise/stationary motion and do
    not issue a prediction.
    """

    if abs(velocity) < velocity_threshold:
        return False

    samples = max(2, int(horizon_sec * FPS))

    for i in range(samples + 1):

        t = (i / samples) * horizon_sec

        predicted_x = current_x + velocity * t

        if inside_corridor(predicted_x):
            return True

    return False


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

def load_trajectories():

    files = sorted(
        DATASET_ROOT.glob("*_trajectories_dynamic.json")
    )

    print("Loading FPL trajectories...")
    print(f"Trajectory files: {len(files)}")

    trajectories = []

    for path in files:

        try:
            with open(path, "r") as f:
                data = json.load(f)

        except Exception as e:
            print(f"WARNING: failed to read {path}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        for traj_id, record in data.items():

            if not isinstance(record, dict):
                continue

            # The FPL files contain the smoothed trajectory.
            points = record.get("traj_sm")

            if points is None:
                points = record.get("traj")

            if not isinstance(points, list):
                continue

            if len(points) < MIN_TRAJECTORY_LENGTH:
                continue

            xs = []

            for point in points:

                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                ):
                    try:
                        xs.append(float(point[0]))
                    except (TypeError, ValueError):
                        pass

            if len(xs) < MIN_TRAJECTORY_LENGTH:
                continue

            # ------------------------------------------------
            # Normalize x coordinates.
            #
            # FPL coordinates are pixel coordinates.
            # Use the trajectory's available image width.
            #
            # We infer width from the maximum x coordinate.
            # ------------------------------------------------

            xs = np.asarray(xs, dtype=np.float64)

            # FPL videos are 1920 px wide.
            # Keep this explicit so the experiment is reproducible.
            IMAGE_WIDTH = 1920.0

            xs = xs / IMAGE_WIDTH

            trajectories.append(xs)

    print(f"Usable trajectories: {len(trajectories)}")

    entered = sum(
        first_corridor_entry(xs) is not None
        for xs in trajectories
    )

    print(f"Trajectories entering corridor: {entered}")

    return trajectories


# ============================================================
# EVALUATION
# ============================================================

def evaluate_trajectory(
    xs,
    history_length,
    horizon_sec,
    velocity_threshold,
):

    n = len(xs)

    entry_idx = first_corridor_entry(xs)

    # --------------------------------------------------------
    # NEVER ENTERS
    # --------------------------------------------------------

    if entry_idx is None:

        false_positive = False

        for current_idx in range(
            history_length - 1,
            n,
        ):

            history = xs[
                current_idx - history_length + 1:
                current_idx + 1
            ]

            velocity = estimate_velocity(history)

            predicted = predict_entry(
                xs[current_idx],
                velocity,
                horizon_sec,
                velocity_threshold,
            )

            if predicted:
                false_positive = True
                break

        if false_positive:
            return 0, 1, 0, 0, None

        return 0, 0, 0, 1, None

    # --------------------------------------------------------
    # ENTERS CORRIDOR
    # --------------------------------------------------------

    prediction_idx = None

    for current_idx in range(
        history_length - 1,
        entry_idx,
    ):

        history = xs[
            current_idx - history_length + 1:
            current_idx + 1
        ]

        velocity = estimate_velocity(history)

        predicted = predict_entry(
            xs[current_idx],
            velocity,
            horizon_sec,
            velocity_threshold,
        )

        if predicted:
            prediction_idx = current_idx
            break

    # --------------------------------------------------------
    # TRUE POSITIVE
    # --------------------------------------------------------

    if prediction_idx is not None:

        warning_frames = entry_idx - prediction_idx

        return (
            1,
            0,
            0,
            0,
            warning_frames,
        )

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    return (
        0,
        0,
        1,
        0,
        None,
    )


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(
    trajectories,
    history_length,
    horizon_sec,
    threshold,
):

    tp = fp = fn = tn = 0
    warnings = []

    for xs in trajectories:

        result = evaluate_trajectory(
            xs,
            history_length,
            horizon_sec,
            threshold,
        )

        r_tp, r_fp, r_fn, r_tn, warning = result

        tp += r_tp
        fp += r_fp
        fn += r_fn
        tn += r_tn

        if warning is not None:
            warnings.append(
                warning / FPS
            )

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

    if warnings:

        median_warning = float(
            np.median(warnings)
        )

        p95_warning = float(
            np.percentile(warnings, 95)
        )

    else:

        median_warning = 0.0
        p95_warning = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "median_warning": median_warning,
        "p95_warning": p95_warning,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    trajectories = load_trajectories()

    if not trajectories:

        print("\nERROR: No usable trajectories found.")
        return

    print()
    print("=" * 115)
    print("VELOCITY-THRESHOLD ONLINE TRAJECTORY-ENTRY BENCHMARK")
    print("=" * 115)

    print(
        f"Corridor X: "
        f"[{CORRIDOR_X_MIN:.2f}, {CORRIDOR_X_MAX:.2f}"
        f"]"
    )

    print(f"FPS: {FPS:.1f}")
    print()

    for threshold in VELOCITY_THRESHOLDS:

        print()
        print(
            "#" * 115
        )
        print(
            f"VELOCITY THRESHOLD = {threshold:.2f}"
        )
        print(
            "#" * 115
        )

        print(
            f"{'HIST':>6}"
            f"{'HORIZON':>10}"
            f"{'TP':>7}"
            f"{'FP':>7}"
            f"{'FN':>7}"
            f"{'TN':>7}"
            f"{'PREC':>10}"
            f"{'RECALL':>10}"
            f"{'FPR':>10}"
            f"{'MED WARN':>12}"
            f"{'P95 WARN':>12}"
        )

        print("-" * 115)

        for history_length in HISTORY_LENGTHS:

            for horizon_sec in HORIZONS_SEC:

                stats = run_benchmark(
                    trajectories,
                    history_length,
                    horizon_sec,
                    threshold,
                )

                print(
                    f"{history_length:6d}"
                    f"{horizon_sec:10.2f}"
                    f"{stats['tp']:7d}"
                    f"{stats['fp']:7d}"
                    f"{stats['fn']:7d}"
                    f"{stats['tn']:7d}"
                    f"{stats['precision'] * 100:9.2f}%"
                    f"{stats['recall'] * 100:9.2f}%"
                    f"{stats['fpr'] * 100:9.2f}%"
                    f"{stats['median_warning']:11.3f}s"
                    f"{stats['p95_warning']:11.3f}s"
                )


if __name__ == "__main__":
    main()
