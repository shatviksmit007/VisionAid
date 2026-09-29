import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "trajectories"

FPS = 20.0

FRAME_WIDTH = 1920.0

CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65

HISTORY_LENGTHS = [5, 8, 10, 15, 20]
HORIZONS_SEC = [0.5, 1.0, 1.5]

MIN_TRAJECTORY_LENGTH = 20


# ============================================================
# CORRIDOR
# ============================================================

def inside_corridor(x):
    return CORRIDOR_X_MIN <= x <= CORRIDOR_X_MAX


def first_corridor_entry(xs):
    for i, x in enumerate(xs):
        if inside_corridor(x):
            return i
    return None


# ============================================================
# CONSTANT-ACCELERATION ESTIMATION
# ============================================================

def estimate_motion(xs):
    """
    Fit:

        x(t) = 0.5*a*t^2 + v*t + b

    over the supplied history.

    Returns:
        velocity
        acceleration
    """

    if len(xs) < 3:
        return 0.0, 0.0

    xs = np.asarray(xs, dtype=np.float64)

    t = np.arange(len(xs), dtype=np.float64) / FPS

    # x = c2*t^2 + c1*t + c0
    coeffs = np.polyfit(t, xs, 2)

    c2, c1, _ = coeffs

    acceleration = 2.0 * c2
    velocity = c1

    return float(velocity), float(acceleration)


# ============================================================
# PREDICTION
# ============================================================

def predict_entry(current_x, velocity, acceleration, horizon_sec):

    steps = max(2, int(horizon_sec * FPS))

    for i in range(steps + 1):

        t = (i / steps) * horizon_sec

        predicted_x = (
            current_x
            + velocity * t
            + 0.5 * acceleration * t * t
        )

        if inside_corridor(predicted_x):
            return True, t

    return False, horizon_sec


# ============================================================
# LOADER
# ============================================================

def load_trajectories():

    files = sorted(
        DATASET_ROOT.glob("*_trajectories_dynamic.json")
    )

    print("Loading FPL trajectories...")
    print(f"Trajectory files: {len(files)}")

    trajectories = []

    for file in files:

        try:
            with open(file, "r") as f:
                data = json.load(f)

        except Exception as e:
            print(f"WARNING: Could not read {file}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        for traj_id, record in data.items():

            if not isinstance(record, dict):
                continue

            points = record.get("traj_sm")

            if not isinstance(points, list):
                continue

            if len(points) < MIN_TRAJECTORY_LENGTH:
                continue

            xs = []

            valid = True

            for point in points:

                if (
                    not isinstance(point, (list, tuple))
                    or len(point) < 2
                ):
                    valid = False
                    break

                try:
                    x = float(point[0])
                except Exception:
                    valid = False
                    break

                x_norm = x / FRAME_WIDTH

                xs.append(x_norm)

            if not valid:
                continue

            xs = np.asarray(xs, dtype=np.float64)

            if np.any(~np.isfinite(xs)):
                continue

            if np.any(xs < -0.05) or np.any(xs > 1.05):
                continue

            trajectories.append({
                "file": file.name,
                "id": traj_id,
                "x": xs,
            })

    print(f"Usable trajectories: {len(trajectories)}")

    entered = sum(
        first_corridor_entry(t["x"]) is not None
        for t in trajectories
    )

    print(f"Trajectories entering corridor: {entered}")

    return trajectories


# ============================================================
# EVALUATION
# ============================================================

def evaluate_trajectory(xs, history_length, horizon_sec):

    n = len(xs)

    entry_idx = first_corridor_entry(xs)

    # --------------------------------------------------------
    # NEVER ENTERS
    # --------------------------------------------------------

    if entry_idx is None:

        any_prediction = False

        for current_idx in range(
            history_length - 1,
            n
        ):

            history = xs[
                current_idx - history_length + 1:
                current_idx + 1
            ]

            velocity, acceleration = estimate_motion(history)

            predicted, _ = predict_entry(
                xs[current_idx],
                velocity,
                acceleration,
                horizon_sec,
            )

            if predicted:
                any_prediction = True
                break

        if any_prediction:
            return {
                "tp": 0,
                "fp": 1,
                "fn": 0,
                "tn": 0,
                "warning": None,
            }

        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 1,
            "warning": None,
        }

    # --------------------------------------------------------
    # ENTERS
    # --------------------------------------------------------

    prediction_idx = None

    for current_idx in range(
        history_length - 1,
        entry_idx
    ):

        history = xs[
            current_idx - history_length + 1:
            current_idx + 1
        ]

        velocity, acceleration = estimate_motion(history)

        predicted, _ = predict_entry(
            xs[current_idx],
            velocity,
            acceleration,
            horizon_sec,
        )

        if predicted:
            prediction_idx = current_idx
            break

    # --------------------------------------------------------
    # TRUE POSITIVE
    # --------------------------------------------------------

    if prediction_idx is not None:

        warning_frames = entry_idx - prediction_idx

        return {
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "warning": warning_frames / FPS,
        }

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    return {
        "tp": 0,
        "fp": 0,
        "fn": 1,
        "tn": 0,
        "warning": None,
    }


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(trajectories):

    print()
    print("Running constant-acceleration benchmark...")
    print()

    print("=" * 110)
    print("CONSTANT-ACCELERATION TRAJECTORY-ENTRY PREDICTION")
    print("=" * 110)

    print(
        f"Corridor X: "
        f"[{CORRIDOR_X_MIN:.2f}, {CORRIDOR_X_MAX:.2f}]"
    )

    print(f"FPS: {FPS}")

    print()

    print(
        f"{'HIST':>5} "
        f"{'HORIZON':>10} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6} "
        f"{'TN':>6} "
        f"{'PREC':>9} "
        f"{'RECALL':>9} "
        f"{'FPR':>8} "
        f"{'MED WARN':>10} "
        f"{'P95 WARN':>10}"
    )

    print("-" * 110)

    for history_length in HISTORY_LENGTHS:

        for horizon_sec in HORIZONS_SEC:

            totals = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }

            warnings = []

            for trajectory in trajectories:

                result = evaluate_trajectory(
                    trajectory["x"],
                    history_length,
                    horizon_sec,
                )

                for key in totals:
                    totals[key] += result[key]

                if result["warning"] is not None:
                    warnings.append(result["warning"])

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

            if warnings:
                med_warning = np.median(warnings)
                p95_warning = np.percentile(warnings, 95)
            else:
                med_warning = 0.0
                p95_warning = 0.0

            print(
                f"{history_length:5d} "
                f"{horizon_sec:10.2f} "
                f"{tp:6d} "
                f"{fp:6d} "
                f"{fn:6d} "
                f"{tn:6d} "
                f"{precision * 100:8.2f}% "
                f"{recall * 100:8.2f}% "
                f"{fpr * 100:7.2f}% "
                f"{med_warning:9.3f}s "
                f"{p95_warning:9.3f}s"
            )

    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    trajectories = load_trajectories()

    if not trajectories:
        print("ERROR: No usable trajectories found.")
        return

    run_benchmark(trajectories)


if __name__ == "__main__":
    main()
