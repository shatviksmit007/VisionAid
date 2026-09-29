import json
from pathlib import Path

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("data/trajectories")

FPS = 20.0

# FPL trajectory coordinates are pixel coordinates.
FRAME_WIDTH = 1920.0

# Normalized corridor.
CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65

HISTORY_LENGTHS = [5, 8, 10, 15, 20]

HORIZONS_SEC = [0.5, 1.0, 1.5]

# Velocity deadband sweep.
VELOCITY_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]

MIN_TRAJECTORY_LENGTH = 20


# ============================================================
# CORRIDOR
# ============================================================

def inside_corridor(x):

    return (
        CORRIDOR_X_MIN
        <= x
        <= CORRIDOR_X_MAX
    )


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

    t = np.arange(
        len(xs),
        dtype=np.float64
    ) / FPS

    velocity, _ = np.polyfit(
        t,
        xs,
        1
    )

    return float(velocity)


# ============================================================
# PREDICTION
# ============================================================

def predict_entry(
    current_x,
    velocity,
    horizon_sec,
):

    samples = max(
        2,
        int(horizon_sec * FPS)
    )

    for i in range(samples + 1):

        t = (
            i / samples
        ) * horizon_sec

        predicted_x = (
            current_x
            + velocity * t
        )

        if inside_corridor(
            predicted_x
        ):
            return True

    return False


# ============================================================
# LOAD FPL TRAJECTORIES
# ============================================================

def load_trajectories():

    files = sorted(
        DATASET_ROOT.glob(
            "*_trajectories_dynamic.json"
        )
    )

    print(
        "Loading FPL trajectories..."
    )

    print(
        f"Trajectory files: {len(files)}"
    )

    trajectories = []

    for path in files:

        try:

            with open(path, "r") as f:
                data = json.load(f)

        except Exception as e:

            print(
                f"Skipping {path.name}: {e}"
            )

            continue

        # ----------------------------------------------------
        # FPL structure:
        #
        # {
        #   "0": {
        #       "traj_sm": [[x,y], ...],
        #       ...
        #   },
        #   "1": {...}
        # }
        # ----------------------------------------------------

        if not isinstance(
            data,
            dict
        ):
            continue

        for trajectory_id, obj in data.items():

            if not isinstance(
                obj,
                dict
            ):
                continue

            points = obj.get(
                "traj_sm"
            )

            if not points:
                continue

            if len(points) < MIN_TRAJECTORY_LENGTH:
                continue

            xs = []

            for point in points:

                if (
                    isinstance(point, list)
                    and len(point) >= 1
                ):

                    try:

                        x_pixel = float(
                            point[0]
                        )

                        # Pixel -> normalized
                        x = (
                            x_pixel
                            / FRAME_WIDTH
                        )

                        xs.append(x)

                    except (
                        TypeError,
                        ValueError
                    ):
                        pass

            if len(xs) < MIN_TRAJECTORY_LENGTH:
                continue

            trajectories.append(
                np.asarray(
                    xs,
                    dtype=np.float64
                )
            )

    print(
        f"Usable trajectories: "
        f"{len(trajectories)}"
    )

    entering = sum(
        first_corridor_entry(xs)
        is not None
        for xs in trajectories
    )

    print(
        f"Trajectories entering corridor: "
        f"{entering}"
    )

    return trajectories


# ============================================================
# EVALUATE ONE TRAJECTORY
# ============================================================

def evaluate_trajectory(
    xs,
    history_length,
    horizon_sec,
    velocity_threshold,
):

    n = len(xs)

    entry_idx = (
        first_corridor_entry(xs)
    )

    # ========================================================
    # NEVER ENTERS
    # ========================================================

    if entry_idx is None:

        for current_idx in range(
            history_length - 1,
            n
        ):

            history = xs[
                current_idx
                - history_length
                + 1:
                current_idx + 1
            ]

            velocity = (
                estimate_velocity(
                    history
                )
            )

            # -------------------------------
            # VELOCITY DEADBAND
            # -------------------------------

            if abs(velocity) < (
                velocity_threshold
            ):

                velocity = 0.0

            predicted = predict_entry(
                xs[current_idx],
                velocity,
                horizon_sec
            )

            if predicted:

                return (
                    0,  # TP
                    1,  # FP
                    0,  # FN
                    0,  # TN
                    None
                )

        return (
            0,
            0,
            0,
            1,
            None
        )

    # ========================================================
    # ENTERS CORRIDOR
    # ========================================================

    for current_idx in range(
        history_length - 1,
        entry_idx
    ):

        history = xs[
            current_idx
            - history_length
            + 1:
            current_idx + 1
        ]

        velocity = (
            estimate_velocity(
                history
            )
        )

        # -------------------------------
        # VELOCITY DEADBAND
        # -------------------------------

        if abs(velocity) < (
            velocity_threshold
        ):

            velocity = 0.0

        predicted = predict_entry(
            xs[current_idx],
            velocity,
            horizon_sec
        )

        if predicted:

            warning_frames = (
                entry_idx
                - current_idx
            )

            warning_seconds = (
                warning_frames
                / FPS
            )

            return (
                1,
                0,
                0,
                0,
                warning_seconds
            )

    # --------------------------------------------------------
    # FALSE NEGATIVE
    # --------------------------------------------------------

    return (
        0,
        0,
        1,
        0,
        None
    )


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(
    trajectories
):

    print()
    print("=" * 115)
    print(
        "VELOCITY-THRESHOLD ONLINE "
        "TRAJECTORY-ENTRY BENCHMARK"
    )
    print("=" * 115)

    print(
        f"Corridor: "
        f"[{CORRIDOR_X_MIN:.2f}, "
        f"{CORRIDOR_X_MAX:.2f}]"
    )

    print(
        f"FPS: {FPS}"
    )

    print(
        f"Frame width: "
        f"{FRAME_WIDTH:.0f}px"
    )

    for threshold in (
        VELOCITY_THRESHOLDS
    ):

        print()
        print(
            "#" * 115
        )

        print(
            f"VELOCITY THRESHOLD = "
            f"{threshold:.2f} "
            f"normalized-x/s"
        )

        print(
            "#" * 115
        )

        print(
            f"{'HIST':>5} "
            f"{'HORIZON':>8} "
            f"{'TP':>7} "
            f"{'FP':>7} "
            f"{'FN':>7} "
            f"{'TN':>7} "
            f"{'PREC':>9} "
            f"{'RECALL':>9} "
            f"{'FPR':>9} "
            f"{'MED WARN':>11} "
            f"{'P95 WARN':>11}"
        )

        print(
            "-" * 115
        )

        for history_length in (
            HISTORY_LENGTHS
        ):

            for horizon in (
                HORIZONS_SEC
            ):

                TP = 0
                FP = 0
                FN = 0
                TN = 0

                warning_times = []

                for xs in trajectories:

                    (
                        tp,
                        fp,
                        fn,
                        tn,
                        warning
                    ) = evaluate_trajectory(
                        xs,
                        history_length,
                        horizon,
                        threshold
                    )

                    TP += tp
                    FP += fp
                    FN += fn
                    TN += tn

                    if (
                        warning
                        is not None
                    ):

                        warning_times.append(
                            warning
                        )

                precision = (
                    TP / (TP + FP)
                    if TP + FP > 0
                    else 0.0
                )

                recall = (
                    TP / (TP + FN)
                    if TP + FN > 0
                    else 0.0
                )

                fpr = (
                    FP / (FP + TN)
                    if FP + TN > 0
                    else 0.0
                )

                if warning_times:

                    median_warning = (
                        np.median(
                            warning_times
                        )
                    )

                    p95_warning = (
                        np.percentile(
                            warning_times,
                            95
                        )
                    )

                else:

                    median_warning = 0.0
                    p95_warning = 0.0

                print(
                    f"{history_length:5d} "
                    f"{horizon:8.2f} "
                    f"{TP:7d} "
                    f"{FP:7d} "
                    f"{FN:7d} "
                    f"{TN:7d} "
                    f"{precision * 100:8.2f}% "
                    f"{recall * 100:8.2f}% "
                    f"{fpr * 100:8.2f}% "
                    f"{median_warning:10.3f}s "
                    f"{p95_warning:10.3f}s"
                )


# ============================================================
# MAIN
# ============================================================

def main():

    trajectories = (
        load_trajectories()
    )

    if not trajectories:

        print(
            "ERROR: No usable "
            "trajectories found."
        )

        return

    run_benchmark(
        trajectories
    )


if __name__ == "__main__":
    main()
