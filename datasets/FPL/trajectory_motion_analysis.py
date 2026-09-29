import json
from pathlib import Path
from collections import Counter

import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("data/trajectories")

FPS = 20.0

IMAGE_WIDTH = 1280.0
IMAGE_HEIGHT = 720.0

CORRIDOR_X_MIN = 0.35
CORRIDOR_X_MAX = 0.65
CORRIDOR_CENTER = 0.50

MIN_TRAJECTORY_LENGTH = 20

# Estimate motion over this many frames.
# 10 frames = 0.5 seconds at 20 FPS.
VELOCITY_HISTORY = 10

# Ignore extremely noisy instantaneous velocities.
# These are only used for classification/statistics.
MIN_REASONABLE_VELOCITY = -2.0
MAX_REASONABLE_VELOCITY = 2.0


# ============================================================
# HELPERS
# ============================================================

def normalize_trajectory(traj):

    xs = []
    ys = []

    for point in traj:

        if not isinstance(point, (list, tuple)):
            continue

        if len(point) < 2:
            continue

        x = float(point[0])
        y = float(point[1])

        xs.append(
            np.clip(
                x / IMAGE_WIDTH,
                0.0,
                1.0,
            )
        )

        ys.append(
            np.clip(
                y / IMAGE_HEIGHT,
                0.0,
                1.0,
            )
        )

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
    )


def estimate_velocity(values):

    if len(values) < 2:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    t = (
        np.arange(len(values))
        / FPS
    )

    velocity, _ = np.polyfit(
        t,
        values,
        1,
    )

    return float(velocity)


def estimate_motion(xs, ys):

    history = min(
        VELOCITY_HISTORY,
        len(xs),
    )

    hx = xs[-history:]
    hy = ys[-history:]

    vx = estimate_velocity(hx)
    vy = estimate_velocity(hy)

    return vx, vy


def corridor_entry_index(xs):

    for i, x in enumerate(xs):

        if (
            CORRIDOR_X_MIN
            <= x
            <= CORRIDOR_X_MAX
        ):
            return i

    return None


def corridor_exit_index(xs, entry_idx):

    if entry_idx is None:
        return None

    # Start after entry.
    for i in range(
        entry_idx + 1,
        len(xs),
    ):

        if not (
            CORRIDOR_X_MIN
            <= xs[i]
            <= CORRIDOR_X_MAX
        ):
            return i

    return None


def trajectory_direction(xs):

    if len(xs) < 2:
        return "UNKNOWN"

    dx = xs[-1] - xs[0]

    if abs(dx) < 0.02:
        return "VERTICAL"

    if dx > 0:
        return "LEFT_TO_RIGHT"

    return "RIGHT_TO_LEFT"


def motion_direction(vx):

    if vx > 0.03:
        return "RIGHT"

    if vx < -0.03:
        return "LEFT"

    return "LATERAL_STATIONARY"


def time_to_boundary(x, vx):
    """
    Time until the trajectory reaches the nearest corridor
    boundary in the direction it is actually moving.

    Returns None when the object is moving away from the corridor.
    Returns 0 when already inside the corridor.
    """

    # Already inside.
    if CORRIDOR_X_MIN <= x <= CORRIDOR_X_MAX:
        return 0.0

    # Left of corridor.
    if x < CORRIDOR_X_MIN:

        # Must move right to reach corridor.
        if vx <= 0.0:
            return None

        return (CORRIDOR_X_MIN - x) / vx

    # Right of corridor.
    if x > CORRIDOR_X_MAX:

        # Must move left to reach corridor.
        if vx >= 0.0:
            return None

        return (x - CORRIDOR_X_MAX) / abs(vx)

    return None

def classify_corridor_relation(
    x,
    vx,
):

    if (
        CORRIDOR_X_MIN
        <= x
        <= CORRIDOR_X_MAX
    ):

        if vx > 0.03:
            return "INSIDE_MOVING_RIGHT"

        if vx < -0.03:
            return "INSIDE_MOVING_LEFT"

        return "INSIDE_STATIONARY"

    if x < CORRIDOR_X_MIN:

        if vx > 0.03:
            return "APPROACHING_FROM_LEFT"

        return "OUTSIDE_LEFT"

    if x > CORRIDOR_X_MAX:

        if vx < -0.03:
            return "APPROACHING_FROM_RIGHT"

        return "OUTSIDE_RIGHT"

    return "UNKNOWN"


# ============================================================
# LOAD
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

            xs, ys = normalize_trajectory(
                traj
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
# ANALYSIS
# ============================================================

def analyze(trajectories):

    motion_records = []

    entry_times = []

    crossing_durations = []

    vx_values = []
    vy_values = []

    direction_counter = Counter()
    relation_counter = Counter()

    approaching_left = []
    approaching_right = []

    never_entered = 0
    entered = 0
    exited = 0

    for traj in trajectories:

        xs = traj["xs"]
        ys = traj["ys"]

        vx, vy = estimate_motion(
            xs,
            ys,
        )

        # ----------------------------------------------------
        # Basic motion
        # ----------------------------------------------------

        vx_values.append(vx)
        vy_values.append(vy)

        direction = trajectory_direction(
            xs
        )

        direction_counter[
            direction
        ] += 1

        # ----------------------------------------------------
        # Corridor
        # ----------------------------------------------------

        entry_idx = corridor_entry_index(
            xs
        )

        if entry_idx is None:

            never_entered += 1

        else:

            entered += 1

            entry_time = (
                entry_idx / FPS
            )

            entry_times.append(
                entry_time
            )

            exit_idx = corridor_exit_index(
                xs,
                entry_idx,
            )

            if exit_idx is not None:

                exited += 1

                crossing_durations.append(
                    (
                        exit_idx
                        - entry_idx
                    ) / FPS
                )

        # ----------------------------------------------------
        # Motion relative to corridor
        # ----------------------------------------------------

        current_x = xs[-1]

        relation = (
            classify_corridor_relation(
                current_x,
                vx,
            )
        )

        relation_counter[
            relation
        ] += 1

        t_boundary = (
            time_to_boundary(
                current_x,
                vx,
            )
        )

        if relation == "APPROACHING_FROM_LEFT":

            approaching_left.append(
                t_boundary
            )

        elif relation == "APPROACHING_FROM_RIGHT":

            approaching_right.append(
                t_boundary
            )

        motion_records.append({
            "file": traj["file"],
            "track_id": traj["track_id"],
            "x": current_x,
            "y": ys[-1],
            "vx": vx,
            "vy": vy,
            "direction": direction,
            "relation": relation,
            "time_to_boundary": t_boundary,
            "entry_idx": entry_idx,
        })

    return {
        "motion_records": motion_records,
        "vx_values": vx_values,
        "vy_values": vy_values,
        "entry_times": entry_times,
        "crossing_durations": crossing_durations,
        "direction_counter": direction_counter,
        "relation_counter": relation_counter,
        "approaching_left": approaching_left,
        "approaching_right": approaching_right,
        "entered": entered,
        "never_entered": never_entered,
        "exited": exited,
    }


# ============================================================
# STATISTICS
# ============================================================

def stats(values):

    values = [
        x for x in values
        if x is not None
        and np.isfinite(x)
        and
        MIN_REASONABLE_VELOCITY
        <= x
        <= MAX_REASONABLE_VELOCITY
    ]

    if not values:
        return None

    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": len(arr),
        "mean": float(
            np.mean(arr)
        ),
        "median": float(
            np.median(arr)
        ),
        "p05": float(
            np.percentile(arr, 5)
        ),
        "p25": float(
            np.percentile(arr, 25)
        ),
        "p75": float(
            np.percentile(arr, 75)
        ),
        "p95": float(
            np.percentile(arr, 95)
        ),
        "min": float(
            np.min(arr)
        ),
        "max": float(
            np.max(arr)
        ),
    }


def print_stats(
    name,
    values,
    unit="",
):

    s = stats(values)

    print()
    print(name)

    if s is None:

        print("  No valid samples")
        return

    print(
        f"  count   : {s['count']}"
    )

    print(
        f"  mean    : {s['mean']:.4f}{unit}"
    )

    print(
        f"  median  : {s['median']:.4f}{unit}"
    )

    print(
        f"  p05     : {s['p05']:.4f}{unit}"
    )

    print(
        f"  p25     : {s['p25']:.4f}{unit}"
    )

    print(
        f"  p75     : {s['p75']:.4f}{unit}"
    )

    print(
        f"  p95     : {s['p95']:.4f}{unit}"
    )

    print(
        f"  min     : {s['min']:.4f}{unit}"
    )

    print(
        f"  max     : {s['max']:.4f}{unit}"
    )


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    trajectories,
    result,
):

    total = len(trajectories)

    print()
    print("=" * 90)
    print(
        "FPL TRAJECTORY MOTION ANALYSIS"
    )
    print("=" * 90)

    print(
        f"Usable trajectories : {total}"
    )

    print(
        f"Entered corridor   : "
        f"{result['entered']} "
        f"({100 * result['entered'] / total:.2f}%)"
    )

    print(
        f"Never entered      : "
        f"{result['never_entered']} "
        f"({100 * result['never_entered'] / total:.2f}%)"
    )

    print(
        f"Entered + exited   : "
        f"{result['exited']}"
    )

    print()
    print("-" * 90)
    print("TRAJECTORY DIRECTION")
    print("-" * 90)

    for key, value in (
        result[
            "direction_counter"
        ]
        .most_common()
    ):

        print(
            f"{key:25s} "
            f"{value:6d} "
            f"({100 * value / total:6.2f}%)"
        )

    print()
    print("-" * 90)
    print("CURRENT CORRIDOR RELATION")
    print("-" * 90)

    for key, value in (
        result[
            "relation_counter"
        ]
        .most_common()
    ):

        print(
            f"{key:30s} "
            f"{value:6d} "
            f"({100 * value / total:6.2f}%)"
        )

    print_stats(
        "X VELOCITY (normalized image units / second)",
        result["vx_values"],
    )

    print_stats(
        "Y VELOCITY (normalized image units / second)",
        result["vy_values"],
    )

    print_stats(
        "CORRIDOR ENTRY TIME FROM TRAJECTORY START",
        result["entry_times"],
        " s",
    )

    print_stats(
        "TIME SPENT INSIDE CORRIDOR",
        result["crossing_durations"],
        " s",
    )

    print_stats(
        "TIME TO CORRIDOR FROM LEFT",
        result["approaching_left"],
        " s",
    )

    print_stats(
        "TIME TO CORRIDOR FROM RIGHT",
        result["approaching_right"],
        " s",
    )

    print()
    print("=" * 90)
    print("FASTEST APPROACHING TRAJECTORIES")
    print("=" * 90)

    candidates = [
        r
        for r in result["motion_records"]
        if r["time_to_boundary"] is not None
        and r["time_to_boundary"] >= 0
    ]

    candidates.sort(
        key=lambda r:
        r["time_to_boundary"]
    )

    print(
        f"{'FILE':18s} "
        f"{'ID':>6s} "
        f"{'X':>7s} "
        f"{'VX':>8s} "
        f"{'T_ENTRY':>9s} "
        f"{'RELATION':>25s}"
    )

    print("-" * 90)

    for r in candidates[:25]:

        print(
            f"{r['file'][:18]:18s} "
            f"{r['track_id']:>6s} "
            f"{r['x']:7.3f} "
            f"{r['vx']:8.3f} "
            f"{r['time_to_boundary']:8.3f}s "
            f"{r['relation']:>25s}"
        )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(records):

    output = Path(
        "trajectory_motion_records.csv"
    )

    with output.open("w") as f:

        f.write(
            "file,track_id,x,y,vx,vy,"
            "direction,relation,"
            "time_to_boundary,entry_idx\n"
        )

        for r in records:

            t = r[
                "time_to_boundary"
            ]

            f.write(
                f"{r['file']},"
                f"{r['track_id']},"
                f"{r['x']:.6f},"
                f"{r['y']:.6f},"
                f"{r['vx']:.6f},"
                f"{r['vy']:.6f},"
                f"{r['direction']},"
                f"{r['relation']},"
                f"{'' if t is None else f'{t:.6f}'},"
                f"{'' if r['entry_idx'] is None else r['entry_idx']}\n"
            )

    print()
    print(
        f"Saved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    trajectories = (
        load_trajectories()
    )

    print(
        f"Usable trajectories: "
        f"{len(trajectories)}"
    )

    result = analyze(
        trajectories
    )

    print_report(
        trajectories,
        result,
    )

    save_csv(
        result["motion_records"]
    )


if __name__ == "__main__":
    main()
