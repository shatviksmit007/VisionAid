import json
import math
from pathlib import Path

ROOT = Path("data/trajectories")

IMAGE_W = 1280.0
IMAGE_H = 720.0

# Camera-centered collision corridor.
CORRIDOR_LEFT = 0.35
CORRIDOR_RIGHT = 0.65

# Ignore tiny trajectories.
MIN_DISPLACEMENT = 20.0

# A trajectory must spend at least this fraction of its
# valid points inside the corridor to count as meaningful.
MIN_CORRIDOR_FRACTION = 0.10


def x_norm(x):
    return x / IMAGE_W


def inside_corridor(x):
    x = x_norm(x)
    return CORRIDOR_LEFT <= x <= CORRIDOR_RIGHT


def analyze_trajectory(points):
    if len(points) < 2:
        return None

    start = points[0]
    end = points[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    displacement = math.hypot(dx, dy)

    if displacement < MIN_DISPLACEMENT:
        return None

    inside = [inside_corridor(p[0]) for p in points]

    inside_count = sum(inside)
    corridor_fraction = inside_count / len(points)

    # Find transitions.
    entered = []
    exited = []

    for i in range(1, len(inside)):
        if not inside[i - 1] and inside[i]:
            entered.append(i)

        if inside[i - 1] and not inside[i]:
            exited.append(i)

    entry_index = entered[0] if entered else None
    exit_index = exited[0] if exited else None

    # Did the person actually cross the corridor?
    crossed = len(entered) > 0 and len(exited) > 0

    # Entered but hasn't left by end of track.
    entered_only = len(entered) > 0 and len(exited) == 0

    # Started inside and left.
    exited_only = len(entered) == 0 and len(exited) > 0

    if crossed:
        classification = "CROSSED_CORRIDOR"
    elif entered_only:
        classification = "ENTERED_CORRIDOR"
    elif exited_only:
        classification = "EXITED_CORRIDOR"
    elif corridor_fraction >= MIN_CORRIDOR_FRACTION:
        classification = "STAYED_IN_CORRIDOR"
    else:
        classification = "OUTSIDE_CORRIDOR"

    # Direction of crossing.
    crossing_direction = "NONE"

    if crossed:
        first_enter = entered[0]

        # Look at the trajectory immediately before/after entry.
        if first_enter > 0 and first_enter < len(points) - 1:
            x_before = points[first_enter - 1][0]
            x_after = points[first_enter + 1][0]

            if x_after > x_before:
                crossing_direction = "LEFT_TO_RIGHT"
            elif x_after < x_before:
                crossing_direction = "RIGHT_TO_LEFT"

    return {
        "num_points": len(points),
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
        "dx": dx,
        "dy": dy,
        "displacement": displacement,
        "corridor_fraction": corridor_fraction,
        "entered": len(entered) > 0,
        "exited": len(exited) > 0,
        "crossed": crossed,
        "classification": classification,
        "crossing_direction": crossing_direction,

        "entry_index": entry_index,
        "exit_index": exit_index,

        # FPL processed data is 20 FPS.
        "entry_time_sec": (
            entry_index / 20.0
            if entry_index is not None
            else None
        ),

        "time_before_entry_sec": (
            (len(points) - entry_index) / 20.0
            if entry_index is not None
            else None
        ),

        "x_velocity_px_per_frame": (
            dx / max(len(points) - 1, 1)
        ),

        "y_velocity_px_per_frame": (
            dy / max(len(points) - 1, 1)
        ),
    }


def analyze_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    results = []

    for track_id, track in data.items():

        traj = track.get("traj_sm")

        if not traj or len(traj) < 2:
            continue

        points = []

        for p in traj:
            if (
                isinstance(p, list)
                and len(p) >= 2
                and isinstance(p[0], (int, float))
                and isinstance(p[1], (int, float))
            ):
                points.append(
                    (float(p[0]), float(p[1]))
                )

        result = analyze_trajectory(points)

        if result is None:
            continue

        result["file"] = path.name
        result["track_id"] = track_id

        results.append(result)

    return results


def main():

    files = sorted(
        ROOT.glob("*_trajectories_dynamic.json")
    )

    print(f"Trajectory files: {len(files)}")

    all_results = []

    for i, path in enumerate(files, 1):

        try:
            results = analyze_file(path)
            all_results.extend(results)

        except Exception as e:
            print(f"[WARN] {path.name}: {e}")

        if i % 25 == 0:
            print(
                f"Processed {i}/{len(files)}"
            )

    print()
    print("=" * 80)
    print("FPL TRAJECTORY GEOMETRY")
    print("=" * 80)

    print(
        f"Tracks analyzed: {len(all_results)}"
    )

    counts = {}

    for r in all_results:
        c = r["classification"]
        counts[c] = counts.get(c, 0) + 1

    print()

    for c, n in sorted(
        counts.items(),
        key=lambda x: -x[1]
    ):
        pct = 100 * n / max(len(all_results), 1)

        print(
            f"{c:20s} "
            f"{n:7d} "
            f"({pct:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Actual corridor crossing cases
    # ---------------------------------------------------------

    crossing = [
        r for r in all_results
        if r["classification"] == "CROSSED_CORRIDOR"
    ]

    print()
    print(
        f"Actual corridor crossings: "
        f"{len(crossing)}"
    )

    print()
    print("=" * 120)
    print("CORRIDOR CROSSING CASES")
    print("=" * 120)

    crossing.sort(
        key=lambda r: (
            r["corridor_fraction"],
            r["displacement"]
        ),
        reverse=True
    )

    for r in crossing[:50]:

        print(
            f'{r["file"]:42s} '
            f'ID={r["track_id"]:>5s} '
            f'pts={r["num_points"]:>4d} '
            f'dx={r["dx"]:>8.1f} '
            f'dy={r["dy"]:>8.1f} '
            f'corr={r["corridor_fraction"]:.2f} '
            f'{r["crossing_direction"]}'
        )

    # ---------------------------------------------------------
    # Strong cases:
    # person spends meaningful time inside corridor
    # ---------------------------------------------------------

    strong = [
        r for r in crossing
        if r["corridor_fraction"] >= 0.25
    ]

    print()
    print(
        f"Strong crossings "
        f"(>=25% of trajectory in corridor): "
        f"{len(strong)}"
    )

    # ---------------------------------------------------------
    # Earliest corridor-entry cases
    # ---------------------------------------------------------

    entering = [
        r for r in all_results
        if r["entry_index"] is not None
    ]

    entering.sort(
        key=lambda r: r["entry_index"]
    )

    print()
    print("=" * 120)
    print("EARLIEST CORRIDOR ENTRIES")
    print("=" * 120)

    for r in entering[:50]:

        print(
            f'{r["file"]:42s} '
            f'ID={r["track_id"]:>5s} '
            f'pts={r["num_points"]:>4d} '
            f'entry={r["entry_index"]:>4d} '
            f'entry_t={r["entry_time_sec"]:>6.2f}s '
            f'xv={r["x_velocity_px_per_frame"]:>7.2f} '
            f'yv={r["y_velocity_px_per_frame"]:>7.2f} '
            f'{r["crossing_direction"]}'
        )

    # ---------------------------------------------------------
    # Fast lateral entries
    # ---------------------------------------------------------

    fast_entries = [
        r for r in entering
        if abs(r["x_velocity_px_per_frame"]) > 2.0
    ]

    fast_entries.sort(
        key=lambda r: abs(
            r["x_velocity_px_per_frame"]
        ),
        reverse=True
    )

    print()
    print("=" * 120)
    print("FAST LATERAL CORRIDOR ENTRIES")
    print("=" * 120)

    for r in fast_entries[:50]:

        print(
            f'{r["file"]:42s} '
            f'ID={r["track_id"]:>5s} '
            f'pts={r["num_points"]:>4d} '
            f'xv={r["x_velocity_px_per_frame"]:>7.2f} '
            f'yv={r["y_velocity_px_per_frame"]:>7.2f} '
            f'entry={r["entry_index"]:>4d} '
            f'{r["crossing_direction"]}'
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    out = Path(
        "fpl_trajectory_geometry.json"
    )

    with open(out, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2
        )

    print()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
