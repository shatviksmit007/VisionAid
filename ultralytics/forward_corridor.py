import cv2
import numpy as np


class ForwardCorridorEstimator:
    """
    Camera-centered forward corridor.

    The optical axis is treated as the user's approximate forward
    direction. Depth and detected objects are used to identify
    occupied regions. The result is the widest reasonably clear
    region around the image center.

    Coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        alpha=0.15,
        initial_left=0.30,
        initial_right=0.70,
    ):
        self.alpha = alpha

        self.left = initial_left
        self.right = initial_right

        self.valid = False

    def _normalize_depth(self, depth):
        depth = np.asarray(depth, dtype=np.float32)

        valid = np.isfinite(depth)

        if not np.any(valid):
            return None

        values = depth[valid]

        lo = np.percentile(values, 5)
        hi = np.percentile(values, 95)

        if hi - lo < 1e-6:
            return None

        depth = (depth - lo) / (hi - lo)

        return np.clip(depth, 0.0, 1.0)

    def _occupancy_from_boxes(self, shape, boxes):
        h, w = shape

        occupancy = np.zeros((h, w), dtype=np.float32)

        if boxes is None:
            return occupancy

        for box in boxes:
            x1, y1, x2, y2 = box

            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h))

            if x2 > x1 and y2 > y1:
                occupancy[y1:y2, x1:x2] = 1.0

        return occupancy

    def _column_depth(self, depth, y0, y1):
        """
        Median depth in the lower image region for every column.
        """
        region = depth[y0:y1]

        valid = np.isfinite(region)

        result = np.full(
            depth.shape[1],
            np.nan,
            dtype=np.float32,
        )

        for x in range(depth.shape[1]):
            values = region[:, x][valid[:, x]]

            if len(values) >= 5:
                result[x] = np.median(values)

        return result

    def estimate(self, depth, boxes=None):
        d = self._normalize_depth(depth)

        if d is None:
            self.valid = False
            return self.left, self.right, False

        h, w = d.shape

        # Lower region corresponds most closely to the space immediately
        # in front of the user.
        y0 = int(0.50 * h)
        y1 = int(0.95 * h)

        column_depth = self._column_depth(
            d,
            y0,
            y1,
        )

        valid = np.isfinite(column_depth)

        if np.count_nonzero(valid) < 0.50 * w:
            self.valid = False
            return self.left, self.right, False

        # Fill missing columns and smooth.
        x = np.arange(w)

        filled = np.interp(
            x,
            x[valid],
            column_depth[valid],
        )

        smooth = cv2.GaussianBlur(
            filled.reshape(1, -1),
            (0, 0),
            sigmaX=20,
        ).ravel()

        # Object occupancy.
        occupancy = self._occupancy_from_boxes(
            d.shape,
            boxes,
        )

        # Look at the same lower region.
        occupied_fraction = np.mean(
            occupancy[y0:y1],
            axis=0,
        )

        # A column is considered occupied if a meaningful portion
        # of the near-field region contains a detected object.
        blocked = occupied_fraction > 0.08

        # Very close depth values indicate nearby surfaces/objects.
        # Use a robust percentile instead of assuming metric depth.
        near_threshold = np.percentile(
            smooth,
            30,
        )

        depth_blocked = smooth <= near_threshold

        blocked |= depth_blocked

        # Find contiguous free runs.
        free = ~blocked

        runs = []
        start = None

        for i, is_free in enumerate(free):
            if is_free and start is None:
                start = i

            elif not is_free and start is not None:
                if i - start >= int(0.08 * w):
                    runs.append((start, i - 1))
                start = None

        if start is not None:
            if w - start >= int(0.08 * w):
                runs.append((start, w - 1))

        if not runs:
            self.valid = False
            return self.left, self.right, False

        center = w / 2.0

        # We want the widest free region that is closest to the
        # optical axis.
        def score(run):
            x1, x2 = run
            width = x2 - x1

            run_center = (x1 + x2) / 2.0
            center_distance = abs(run_center - center)

            return width - 2.0 * center_distance

        best = max(runs, key=score)

        x1, x2 = best

        target_left = x1 / w
        target_right = x2 / w

        # Keep a reasonable minimum corridor.
        if target_right - target_left < 0.20:
            mid = (target_left + target_right) / 2.0
            target_left = mid - 0.10
            target_right = mid + 0.10

        target_left = float(
            np.clip(target_left, 0.10, 0.45)
        )

        target_right = float(
            np.clip(target_right, 0.55, 0.90)
        )

        # Smooth the corridor.
        self.left = (
            (1.0 - self.alpha) * self.left
            + self.alpha * target_left
        )

        self.right = (
            (1.0 - self.alpha) * self.right
            + self.alpha * target_right
        )

        self.valid = True

        return self.left, self.right, True

    def contains(self, x_norm):
        return (
            self.left <= x_norm <= self.right
        )

    def get(self):
        return self.left, self.right, self.valid
