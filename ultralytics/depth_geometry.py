import cv2
import numpy as np


class ForwardRegionEstimator:
    """
    Lightweight scene-relative forward-region estimator.

    Uses monocular relative depth to estimate a central region
    through which forward motion appears plausible.

    This is deliberately conservative and does not assume metric depth,
    camera intrinsics, or a fixed vanishing point.
    """

    def __init__(
        self,
        alpha=0.12,
        min_valid_fraction=0.45,
    ):
        self.alpha = alpha
        self.min_valid_fraction = min_valid_fraction

        self.left = 0.30
        self.right = 0.70

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

    def _object_mask(self, shape, boxes):
        h, w = shape

        mask = np.ones((h, w), dtype=np.uint8)

        if boxes is None:
            return mask

        for box in boxes:
            x1, y1, x2, y2 = box

            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h))

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0

        return mask

    def _profile(self, depth, mask, y0, y1):
        h, w = depth.shape

        profile = np.full(w, np.nan, dtype=np.float32)

        for x in range(w):
            values = depth[y0:y1, x]
            valid = mask[y0:y1, x] > 0

            values = values[valid]
            values = values[np.isfinite(values)]

            if len(values) >= 8:
                profile[x] = np.median(values)

        return profile

    def _smooth(self, profile):
        valid = np.isfinite(profile)

        if np.count_nonzero(valid) < 10:
            return None

        x = np.arange(len(profile))

        filled = np.interp(
            x,
            x[valid],
            profile[valid],
        )

        return cv2.GaussianBlur(
            filled.reshape(1, -1),
            (0, 0),
            sigmaX=18,
        ).ravel()

    def estimate(self, depth, boxes=None):
        d = self._normalize_depth(depth)

        if d is None:
            self.valid = False
            return self.left, self.right, False

        h, w = d.shape

        mask = self._object_mask(d.shape, boxes)

        # Use the lower portion of the image where forward ground
        # structure is most likely to appear.
        y0 = int(0.40 * h)
        y1 = int(0.92 * h)

        profile = self._profile(
            d,
            mask,
            y0,
            y1,
        )

        valid_fraction = np.mean(np.isfinite(profile))

        if valid_fraction < self.min_valid_fraction:
            self.valid = False
            return self.left, self.right, False

        profile = self._smooth(profile)

        if profile is None:
            self.valid = False
            return self.left, self.right, False

        # Look at the central region first.
        center = w * 0.50

        # Search for the broadest region whose depth behaviour is
        # reasonably similar to its neighbours.
        local_gradient = np.abs(
            np.gradient(profile)
        )

        # Suppress very sharp transitions caused by isolated objects
        # or strong depth discontinuities.
        threshold = np.percentile(
            local_gradient,
            70,
        )

        stable = local_gradient <= threshold

        # Only consider the useful central image area.
        search_left = int(0.10 * w)
        search_right = int(0.90 * w)

        stable[:search_left] = False
        stable[search_right:] = False

        # Find contiguous stable runs.
        runs = []

        start = None

        for i, value in enumerate(stable):
            if value and start is None:
                start = i

            elif not value and start is not None:
                if i - start >= int(0.08 * w):
                    runs.append((start, i - 1))
                start = None

        if start is not None:
            if w - start >= int(0.08 * w):
                runs.append((start, w - 1))

        if not runs:
            self.valid = False
            return self.left, self.right, False

        # Prefer runs close to the optical center.
        def run_score(run):
            x1, x2 = run

            run_center = (x1 + x2) / 2.0
            width = x2 - x1

            center_distance = abs(run_center - center)

            return width - 1.5 * center_distance

        best = max(runs, key=run_score)

        x1, x2 = best

        target_left = x1 / w
        target_right = x2 / w

        # Don't allow absurdly narrow or wide regions.
        target_width = target_right - target_left

        if target_width < 0.20:
            mid = (target_left + target_right) / 2.0
            target_left = mid - 0.10
            target_right = mid + 0.10

        if target_width > 0.80:
            mid = (target_left + target_right) / 2.0
            target_left = mid - 0.40
            target_right = mid + 0.40

        target_left = float(np.clip(target_left, 0.10, 0.45))
        target_right = float(np.clip(target_right, 0.55, 0.90))

        # Temporal smoothing.
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

    def get(self):
        return self.left, self.right, self.valid
