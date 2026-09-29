import cv2
import numpy as np


class VanishingPointEstimator:
    def __init__(
        self,
        initial_x=0.50,
        initial_y=0.15,
        alpha=0.08,
        min_lines=8,
    ):
        self.vp_x = initial_x
        self.vp_y = initial_y
        self.alpha = alpha
        self.min_lines = min_lines

    def _line_intersection(self, l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        det = a1 * b2 - a2 * b1

        if abs(det) < 1e-6:
            return None

        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det

        return float(x), float(y)

    def _detect_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Work at 640 px width for speed.
        scale = min(1.0, 640.0 / w)

        if scale < 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        # Slight blur reduces tiny irrelevant edges.
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=55,
            minLineLength=50,
            maxLineGap=25,
        )

        if lines is None:
            return []

        result = []

        for item in lines[:, 0]:
            x1, y1, x2, y2 = map(float, item)

            dx = x2 - x1
            dy = y2 - y1

            length = np.hypot(dx, dy)

            if length < 50:
                continue

            angle = np.degrees(np.arctan2(dy, dx))

            # Normalize to [-90, 90].
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # We care about perspective lines, not horizontal edges.
            if abs(angle) < 15:
                continue

            # Ignore almost vertical lines.
            if abs(angle) > 82:
                continue

            if scale < 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
                length /= scale

            # Keep only lines whose lower endpoint is reasonably
            # low in the image. This suppresses many text/person edges.
            max_y = max(y1, y2)

            if max_y < 0.30 * h / scale:
                continue

            result.append(
                {
                    "line": (x1, y1, x2, y2),
                    "angle": angle,
                    "length": length,
                }
            )

        return result

    def _candidate_intersections(self, frame, lines):
        h, w = frame.shape[:2]

        candidates = []

        # Separate opposing slopes.
        left_lines = []
        right_lines = []

        for item in lines:
            angle = item["angle"]

            if angle < -15:
                left_lines.append(item)
            elif angle > 15:
                right_lines.append(item)

        if not left_lines or not right_lines:
            return []

        for left in left_lines:
            for right in right_lines:
                # Require meaningful directional difference.
                angle_diff = abs(left["angle"] - right["angle"])

                if angle_diff < 25:
                    continue

                if angle_diff > 150:
                    angle_diff = 180 - angle_diff

                if angle_diff < 25:
                    continue

                point = self._line_intersection(
                    left["line"],
                    right["line"],
                )

                if point is None:
                    continue

                x, y = point

                # Candidate VP should be around the forward/horizon
                # region. Allow some margin outside the image.
                if x < -0.10 * w or x > 1.10 * w:
                    continue

                if y < -0.05 * h or y > 0.55 * h:
                    continue

                # Normalize.
                xn = x / w
                yn = y / h

                # Forward VP should not be near the extreme sides.
                if not 0.20 <= xn <= 0.80:
                    continue

                # Weight long, strongly intersecting lines.
                angle_weight = np.sin(np.radians(angle_diff))

                weight = (
                    left["length"]
                    * right["length"]
                    * angle_weight
                )

                candidates.append(
                    (xn, yn, weight)
                )

        return candidates

    def _cluster_candidates(self, candidates):
        if len(candidates) < 3:
            return None

        pts = np.array(
            [[x, y] for x, y, _ in candidates],
            dtype=np.float64,
        )

        weights = np.array(
            [weight for _, _, weight in candidates],
            dtype=np.float64,
        )

        # Work with normalized coordinates.
        # A radius of ~0.07 means nearby VP hypotheses belong
        # to the same scene-geometry cluster.
        radius = 0.07

        best_score = -1.0
        best_center = None

        for i in range(len(pts)):
            distances = np.linalg.norm(
                pts - pts[i],
                axis=1,
            )

            mask = distances <= radius

            if np.count_nonzero(mask) < 3:
                continue

            score = np.sum(weights[mask])

            if score > best_score:
                best_score = score

                cluster_pts = pts[mask]
                cluster_weights = weights[mask]

                center = np.average(
                    cluster_pts,
                    axis=0,
                    weights=cluster_weights,
                )

                best_center = center

        if best_center is None:
            return None

        # Require enough supporting candidates.
        support = np.sum(
            np.linalg.norm(
                pts - best_center,
                axis=1,
            ) <= radius
        )

        if support < 3:
            return None

        return float(best_center[0]), float(best_center[1])

    def estimate(self, frame):
        lines = self._detect_lines(frame)

        if len(lines) < self.min_lines:
            return self.vp_x, self.vp_y, False

        candidates = self._candidate_intersections(
            frame,
            lines,
        )

        if len(candidates) < 3:
            return self.vp_x, self.vp_y, False

        candidate = self._cluster_candidates(candidates)

        if candidate is None:
            return self.vp_x, self.vp_y, False

        target_x, target_y = candidate

        # Safety bounds.
        target_x = float(np.clip(target_x, 0.20, 0.80))
        target_y = float(np.clip(target_y, 0.05, 0.45))

        # Temporal smoothing.
        self.vp_x = (
            (1.0 - self.alpha) * self.vp_x
            + self.alpha * target_x
        )

        self.vp_y = (
            (1.0 - self.alpha) * self.vp_y
            + self.alpha * target_y
        )

        return self.vp_x, self.vp_y, True

    def get(self):
        return self.vp_x, self.vp_y
