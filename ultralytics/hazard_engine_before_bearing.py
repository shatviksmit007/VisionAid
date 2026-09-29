from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import math
import time
import numpy as np

# Converging perspective corridor
VP_X = 0.50
VP_Y = 0.15
BOTTOM_LEFT_X = 0.28
BOTTOM_RIGHT_X = 0.72
BOTTOM_Y = 1.00



# ============================================================
# PATH GEOMETRY
# ============================================================

# Normalized image coordinates.
#
# The corridor converges toward this region near the horizon.
# These are intentionally conservative starting values and
# should be tuned against the actual POV footage.

CORRIDOR_TOP_Y = 0.18
CORRIDOR_BOTTOM_Y = 1.00

# Half-width of the walking corridor at the top and bottom.
#
# Keep the top relatively narrow so distant/off-path objects
# don't get included just because they are high in the frame.
CORRIDOR_TOP_HALF_WIDTH = 0.055
CORRIDOR_BOTTOM_HALF_WIDTH = 0.22


# ============================================================
# PREDICTION / REACTION
# ============================================================

# How far ahead we predict the object's horizontal footpoint.
#
# This is NOT physical time-to-collision.
# It is simply an anticipation horizon for corridor entry.
PREDICTION_HORIZON = 0.35

# Additional normalized horizontal margin.
#
# Allows an object moving toward the corridor to be considered
# before its current footpoint actually crosses the boundary.
PREDICTION_MARGIN = 0.035

# Minimum number of position samples required before
# predicting horizontal motion.
MIN_POSITION_HISTORY = 4

# Don't trust absurdly large image-space velocities.
MAX_X_VELOCITY = 1.5


# ============================================================
# TTC
# ============================================================

LOW_TTC = 3.0
MEDIUM_TTC = 1.0

MIN_HISTORY = 5
HISTORY_SECONDS = 0.6

# Minimum positive relative-depth velocity required before
# declaring that an object is approaching.
MIN_APPROACHING_VELOCITY = 0.03

MAX_REASONABLE_TTC = 20.0


# ============================================================
# TRACKING
# ============================================================

MIN_IOU = 0.25
TRACK_TIMEOUT = 1.0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox: tuple
    confidence: float
    relative_depth: float


@dataclass
class ObjectState:
    track_id: int
    class_id: int
    class_name: str

    bbox: tuple
    confidence: float

    # (timestamp, relative_depth)
    depth_history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )

    # (timestamp, normalized_foot_x, normalized_foot_y)
    position_history: deque = field(
        default_factory=lambda: deque(maxlen=30)
    )

    velocity: Optional[float] = None
    ttc: Optional[float] = None

    path_overlap: float = 0.0
    predicted_path_overlap: float = 0.0

    current_x: float = 0.0
    predicted_x: float = 0.0

    x_velocity: float = 0.0

    risk: str = "NONE"

    last_seen: float = 0.0


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def footpoint(bbox):
    """
    Bottom-center of bounding box.

    For pedestrians this is a better approximation of where
    the object intersects the ground plane than bbox center.
    """
    x1, y1, x2, y2 = bbox

    x = (x1 + x2) * 0.5
    y = y2

    return x, y


def corridor_bounds(y_norm):
    """Return left/right x boundaries of the converging walking corridor."""
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    # Above the vanishing point there is no usable walking corridor.
    if y_norm <= VP_Y:
        return VP_X, VP_X

    # Interpolate along the two perspective boundary lines:
    #   VP -> bottom-left
    #   VP -> bottom-right
    t = (y_norm - VP_Y) / (BOTTOM_Y - VP_Y)

    left = VP_X + t * (BOTTOM_LEFT_X - VP_X)
    right = VP_X + t * (BOTTOM_RIGHT_X - VP_X)

    return float(left), float(right)


def corridor_score(x_norm, y_norm):
    """
    Score how strongly a footpoint lies inside the walking corridor.

    1.0 = corridor center
    0.0 = corridor boundary or outside
    """
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    y_norm = float(np.clip(y_norm, 0.0, 1.0))

    if y_norm <= VP_Y:
        return 0.0

    left, right = corridor_bounds(y_norm)

    if x_norm < left or x_norm > right:
        return 0.0

    width = right - left
    if width <= 1e-9:
        return 0.0

    center = (left + right) / 2.0
    distance_from_center = abs(x_norm - center)

    score = 1.0 - (distance_from_center / (width / 2.0))

    return float(np.clip(score, 0.0, 1.0))


def predicted_corridor_score(
    current_x,
    y_norm,
    x_velocity,
    horizon=PREDICTION_HORIZON
):
    """
    Predict horizontal footpoint location after a short horizon.

    This is used only for anticipating corridor entry.
    """

    predicted_x = current_x + x_velocity * horizon

    predicted_x = np.clip(predicted_x, 0.0, 1.0)

    score = corridor_score(
        predicted_x,
        y_norm
    )

    return predicted_x, score


# ============================================================
# HAZARD ENGINE
# ============================================================

class HazardEngine:

    def __init__(self):
        self.tracks = {}
        self.next_track_id = 0

    # --------------------------------------------------------
    # TRACK ASSOCIATION
    # --------------------------------------------------------

    def associate(self, detections):

        if not self.tracks or not detections:
            return {}, set(range(len(detections))), set(self.tracks)

        candidates = []

        for det_idx, det in enumerate(detections):

            for track_id, track in self.tracks.items():

                iou = bbox_iou(
                    det.bbox,
                    track.bbox
                )

                if iou >= MIN_IOU:
                    candidates.append(
                        (
                            iou,
                            det_idx,
                            track_id
                        )
                    )

        # Highest IoU first.
        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        assignments = {}

        used_detections = set()
        used_tracks = set()

        for iou, det_idx, track_id in candidates:

            if det_idx in used_detections:
                continue

            if track_id in used_tracks:
                continue

            assignments[det_idx] = track_id

            used_detections.add(det_idx)
            used_tracks.add(track_id)

        unmatched_detections = (
            set(range(len(detections)))
            - used_detections
        )

        unmatched_tracks = (
            set(self.tracks.keys())
            - used_tracks
        )

        return (
            assignments,
            unmatched_detections,
            unmatched_tracks
        )

    # --------------------------------------------------------
    # DEPTH VELOCITY
    # --------------------------------------------------------

    def estimate_velocity(self, history):

        if len(history) < MIN_HISTORY:
            return None

        now = history[-1][0]

        recent = [
            (t, d)
            for t, d in history
            if now - t <= HISTORY_SECONDS
        ]

        if len(recent) < MIN_HISTORY:
            return None

        t = np.array(
            [x[0] for x in recent],
            dtype=np.float64
        )

        d = np.array(
            [x[1] for x in recent],
            dtype=np.float64
        )

        # Normalize time around zero.
        t = t - t[-1]

        if np.ptp(t) <= 1e-6:
            return None

        slope = np.polyfit(t, d, 1)[0]

        return float(slope)

    # --------------------------------------------------------
    # HORIZONTAL MOTION
    # --------------------------------------------------------

    def estimate_x_velocity(self, history):

        if len(history) < MIN_POSITION_HISTORY:
            return 0.0

        now = history[-1][0]

        recent = [
            (t, x)
            for t, x, y in history
            if now - t <= HISTORY_SECONDS
        ]

        if len(recent) < MIN_POSITION_HISTORY:
            return 0.0

        t = np.array(
            [p[0] for p in recent],
            dtype=np.float64
        )

        x = np.array(
            [p[1] for p in recent],
            dtype=np.float64
        )

        t = t - t[-1]

        if np.ptp(t) <= 1e-6:
            return 0.0

        velocity = np.polyfit(
            t,
            x,
            1
        )[0]

        velocity = float(
            np.clip(
                velocity,
                -MAX_X_VELOCITY,
                MAX_X_VELOCITY
            )
        )

        return velocity

    # --------------------------------------------------------
    # TTC
    # --------------------------------------------------------

    def calculate_ttc(self, velocity, depth):

        if velocity is None:
            return None

        # For the current Depth Anything representation:
        #
        # increasing depth value = object becoming closer.
        #
        # Therefore positive velocity means approaching.

        if velocity <= MIN_APPROACHING_VELOCITY:
            return None

        if depth <= 0:
            return None

        ttc = depth / velocity

        if not math.isfinite(ttc):
            return None

        if ttc <= 0:
            return None

        if ttc > MAX_REASONABLE_TTC:
            return None

        return float(ttc)

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    def calculate_risk(
        self,
        path_overlap,
        predicted_path_overlap,
        velocity,
        ttc
    ):

        # Completely irrelevant object.
        if (
            path_overlap <= 0.0
            and predicted_path_overlap <= 0.0
        ):
            return "NONE"

        # Object isn't approaching.
        if (
            velocity is None
            or velocity <= MIN_APPROACHING_VELOCITY
        ):
            return "NONE"

        # If the object is currently outside but predicted
        # to enter the path, treat it conservatively as LOW.
        entering_path = (
            path_overlap <= 0.0
            and predicted_path_overlap > 0.0
        )

        if ttc is None:
            return "LOW"

        # Object is approaching but still has substantial time.
        if ttc > LOW_TTC:
            return "LOW"

        # Medium danger.
        if ttc > MEDIUM_TTC:
            return "MEDIUM"

        # Critical.
        return "URGENT"

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        detections,
        frame_width,
        frame_height,
        timestamp=None
    ):

        if timestamp is None:
            timestamp = time.time()

        assignments, unmatched_detections, unmatched_tracks = \
            self.associate(detections)

        results = []

        # ----------------------------------------------------
        # UPDATE EXISTING TRACKS
        # ----------------------------------------------------

        for det_idx, track_id in assignments.items():

            det = detections[det_idx]

            track = self.tracks[track_id]

            track.bbox = det.bbox
            track.confidence = det.confidence
            track.class_id = det.class_id
            track.class_name = det.class_name

            track.last_seen = timestamp

            # -----------------------------
            # Depth history
            # -----------------------------

            if det.relative_depth > 0:

                track.depth_history.append(
                    (
                        timestamp,
                        det.relative_depth
                    )
                )

            track.velocity = self.estimate_velocity(
                track.depth_history
            )

            track.ttc = self.calculate_ttc(
                track.velocity,
                det.relative_depth
            )

            # -----------------------------
            # Position history
            # -----------------------------

            fx, fy = footpoint(det.bbox)

            x_norm = fx / frame_width
            y_norm = fy / frame_height

            track.position_history.append(
                (
                    timestamp,
                    x_norm,
                    y_norm
                )
            )

            track.current_x = x_norm

            track.x_velocity = self.estimate_x_velocity(
                track.position_history
            )

            # -----------------------------
            # Current corridor
            # -----------------------------

            track.path_overlap = corridor_score(
                x_norm,
                y_norm
            )

            # -----------------------------
            # Predicted corridor
            # -----------------------------

            predicted_x, predicted_score = \
                predicted_corridor_score(
                    current_x=x_norm,
                    y_norm=y_norm,
                    x_velocity=track.x_velocity
                )

            track.predicted_x = predicted_x

            # Apply small anticipation margin.
            #
            # This effectively gives the system a little warning
            # before the predicted trajectory actually reaches
            # the corridor.

            if predicted_score <= 0:

                left, right = corridor_bounds(
                    y_norm
                )

                # If moving toward the corridor, expand it
                # slightly in the direction of motion.
                predicted_x_margin = predicted_x

                if x_norm < left and track.x_velocity > 0:
                    predicted_x_margin += PREDICTION_MARGIN

                elif x_norm > right and track.x_velocity < 0:
                    predicted_x_margin -= PREDICTION_MARGIN

                predicted_x_margin = np.clip(
                    predicted_x_margin,
                    0.0,
                    1.0
                )

                predicted_score = corridor_score(
                    predicted_x_margin,
                    y_norm
                )

            track.predicted_path_overlap = predicted_score

            # -----------------------------
            # Risk
            # -----------------------------

            track.risk = self.calculate_risk(
                path_overlap=track.path_overlap,
                predicted_path_overlap=track.predicted_path_overlap,
                velocity=track.velocity,
                ttc=track.ttc
            )

            results.append(track)

        # ----------------------------------------------------
        # CREATE NEW TRACKS
        # ----------------------------------------------------

        for det_idx in unmatched_detections:

            det = detections[det_idx]

            track_id = self.next_track_id

            self.next_track_id += 1

            track = ObjectState(
                track_id=track_id,
                class_id=det.class_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                last_seen=timestamp
            )

            # Initial depth sample.
            if det.relative_depth > 0:
                track.depth_history.append(
                    (
                        timestamp,
                        det.relative_depth
                    )
                )

            # Initial position.
            fx, fy = footpoint(det.bbox)

            x_norm = fx / frame_width
            y_norm = fy / frame_height

            track.position_history.append(
                (
                    timestamp,
                    x_norm,
                    y_norm
                )
            )

            track.current_x = x_norm
            track.predicted_x = x_norm

            track.path_overlap = corridor_score(
                x_norm,
                y_norm
            )

            track.predicted_path_overlap = \
                track.path_overlap

            track.risk = "NONE"

            self.tracks[track_id] = track

            results.append(track)

        # ----------------------------------------------------
        # REMOVE OLD TRACKS
        # ----------------------------------------------------

        dead_tracks = []

        for track_id in unmatched_tracks:

            track = self.tracks[track_id]

            if timestamp - track.last_seen > TRACK_TIMEOUT:
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.tracks[track_id]

        return results
