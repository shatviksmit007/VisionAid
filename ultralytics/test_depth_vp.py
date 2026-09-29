import cv2
import numpy as np

# Import the exact depth pipeline from your existing benchmark.
from video_ttc_benchmark import get_depth_map

from depth_geometry import DepthGeometryEstimator


VIDEO = "../test_videos/test_vid1.mp4"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

estimator = DepthGeometryEstimator()

frame_no = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if frame_no % 30 == 0:
        depth = get_depth_map(frame)

        vp_x, vp_y, valid = estimator.estimate(depth)

        print(
            f"frame={frame_no:4d} "
            f"VP=({vp_x:.3f}, {vp_y:.3f}) "
            f"valid={valid}"
        )

    frame_no += 1

cap.release()
