import cv2
from vanishing_point import VanishingPointEstimator

VIDEO = "../test_videos/test_vid1.mp4"

cap = cv2.VideoCapture(VIDEO)
estimator = VanishingPointEstimator()

frame_no = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if frame_no % 30 == 0:
        x, y, valid = estimator.estimate(frame)

        print(
            f"frame={frame_no:4d} "
            f"VP=({x:.3f}, {y:.3f}) "
            f"valid={valid}"
        )

    frame_no += 1

cap.release()
