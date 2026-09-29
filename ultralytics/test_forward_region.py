import cv2
from ultralytics import YOLO

from video_ttc_benchmark import get_depth_map
from depth_geometry import ForwardRegionEstimator

VIDEO = "../test_videos/test_vid1.mp4"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

yolo = YOLO("yolo11n.pt")
estimator = ForwardRegionEstimator()

frame_no = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if frame_no % 30 == 0:
        results = yolo.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False,
            device=0,
        )

        boxes = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                boxes.append(box)

        depth = get_depth_map(frame)

        left, right, valid = estimator.estimate(
            depth,
            boxes,
        )

        print(
            f"frame={frame_no:4d} "
            f"REGION=({left:.3f}, {right:.3f}) "
            f"width={right-left:.3f} "
            f"valid={valid}"
        )

    frame_no += 1

cap.release()
