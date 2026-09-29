import cv2
from ultralytics import YOLO

from camera_bearing import CameraBearingGeometry


VIDEO = "../test_videos/test_vid1.mp4"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO}")

yolo = YOLO("yolo11n.pt")

geometry = CameraBearingGeometry(
    hfov_deg=70.0,
    collision_half_angle_deg=15.0,
)

frame_no = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if frame_no % 30 == 0:

        h, w = frame.shape[:2]

        results = yolo.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False,
            device=0,
        )

        print(f"\nframe={frame_no}")

        if results and results[0].boxes is not None:

            boxes = results[0].boxes

            for i, box in enumerate(
                boxes.xyxy.cpu().numpy()
            ):
                x1, y1, x2, y2 = box

                # Bottom-center = approximate ground contact.
                foot_x = (x1 + x2) / 2.0

                x_norm = foot_x / w

                bearing = geometry.bearing_deg(
                    x_norm
                )

                inside = (
                    abs(bearing)
                    <= geometry.collision_half_angle_deg
                )

                print(
                    f"  person {i:2d} "
                    f"x={x_norm:.3f} "
                    f"bearing={bearing:+6.2f}° "
                    f"corridor={inside}"
                )

    frame_no += 1

cap.release()
