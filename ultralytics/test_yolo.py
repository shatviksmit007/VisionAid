from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=0,
    imgsz=640,
    verbose=True,
)

result = results[0]

print("\n========== DETECTIONS ==========")

for i, box in enumerate(result.boxes):
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    print(
        f"{i:2d} | "
        f"{result.names[class_id]:15s} | "
        f"confidence={confidence:.3f} | "
        f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
    )
